import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from typing import Optional, List, Dict
import torch
import re

class MathProblemSolver:
    def __init__(self, model_name: str, cuda_devices: str = "0,1", quantization_bits: Optional[int] = 4):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

        # Set CUDA_VISIBLE_DEVICES
        self.set_cuda_devices(cuda_devices)

        # Initialize tokenizer and model
        self.initialize_model(quantization_bits)

    def set_cuda_devices(self, devices: str):
        """
        Set CUDA_VISIBLE_DEVICES environment variable to select GPUs.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = devices

    def set_quantization(self, quantization_bits: Optional[int]) -> Optional[BitsAndBytesConfig]:
        """
        Configure quantization settings based on the specified number of bits.
        """
        if quantization_bits in [4, 8]:
            bnb_config = BitsAndBytesConfig()
            if quantization_bits == 4:
                bnb_config.load_in_4bit = True
                bnb_config.bnb_4bit_quant_type = "nf4"
                bnb_config.bnb_4bit_use_double_quant = True
                bnb_config.bnb_4bit_compute_dtype = torch.bfloat16
            elif quantization_bits == 8:
                bnb_config.load_in_8bit = True
            return bnb_config
        return None

    def initialize_model(self, quantization_bits: Optional[int]):
        """
        Initialize the model and tokenizer with the given quantization settings.
        """
        model_config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            config=model_config,
            quantization_config=self.set_quantization(quantization_bits),
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model.generation_config.temperature=None
        self.model.generation_config.top_p=None

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_chat(self, chat_history: List[Dict], max_length: int = 2048):
        """
        Tokenize the chat history with padding and truncation.
        """
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=True,
            continue_final_message=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

    def generate_output(self, tokenized_chat, do_sample: bool = False, temperature: float = 1.0, max_new_tokens: int = 1024):
        # Move tokenized_chat tensor to the correct device
        tokenized_chat = tokenized_chat.to(self.model.device)
        return self.model.generate(
            tokenized_chat,  # Passing the tensor directly
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=do_sample,
            temperature=temperature
        )


    def decode_output(self, outputs, skip_sepcial_tokens):
        """
        Decode the generated output.
        """
        return self.tokenizer.decode(outputs[0], skip_special_tokens=skip_sepcial_tokens)

    def solve_problem(self, problem: str, few_shot_examples: List[Dict]):
        """
        Solve a math problem using a step-by-step reasoning approach.
        """
        problem_initial_prompt = [
            {
                "role": "user",
                "content": "Solve this math problem step by step. Problem : \n" + problem,
            },
            {
                "role": "assistant",
                "content": """
Reasoning step-by-step:
<STEP>"""
            }
        ]

        chat_history = few_shot_examples + problem_initial_prompt
        tokenized_chat = self.tokenize_chat(chat_history)

        outputs = self.generate_output(tokenized_chat)
        result = self.decode_output(outputs, False).split("Reasoning step-by-step:")[-1]

        step_index = 1
        forward_look = 1
        intial_answer = result.split("<ANSWER>")[-1].split("</ANSWER>")[0].strip()
        pass_count = 0
        reject_count = 0
        # print(f"Initial Solution: {result}")
        # print("\n\n******************\n\n")

        # False -> CoT
        # current_step_index <= 5 -> 5 iter
        # current_step_index <= 10 -> 10 iter
        # True -> LLM's judgement
        while True:
            forward_step_index = step_index + forward_look
            steps = re.findall(r"<STEP>(.*?)</STEP>", result, re.DOTALL)
            if step_index > len(steps):
                break

            # if forward_step_index is larger than the number of steps, we give full steps. Otherwise, we give steps up to forward_step_index
            current_step = steps[:forward_step_index] if forward_step_index <= len(steps) else steps

            # --- FEEDBACK PROMPT ---
            # Now we instruct the model: “Give feedback on the step immediately preceding the most recent step only”
            feedback_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are a math expert reviewing problem-solving steps. "
                        "First, the user will provide all the reasoning steps for a problem. "
                        "Then, they will extract a **specific step** for you to evaluate. "
                        "Your response should follow these guidelines:\n"
                        "1. Check if the step is **logically consistent** with the reasoning steps.\n"
                        "2. If the step is **mathematically and logically correct**, respond with: 'Step is correct.'\n"
                        "3. If there is an **error**, provide a concise explanation of the mistake and suggest a correction.\n"
                        "4. Avoid unnecessary suggestions or changes to correct steps.\n"
                        "5. Keep your feedback factual, clear, and precise. \n"
                        "6. Check if the step demonstrates a clear understanding of the problem.\n"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Here is the full reasoning for the problem:\n\n"
                        "Problem: " + problem + "\n\n"
                        "Steps:\n" + "\n".join(f"<STEP>{step}</STEP>" for step in current_step)
                    )
                },
                {
                    "role": "assistant",
                    "content": "Yes, I got it. What step should I feedback on?"
                },
                {
                    "role": "user",
                    "content": "<STEP>" + current_step[step_index - 1] + "</STEP>"
                },
                {
                    "role": "assistant",
                    "content": "Feedback:\n"
                }
            ]

            feedback_tokenized_chat = self.tokenize_chat(feedback_prompt)
            feedback_output1 = self.generate_output(feedback_tokenized_chat, True, 0.8)
            feedback_output2 = self.generate_output(feedback_tokenized_chat, True, 0.8)
            feedback_output3 = self.generate_output(feedback_tokenized_chat, True, 0.8)

            feedback1 = self.decode_output(feedback_output1, False)\
                        .split("Feedback:")[-1].strip()
            feedback2 = self.decode_output(feedback_output2, False)\
                        .split("Feedback:")[-1].strip()
            feedback3 = self.decode_output(feedback_output3, False)\
                        .split("Feedback:")[-1].strip()
            feedback_list = [feedback1, feedback2, feedback3]
            # print(feedback_list)
            # print("\n\n******************\n\n")
            
            # Decide which feedback to use
            feedback_choose_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are evaluating three different feedback responses for a specific step in a math problem. "
                        "Your task is to analyze each feedback response carefully and select the best one. "
                        "Follow these guidelines:\n"
                        "1. **Consider the problem statement and full reasoning steps** before assessing feedback.\n"
                        "2. **Ensure the selected feedback correctly identifies errors (if any) or confirms correctness.**\n"
                        "3. **Check if the feedback is clear, precise, and provides an accurate correction (if needed).**\n"
                        "4. **Reject feedback that is misleading, vague, or introduces unnecessary modifications.**\n"
                        "5. Provide a structured comparison of all three feedback options before making a final choice.\n"
                        "6. Conclude your response **strictly** in this format: 'The best choice is 1.', 'The best choice is 2.', or 'The best choice is 3.'."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"**Problem:** {problem}\n\n"
                        f"**Full Reasoning Steps:**\n" + "\n".join(f"<STEP>{step}</STEP>" for step in current_step) +
                        f"\n\n**Step under review:**\n<STEP>{current_step[step_index - 1]}</STEP>\n\n"
                        "Now, evaluate the following three feedback responses and **strictly select the best one**:\n\n"
                        f"**Feedback 1:** {feedback1}\n\n"
                        f"**Feedback 2:** {feedback2}\n\n"
                        f"**Feedback 3:** {feedback3}\n\n"
                        "Provide a structured analysis of each feedback response and conclude **only** with one of the following statements:\n"
                        "'The best choice is 1.'\n"
                        "'The best choice is 2.'\n"
                        "'The best choice is 3.'"
                    )
                },
                {
                    "role": "assistant",
                    "content": "Okay, let me evaluate the feedback responses."
                }
            ]
            feedback_choose_chat = self.tokenize_chat(feedback_choose_prompt)
            feedback_choose_output = self.generate_output(feedback_choose_chat)
            # feedback_choose_result = self.decode_output(feedback_choose_output, False).split("The best choice is ")[-1].strip()[0]
            # feedback = feedback_list[int(feedback_choose_result) - 1]

            feedback_choose_result = self.decode_output(feedback_choose_output, True).split("assistant\n")[-1]
            feedback_choose_result = feedback_choose_result.split("best choice is ")[-1].strip()

            if "1" in feedback_choose_result:
                feedback = feedback_list[0]
            elif "2" in feedback_choose_result:
                feedback = feedback_list[1]
            else:
                feedback = feedback_list[2]

            print(f"Feedback: {feedback}")
            print("\n\n******************\n\n")

            if "step is correct" in feedback.lower():
                pass_count += 1
                step_index += 1
                continue
            print("fixing the step...")
            # Decide which feedback to use
            feedback_verification_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert in mathematical reasoning and problem-solving verification. "
                        "Your task is to evaluate the quality of feedback provided for a specific problem-solving step. "
                        "You must determine if the feedback correctly identifies errors, provides clear and precise guidance, "
                        "and does not modify correct parts of the step. Your analysis should be based on the problem statement, "
                        "previously validated steps, and the designated step under review."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Analyze the feedback provided below in the context of the problem and the previously validated steps. "
                        "**Problem:**\n" + problem + "\n\n"
                        "**Previously Validated Steps:**\n" + "\n".join(f"<STEP>{step}</STEP>" for step in current_step[:step_index - 1]) + "\n\n"
                        "**Designated Step Under Review:**\n<STEP>" + current_step[step_index - 1] + "</STEP>\n\n"
                        "**Feedback Provided:**\n" + feedback + "\n\n"
                        "Verify the feedback, and then conclude on the final line with exactly one of these statements:\n"
                        "'The feedback is valid.' or 'The feedback is invalid.'"
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "Analysis:\n"
                    )
                }
            ]

            feedback_choose_chat = self.tokenize_chat(feedback_verification_prompt)
            feedback_choose_output = self.generate_output(feedback_choose_chat)
            feedback_choose_result = self.decode_output(feedback_choose_output, False).split("The feedback is ")[-1].strip()
            # if feedback is invalid just continue to the next step. If feedback is valid, use it to refine the solution
            if "invalid" in feedback_choose_result:
                reject_count += 1
                step_index += 1
                continue
            
            # --- REFINE PROMPT ---
            # Now instruct the model to refine the step immediately preceding the most recent step
            refine_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are refining a problem-solving step based on expert feedback. "
                        "Your task is to correct **only** the specific step identified in the feedback, "
                        "while keeping all previously validated steps unchanged. "
                        "Ensure that your correction strictly addresses the identified error without modifying correct logic. "
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Refine the specific step only, based on the feedback provided below.\n\n"
                        "**Problem:** " + problem + "\n\n"
                        f"**Full Reasoning Steps:**\n" + "\n".join(f"<STEP>{step}</STEP>" for step in current_step) +
                        f"\n\n**Step under review:**\n<STEP>{current_step[step_index - 1]}</STEP>\n\n"
                        f"**Feedback:**\n{feedback}\n\n"
                        "**Provide the corrected step below without modifying other correct steps.**"
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "Refined Reasoning Step-by-Step:\n"
                        + "\n".join(f"<STEP>{step}</STEP>" for step in current_step[:step_index - 1]) +  # Keep previous steps unchanged
                        "<STEP>"  # This signals where the refined step will be inserted
                    )
                }
            ]

            # print(f"Refine Prompt: {refine_prompt}")

            refine_tokenized_chat = self.tokenize_chat(refine_prompt)
            refine_output = self.generate_output(refine_tokenized_chat)

            refined_result = self.decode_output(refine_output, False).split("Refined Reasoning Step-by-Step:\n")[-1]
            refined_steps = re.findall(r"<STEP>(.*?)</STEP>", refined_result, re.DOTALL)
            # print(f"Refined Steps: {refined_steps}")
            # print("\n\n******************\n\n")
            previous_steps = "\n".join(f"<STEP>{step}</STEP>" for step in refined_steps[:forward_step_index - 1])

            # --- CONTINUATION PROMPT ---
            # This now uses the refined steps up to current_step_index (i.e. n+1 steps)
            continuation_prompt = few_shot_examples + [
                {
                    "role": "user",
                    "content": "Follow the format of the previous solutions. Solve this math problem step by step. Problem : \n" + problem,
                },
                {
                    "role": "assistant",
                    "content": "Reasoning step-by-step:\n" + previous_steps
                }
            ]

            continuation_tokenized_chat = self.tokenize_chat(continuation_prompt)
            continuation_output = self.generate_output(continuation_tokenized_chat)

            continued_result = self.decode_output(continuation_output, False).split("Reasoning step-by-step:")[-1]
            continued_steps = re.findall(r"<STEP>(.*?)</STEP>", continued_result, re.DOTALL)
            continued_answer = continued_result.split("<ANSWER>")[-1].split("</ANSWER>")[0].strip()
            result = "\n".join(f"<STEP>{step}</STEP>" for step in continued_steps) + f"<ANSWER>{continued_answer}</ANSWER>"
            step_index += 1
            print(f"Refined Solution: {result}")
            print("\n\n******************\n\n")
        return [intial_answer, result.split("<ANSWER>")[-1].split("</ANSWER>")[0].strip(), pass_count, len(steps) - reject_count, len(steps)]


# Usage Example
if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    solver = MathProblemSolver(model_name=model_name)

    few_shot_examples_GSM8K = [
        {
        "role": "system",
        "content": "You are a helpful assistant that solves math problems step by step. Follow the examples provided and solve the given problem logically.\n\n"
"Format each step as follows:\n"
"<STEP>\n"
"Your step-by-step reasoning goes here.\n"
"</STEP>\n\n"
"Provide the final answer in this format:\n"
"<ANSWER>\n"
"Final answer here.\n"
"</ANSWER>"
        },
        {
            "role": "user",
            "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
        },
        {
            "role": "assistant",
            "content": """
<STEP>
Natalia sold half as many clips in May as in April. To calculate the number of clips sold in May, we divide the number sold in April by 2:
48 / 2 = 24.
</STEP>
<STEP>
To find the total number of clips sold in April and May, we add the clips sold in April to the clips sold in May:
48 + 24 = 72.
</STEP>
<ANSWER>
72
</ANSWER>
"""
        },
        {
            "role": "user",
            "content": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?"
        },
        {
            "role": "assistant",
            "content": """
<STEP>
Initially, Betty has only half the money needed for the wallet. To calculate this amount, divide the wallet's cost by 2:
100 / 2 = 50.
</STEP>
<STEP>
Betty's grandparents gave her twice the amount her parents gave her. Since her parents gave her $15, her grandparents gave her:
15 * 2 = 30.
</STEP>
<STEP>
Now, calculate the total money Betty has received:
50 (Betty's initial amount) + 15 (from parents) + 30 (from grandparents) = 95.
</STEP>
<STEP>
Finally, calculate how much more Betty needs by subtracting the total money she has from the wallet's cost:
100 - 95 = 5.
</STEP>
<ANSWER>
5
</ANSWER>
"""
        }
    ]


    problem = "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?"
    solution = solver.solve_problem(problem, few_shot_examples_GSM8K)
    print(solution)
