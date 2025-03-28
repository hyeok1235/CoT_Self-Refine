import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from typing import Optional, List, Dict
import torch
import torch.nn.functional as F
import re
import csv

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

    
    def generate_output(self, tokenized_chat, do_sample: bool = False, temperature: float = 0.0, max_new_tokens: int = 1024):
        # Move tokenized_chat tensor to the correct device
        tokenized_chat = tokenized_chat.to(self.model.device)
        return self.model.generate(
            tokenized_chat,  # Passing the tensor directly
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=do_sample,
            temperature=temperature
        )
    
    def generate_with_confidence(self, tokenized_input, do_sample: bool = False, temperature: float = 0.0, max_new_tokens: int = 1024):
        # return_dict_in_generate와 output_scores를 True로 설정합니다.
        outputs = self.model.generate(
            tokenized_input.to(self.model.device),
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=do_sample,
            temperature=temperature,
            output_scores=True,
            return_dict_in_generate=True,
        )
        
        generated_tokens = outputs.sequences  # (batch_size, total_length)
        all_log_probs = []
        
        for i, score in enumerate(outputs.scores):
            # generated_tokens[:, i] 는 각 토큰의 인덱스 (프롬프트 없이 바로 생성된 경우)
            token_indices = generated_tokens[:, i]
            log_probs = F.log_softmax(score, dim=-1)  # (batch_size, vocab_size)
            chosen_log_probs = log_probs.gather(dim=-1, index=token_indices.unsqueeze(-1)).squeeze(-1)
            all_log_probs.append(chosen_log_probs)
        
        # (batch_size, num_generated) 텐서로 만듭니다.
        all_log_probs = torch.stack(all_log_probs, dim=1)
        avg_log_probs = all_log_probs.mean(dim=1)  # (batch_size,)
        
        # 기존: perplexity = exp(-avg_log_probs), confidence = 1/perplexity
        # 새로운 접근: 로그 perplexity 혹은 -avg_log_probs를 "confidence score"로 사용
        # 예를 들어, higher (-avg_log_probs) means higher confidence.
        log_confidence = -avg_log_probs  # (batch_size,)
        # Alternatively, we can compute perplexity and then take its log.
        perplexity = torch.exp(-avg_log_probs)
        
        # Return the average log confidence and perplexity for reference.
        return log_confidence.mean().item()

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
        outputs_confidence = self.generate_with_confidence(tokenized_chat)
        result = self.decode_output(outputs, False).split("Reasoning step-by-step:")[-1]

        step_index = 1
        forward_look = 1
        pass_count = 0
        reject_count = 0
        result_pairs = []

        intial_answer = result.split("<ANSWER>")[-1].split("</ANSWER>")[0].strip()

        # print("initial answer confidence : ", outputs_confidence, ", Answer :", intial_answer)
        # print("--------------------")

        result_pairs.append({"question" : problem})
        result_pairs.append({"reasoning_path": result, "answer": intial_answer, "confidence": outputs_confidence})
        # print(f"Initial Solution: {result}")
        # print("\n\n******************\n\n")

        # False -> CoT
        # step_index <= 5 -> 5 iter
        # step_index <= 10 -> 10 iter
        # True -> LLM's judgement
        while step_index <= 20:
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
                        "2. If the step is **mathematically and logically correct**, start with: 'Step is correct.'\n"
                        "3. If there is an **error**, start with 'Step is incorrect'. Then provide a concise explanation of the mistake and suggest a correction.\n"
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
            feedback_output1 = self.generate_output(feedback_tokenized_chat, True, 0.2)
            feedback_output2 = self.generate_output(feedback_tokenized_chat, True, 0.2)
            feedback_output3 = self.generate_output(feedback_tokenized_chat, True, 0.2)

            feedback1 = self.decode_output(feedback_output1, False)\
                        .split("Feedback:")[-1].strip()
            feedback2 = self.decode_output(feedback_output2, False)\
                        .split("Feedback:")[-1].strip()
            feedback3 = self.decode_output(feedback_output3, False)\
                        .split("Feedback:")[-1].strip()

            # feedback_list = [feedback1, feedback2, feedback3]
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
                feedback = feedback1
            elif "2" in feedback_choose_result:
                feedback = feedback2
            else:
                feedback = feedback3

            # print(f"Feedback: {feedback}")
            # print("\n\n******************\n\n")
            
            if "step is incorrect" not in feedback.lower():
                pass_count += 1
                step_index += 1
                # print("## moving on ##")
                continue

            result_pairs.append({"feedback": feedback})
            # print(f"Feedback: {feedback}")
            # print("\n\n******************\n\n")

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
                        "**Reasoning paths:**\n" + "\n".join(f"<STEP>{step}</STEP>" for step in current_step) + "\n\n"
                        "**Designated Step Under Review:**\n<STEP>" + current_step[-1] + "</STEP>\n\n"
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
            feedback_choose_decoded = self.decode_output(feedback_choose_output, False)
            # print("feedback_choose_decoded : ", feedback_choose_decoded)
            # print("\n\n******************\n\n")
            feedback_choose_result = feedback_choose_decoded.split("Analysis:")[-1].strip()
            # if feedback is invalid just continue to the next step. If feedback is valid, use it to refine the solution
            if "the feedback is invalid" in feedback_choose_result.lower():
                reject_count += 1
                step_index += 1
                continue

            # print("refining")

            # --- REFINE PROMPT ---
            # Now instruct the model to refine the step immediately preceding the most recent step
            refine_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are refining a problem-solving step based on expert feedback. "
                        "Your task is to correct **only** the specific step identified in the feedback, "
                        "while keeping all previously validated steps unchanged. "
                        "Ensure that your correction strictly addresses the identified error **without modifying correct logic**. "
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Refine last step only, based on the feedback provided below.\n\n"
                        "**Problem:** " + problem + "\n\n"
                        "**Reasoning step-by-step (before refinement):**\n"
                        + "\n".join(f"<STEP>{step}</STEP>" for step in current_step[:step_index]) +
                        f"\n\n**Feedback on last step :** {feedback}\n\n"
                        "**Provide the corrected step below without modifying other correct steps.**"
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "**Refined Reasoning Step-by-Step:**\n"
                        + "\n".join(f"<STEP>{step}</STEP>" for step in current_step[:step_index - 1]) +  # Keep previous steps unchanged
                        "<STEP>"  # This signals where the refined step will be inserted
                    )
                }
            ]


            refine_tokenized_chat = self.tokenize_chat(refine_prompt)
            refine_output = self.generate_output(refine_tokenized_chat)

            refined_result = self.decode_output(refine_output, False).split("**Refined Reasoning Step-by-Step:**\n")[-1]
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
                    "content": "Reasoning step-by-step:\n" + previous_steps + "<ANSWER>"
                }
            ]

            continuation_tokenized_chat = self.tokenize_chat(continuation_prompt)
            continuation_output = self.generate_output(continuation_tokenized_chat)
            outputs_confidence = self.generate_with_confidence(continuation_tokenized_chat)

            continued_result = self.decode_output(continuation_output, False).split("Reasoning step-by-step:")[-1]
            continued_steps = re.findall(r"<STEP>(.*?)</STEP>", continued_result, re.DOTALL)
            continued_answer = continued_result.split("<ANSWER>")[-1].split("</ANSWER>")[0].strip()
            result = "\n".join(f"<STEP>{step}</STEP>" for step in continued_steps) + f"<ANSWER>{continued_answer}</ANSWER>"

            # print("continued answer confidence : ", outputs_confidence, ", Answer :", continued_answer)
            # print("--------------------")

            result_pairs.append({"reasoning_path": continued_result, "answer": continued_answer, "confidence": outputs_confidence})
            step_index += 1
            # print(f"Refined Solution: {result}")
            # print("\n\n******************\n\n")
        
        # Step 1: Remove all '\n' characters in result_pairs
        cleaned_result_pairs = [
            {
                key: (value.replace("\n", " ") if isinstance(value, str) else value)
                for key, value in candidate.items()
            }
            for candidate in result_pairs
        ]

        # print(cleaned_result_pairs)

        # Helper function to clean text (remove newline and carriage return characters)
        def clean_text(val):
            if isinstance(val, str):
                return val.replace("\n", " ").replace("\r", " ")
            return val

        # Helper function to flatten a dictionary into a "key: value" string.
        def flatten_dict(d):
            # Sort keys for consistency
            return ", ".join(f"{k}: {clean_text(v)}" for k, v in sorted(d.items()))
        
        # Flatten each dictionary in the list.
        flattened_data = [flatten_dict(d) for d in cleaned_result_pairs]

        # The entire list will be saved as one row in the CSV file.
        # The columns will be named "0", "1", "2", ... corresponding to each dictionary.
        num_columns = len(flattened_data)
        header = [str(i) for i in range(num_columns)]

        # Write to CSV file in append mode.
        log_file = "results_no_veri.csv"
        write_header = not os.path.exists(log_file) or os.stat(log_file).st_size == 0

        with open(log_file, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(header)
            writer.writerow(flattened_data)

        # Step 2: Extract only the elements that contain the "answer" field
        processed_result = [
            candidate for candidate in cleaned_result_pairs if "answer" in candidate
        ]

        best_result = max(processed_result, key=lambda x: x["confidence"])
        return [intial_answer, best_result['answer'], reject_count, len(steps) - pass_count, len(steps)] # reject된 피드백의 개수, valid한 피드백의 개수, 전체 step의 개수


# Usage Example
if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    solver = MathProblemSolver(model_name=model_name)

    few_shot_examples_GSM8K = [
      {
        "role": "system",
        "content": "You are a helpful assistant that solves math problems step by step. Follow the examples provided and solve the given problem logically."
      },
      {
        "role": "user",
        "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
      },
      {
        "role": "assistant",
        "content": """
Reasoning step-by-step:
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
Reasoning step-by-step:
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

    # Answer = 70,000, 80000 * 2.5 - 80000 - 50000
    problem = "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?"
    problem2 = "Ram uses a lot of pens. He discovered that he can save money by mixing the ink from five empty pens to make one full pen. If he buys 25 pens and then uses them to make new pens when the ink runs low, how many total pens does he get to have?"
    solution = solver.solve_problem(problem, few_shot_examples_GSM8K)
    print(solution)

