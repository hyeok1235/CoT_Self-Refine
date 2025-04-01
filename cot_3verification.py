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
        if do_sample == False:
            return self.model.generate(
                tokenized_chat,  # Passing the tensor directly
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=do_sample
            )
        else:
            return self.model.generate(
                tokenized_chat,  # Passing the tensor directly
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=do_sample,
                temperature=temperature
            )
    
    def generate_with_confidence(self, tokenized_input, do_sample: bool = True, temperature: float = 0.1, max_new_tokens: int = 1024):
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
        # outputs_confidence = self.generate_with_confidence(tokenized_chat)
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
        # result_pairs.append({"reasoning_path": result, "answer": intial_answer, "confidence": outputs_confidence})
        result_pairs.append({"reasoning_path": result, "answer": intial_answer})
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
            # The original feedback prompts remain unchanged.

            feedback_prompts = [
                [  # Section 1: Calculation Errors
                    {
                        "role": "system",
                        "content": (
                            "You are a mathematics expert identifying critical calculation errors. Examine the designated step for:"
                            "\n- Severe arithmetic miscalculations"
                            "\n- Fundamental algebraic errors"
                            "\n- Incorrect formula applications"
                            "\nIf you find such errors, respond with: 'Feedback: Critical calculation error detected.' Then:"
                            "1. Clearly state the mathematical law/rule violated"
                            "2. Show the incorrect vs correct calculation"
                            "3. Explain the impact on final answer"
                            "\nIf the step is correct and no critical issues are found, it's crucial to respond with: 'Feedback: No critical errors found.'"
                            "\nIf you notice minor issues or nitpicks that don't significantly impact the solution, also respond with: 'Feedback: No critical errors found.'"
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "Here is the current reasoning path for the problem:\n\n"
                            "Problem: Calculate the average score of a math test with the following scores: 85, 90, 75, and 80.\n\n"
                            "Steps:\n"
                            "Step 1: List all the test scores: 85, 90, 75, and 80.\n"
                            "Step 2: Add all the scores together: 85 + 90 + 75 + 80 = 330.\n"
                            "Step 3: Count the total number of scores, which is 4.\n"
                            "Step 4: Divide the sum by the count to get the average: 330 ÷ 4 = 82.5.\n"
                            "Feedback on this step: Step 3: Count the total number of scores, which is 4.\n"
                        )
                    },
                    {
                        "role": "assistant",
                        "content": "Feedback: No critical errors found."
                    },
                    {
                        "role": "user",
                        "content": (
                            "Here is the current reasoning path for the problem:\n\n"
                            "Problem: Max bought 3 notebooks at $4.50 each and 2 pens at $0.75 each. Calculate the total cost.\n\n"
                            "Steps:\n"
                            "Step 1: Identify the items and their costs: 3 notebooks at $4.50 each and 2 pens at $0.75 each.\n"
                            "Step 2: Calculate the cost of notebooks: 3 × $4.50 = $13.50.\n"
                            "Step 3: Calculate the cost of pens: 2 × $0.75 = $1.50.\n"
                            "Step 4: Add the costs together: $13.50 + $1.50 = $15.50.\n"
                            "Step 5: The total cost is $15.50.\n"
                            "Feedback on this step: Step 4: Add the costs together: $13.50 + $1.50 = $15.50.\n"
                        )
                    },
                    {
                        "role": "assistant",
                        "content": (
                            "Feedback: Critical calculation error detected. The correct sum of $13.50 and $1.50 is $15.00, not $15.50. "
                            "This miscalculation inflates the total cost and affects the final answer."
                        )
                    }
                ],
                [  # Section 2: Problem Interpretation Errors
                    {
                        "role": "system",
                        "content": (
                            "You are a problem-solving expert detecting fundamental misinterpretations. Flag only:"
                            "\n- Misidentification of core problem requirements"
                            "\n- Incorrect assumptions changing problem meaning"
                            "\n- Missing critical problem constraints"
                            "\nIf detected, respond with: 'Feedback: Critical misinterpretation detected.' Then:"
                            "1. Quote the misunderstood problem component"
                            "2. Explain the distorted understanding"
                            "3. Provide the accurate interpretation"
                            "\nIgnore: Alternative valid interpretations or minor phrasing disagreements."
                            "\nIf the step is correct and no critical issues are found, it's crucial to respond with: 'Feedback: No critical errors found.'"
                            "\nIf you notice minor issues or nitpicks that don't significantly impact the solution, also respond with: 'Feedback: No critical errors found.'"
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "Here is the current reasoning path for the problem:\n\n"
                            "Problem: A pizza originally costs $20 and is offered at a 25% discount. What is the sale price?\n\n"
                            "Steps:\n"
                            "Step 1: Identify the original price, which is $20.\n"
                            "Step 2: Calculate the discount amount as 25% of $20: 0.25 × $20 = $5.\n"
                            "Step 3: Add the discount to the original price: $20 + $5 = $25.\n"
                            "Step 4: Conclude that the sale price is $25.\n"
                            "Feedback on this step: Step 3: Add the discount to the original price: $20 + $5 = $25.\n"
                        )
                    },
                    {
                        "role": "assistant",
                        "content": (
                            "Feedback: Critical misinterpretation detected. The step incorrectly adds the discount to the original price instead of subtracting it. "
                            "A discount reduces the price, so the calculation should be $20 - $5 = $15."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "Here is the current reasoning path for the problem:\n\n"
                            "Problem: John has 5 apples and gives 2 to Mary. How many apples does he have left?\n\n"
                            "Steps:\n"
                            "Step 1: Start with John's initial number of apples, which is 5.\n"
                            "Step 2: Determine that John gives away 2 apples to Mary.\n"
                            "Step 3: Subtract the number of apples given away from the initial amount: 5 - 2 = 3.\n"
                            "Step 4: Conclude that John has 3 apples remaining.\n"
                            "Feedback on this step: Step 3: Subtract the number of apples given away from the initial amount: 5 - 2 = 3.\n"
                        )
                    },
                    {
                        "role": "assistant",
                        "content": "Feedback: No critical errors found."
                    }
                ],
                [  # Section 3: Logical Consistency Between Steps
                    {
                        "role": "system",
                        "content": (
                            "You are a logic expert identifying solution-breaking inconsistencies. Flag only:"
                            "\n- Contradictions with previous validated steps"
                            "\n- Invalid logical leaps (missing necessary intermediate steps)"
                            "\n- Violations of mathematical proof principles"
                            "\nIf found, respond with: 'Feedback: Critical inconsistency detected.' Then:"
                            "1. Identify the exact logical fracture point"
                            "2. Show the chain break using previous steps"
                            "3. Provide the minimum correction needed"
                            "\nIgnore: Stylistic variations in reasoning."
                            "\nIf the step is correct and no critical issues are found, it's crucial to respond with: 'Feedback: No critical errors found.'"
                            "\nIf you notice minor issues or nitpicks that don't significantly impact the solution, also respond with: 'Feedback: No critical errors found.'"
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "Here is the current reasoning path for the problem:\n\n"
                            "Problem: Calculate the area and perimeter of a rectangle with dimensions 8 cm by 5 cm.\n\n"
                            "Steps:\n"
                            "Step 1: Identify the rectangle's dimensions as 8 cm length and 5 cm width.\n"
                            "Step 2: Calculate the area using the formula: Area = length × width.\n"
                            "Step 3: Substitute the values to find the area: 8 cm × 5 cm = 40 square cm.\n"
                            "Step 4: For the perimeter, incorrectly add the dimensions: 8 cm + 5 cm = 13 cm as the perimeter.\n"
                            "Step 5: Conclude that the area is 40 square cm and the perimeter is 13 cm.\n"
                            "Feedback on this step: Step 4: For the perimeter, incorrectly add the dimensions: 8 cm + 5 cm = 13 cm as the perimeter.\n"
                        )
                    },
                    {
                        "role": "assistant",
                        "content": (
                            "Feedback: Critical inconsistency detected. The perimeter of a rectangle is calculated by adding twice the length and twice the width (2l + 2w), "
                            "not by simply adding the dimensions. The correct perimeter is 2 × 8 + 2 × 5 = 16 + 10 = 26 cm, not 13 cm."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "Here is the current reasoning path for the problem:\n\n"
                            "Problem: Calculate the area of a triangle with base 10 cm and height 8 cm.\n\n"
                            "Steps:\n"
                            "Step 1: Identify the triangle's base as 10 cm and height as 8 cm.\n"
                            "Step 2: Recall the formula for the area of a triangle: Area = (1/2) × base × height.\n"
                            "Step 3: Substitute the values into the formula: Area = (1/2) × 10 cm × 8 cm.\n"
                            "Step 4: Calculate: Area = (1/2) × 80 square cm = 40 square cm.\n"
                            "Step 5: Conclude that the triangle's area is 40 square cm.\n"
                            "Feedback on this step: Step 4: Calculate: Area = (1/2) × 80 square cm = 40 square cm.\n"
                        )
                    },
                    {
                        "role": "assistant",
                        "content": "Feedback: No critical errors found."
                    }
                ]
            ]

            feedback_responses = []

            for prompt in feedback_prompts:
                feedback_prompt = prompt + [
                    {
                        "role": "user",
                        "content": (
                            f"Here is the current reasoning path for the problem:\n\n"
                            f"Problem: {problem}\n\n"
                            f"Steps:\n" + "\n".join(f"<STEP>{step}</STEP>" for step in current_step) +
                            f"Feedback on this step: <STEP>{current_step[step_index - 1]}</STEP>\n"
                        )
                    },
                    {"role": "assistant", "content": "Feedback:"}
                ]

                tokenized_chat = self.tokenize_chat(feedback_prompt)
                output = self.generate_output(tokenized_chat)
                decoded_feedback = self.decode_output(output, False).split("Feedback:")[-1].strip().split("<|eot_id|>")[0].strip()

                if "no critical errors found" in decoded_feedback.lower():
                    continue
                feedback_responses.append(decoded_feedback)
            
            if feedback_responses == []:
                pass_count += 1
                step_index += 1
                continue

            # Define three different verification system prompts.
            verification_system_prompts = [
                {
                    "role": "system",
                    "content": (
                        "You are verifying feedback about calculation errors. Determine if the feedback:"
                        "\n1. Correctly identifies a genuine calculation error (arithmetic, algebraic, or mathematical manipulation)"
                        "\n2. Provides the correct mathematical solution"
                        "\n3. Does not falsely flag correct calculations as errors"
                        "\nIf the feedback accurately identifies a real calculation error, reply with: 'The feedback is valid.'"
                        "\nOtherwise, reply with: 'The feedback is invalid.'"
                    )
                },
                {
                    "role": "system",
                    "content": (
                        "You are verifying feedback about problem interpretation. Determine if the feedback:"
                        "\n1. Correctly identifies a genuine misunderstanding of what the problem is asking"
                        "\n2. Accurately explains how the step misinterprets the problem requirements"
                        "\n3. Does not falsely flag correct interpretations as misunderstandings"
                        "\nIf the feedback accurately identifies a real problem interpretation error, reply with: 'The feedback is valid.'"
                        "\nOtherwise, reply with: 'The feedback is invalid.'."
                    )
                },
                {
                    "role": "system",
                    "content": (
                        "You are verifying feedback about logical consistency between steps. Determine if the feedback:"
                        "\n1. Correctly identifies a genuine logical inconsistency with previous steps"
                        "\n2. Accurately explains why the step doesn't align with the solution approach"
                        "\n3. Does not falsely flag logically consistent steps as errors"
                        "\nIf the feedback accurately identifies a real logical inconsistency, reply with: 'The feedback is valid.'"
                        "\nOtherwise, reply with: 'The feedback is invalid.'"
                    )
                },
            ]

            valid_feedbacks = []  # to store only the approved feedback responses

            # For every feedback response, run it through each verification system prompt.
            for i, fb in enumerate(feedback_responses.copy()):
                feedback_verification_prompt = [
                    verification_system_prompts[i],
                    {
                        "role": "user",
                        "content": (
                            "Now verify the feedback provided below."
                            f"Problem: {problem}\n\n"
                            f"Previously validated steps: " + "\n".join(f"<STEP>{step}</STEP>" for step in current_step[:step_index - 1]) + "\n\n"
                            f"Designated Step Under Review: <STEP>{current_step[step_index - 1]}</STEP>\n\n"
                            f"Forward-looking Step: <STEP>{current_step[-1]}</STEP>\n\n"
                            f"Feedback Provided: {fb}\n\n"
                            "Evaluate the feedback and respond as instructed."
                        )
                    },
                    {
                    "role": "assistant",
                    "content": "The feedback is"
                    }
                ]
                tokenized_verification = self.tokenize_chat(feedback_verification_prompt)
                verification_output = self.generate_output(tokenized_verification)
                verification_result = self.decode_output(verification_output, False).split("Evaluate the feedback and respond as instructed.")[-1].strip()
                if "feedback is valid" in verification_result.lower():
                    valid_feedbacks.append(fb.split("<|eot_id|>")[0])


            # Check if final_feedback indicates no additional changes required
            # If no valid feedback remains, skip refinement
            if valid_feedbacks == []:
                reject_count += 1
                step_index += 1
                continue
            
            feedback = "; ".join(valid_feedbacks)
            # Append final feedback result
            result_pairs.append({"feedback": feedback})
            # print(f"Feedback: {feedback}")
            # print("\n\n******************\n\n")

            # print("refining")

            # --- REFINE PROMPT ---
            # Now instruct the model to refine the step immediately preceding the most recent step
            refine_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are refining a problem-solving step based on expert feedback. "
                        "Your task is to refine **only** the specific step identified in the feedback, "
                        "while keeping all previously validated steps unchanged. "
                        "Ensure that your correction strictly addresses the identified error **without modifying correct logic**. "
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Refine specific step only, based on the feedback provided below.\n\n"
                        "**Problem:** " + problem + "\n\n"
                        "**Reasoning step-by-step (before refinement):**\n"
                        + "\n".join(f"<STEP>{step}</STEP>" for step in current_step[:forward_step_index]) +
                        f"Step to refine : <STEP>{current_step[step_index - 1]}</STEP>"
                        f"\n\n**Feedback on the step to refine : {feedback}\n\n"
                        "**Provide the corrected step below without modifying other correct steps.**"
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "**Refined Reasoning Step-by-Step:**\n"
                        + "\n".join(f"<STEP>{step}</STEP>" for step in current_step[:step_index - 1]) # Keep previous steps unchanged
                        + "<STEP>"
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
                    "content": "Solve this math problem step by step. Problem : \n" + problem,
                },
                {
                    "role": "assistant",
                    "content": "Reasoning step-by-step:\n" + previous_steps
                }
            ]

            continuation_tokenized_chat = self.tokenize_chat(continuation_prompt)
            continuation_output = self.generate_output(continuation_tokenized_chat)
            # outputs_confidence = self.generate_with_confidence(continuation_tokenized_chat)

            continued_result = self.decode_output(continuation_output, False).split("Reasoning step-by-step:")[-1]
            continued_steps = re.findall(r"<STEP>(.*?)</STEP>", continued_result, re.DOTALL)
            continued_answer = continued_result.split("<ANSWER>")[-1].split("</ANSWER>")[0].strip()
            result = "\n".join(f"<STEP>{step}</STEP>" for step in continued_steps) + f"<ANSWER>{continued_answer}</ANSWER>"

            # print("continued answer confidence : ", outputs_confidence, ", Answer :", continued_answer)
            # print("--------------------")

            # result_pairs.append({"reasoning_path": continued_result, "answer": continued_answer, "confidence": outputs_confidence})
            result_pairs.append({"reasoning_path": continued_result, "answer": continued_answer})
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
        log_file = "results_divide_feedback.csv"
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

        best_result = processed_result[-1]  # The last element is the best result
        return [intial_answer, best_result['answer'], reject_count, len(steps) - pass_count, len(steps)] # reject된 피드백의 개수, valid한 피드백의 개수, 전체 step의 개수


# Usage Example
if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    solver = MathProblemSolver(model_name=model_name)

    few_shot_examples_GSM8K = [
      {
        "role": "system",
        "content": "You are a helpful assistant that solves math problems step by step. "
        "For every reasoning step, enclose your output exactly within the <STEP> and </STEP> tags. "
        "Your final answer value must be provided enclosed within the <ANSWER> and </ANSWER> tags. "
        "IMPORTANT: Do not include any additional text outside these tags."
      },
      {
        "role": "user",
        "content": ("Solve this math problem step by step. Problem : \n"
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?")
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
        "content": ("Solve this math problem step by step. Problem : \n"
                    "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?"
                )
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
    # Answer 2 = 31, 25 + 5+ 1
    problem2 = "Ram uses a lot of pens. He discovered that he can save money by mixing the ink from five empty pens to make one full pen. If he buys 25 pens and then uses them to make new pens when the ink runs low, how many total pens does he get to have?"
    # Answer 3 = 60
    problem3 = "Rita hand-picks Junebugs off of her plants every summer.  On Monday, she removed 39 Junebugs.  On both Tuesday and Wednesday, she removed twice as many Junebugs as she did on Monday.  Thursday she removed 48 and on Friday she removed 57.  What is the average number of Junebugs that she removes per day?"
    solution = solver.solve_problem(problem2, few_shot_examples_GSM8K)
    print(solution)

