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
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

    def generate_output(self, tokenized_chat, max_new_tokens: int = 1024):
        # Since tokenized_chat is a tensor, move it to the correct device.
        tokenized_chat = tokenized_chat.to(self.model.device)
        return self.model.generate(
            tokenized_chat,  # Passing the tensor directly
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False
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
                "content": problem,
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
        result = self.decode_output(outputs, False).split("<STEP><|eot_id|><|start_header_id|>assistant<|end_header_id|>")[-1]

        current_step_index = 1

        # False -> CoT
        # current_step_index <= 5 -> 5 iter
        # current_step_index <= 10 -> 10 iter
        while current_step_index <= 5:
            steps = re.findall(r"<STEP>(.*?)</STEP>", result, re.DOTALL)
            if current_step_index > len(steps):
                break

            current_step = steps[:current_step_index]
            feedback_prompt = [
                {
                    "role": "user",
                    "content": "Feedback on the most recent step only. Go through the step thoroughly and check if there are any errors or incorrect approaches.\n\n"
                    + "Problem: " + problem
                    + "\n\nReasoning step-by-step:\n" + "\n".join(f"<STEP>{step}</STEP>" for step in current_step)
                },
                {
                    "role": "assistant",
                    "content": "Feedback : \n"
                }
            ]

            feedback_tokenized_chat = self.tokenize_chat(feedback_prompt)
            feedback_output = self.generate_output(feedback_tokenized_chat)

            feedback = self.decode_output(feedback_output, False).split("Feedback :<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[-1].strip()

            refine_prompt = few_shot_examples + [
                {
                    "role": "user",
                    "content": "Refine the most recent reasoning step only, based on the feedback provided. Ensure the logic is clear and fits with the previous steps.\n\n"
                    + "Problem: " + problem
                    + "\n\nReasoning step-by-step:\n" + "\n".join(f"<STEP>{step}</STEP>" for step in current_step)
                    + "\n\nFeedback: " + feedback
                },
                {
                    "role": "assistant",
                    "content": "Problem: " + problem
                    + "\n\nReasoning step-by-step:\n" + "\n".join(f"<STEP>{step}</STEP>" for step in current_step[:-1])
                    + "<STEP>\n"
                }
            ]

            refine_tokenized_chat = self.tokenize_chat(refine_prompt)
            refine_output = self.generate_output(refine_tokenized_chat)

            refined_result = self.decode_output(refine_output, True).split("Reasoning step-by-step:")[-1].replace("assistant", "")
            refined_steps = re.findall(r"<STEP>(.*?)</STEP>", refined_result, re.DOTALL)

            continuation_prompt = few_shot_examples + [
                {
                    "role": "user",
                    "content": problem,
                },
                {
                    "role": "assistant",
                    "content": "Reasoning step-by-step:\n" + "\n".join(f"<STEP>{step}</STEP>" for step in refined_steps[:current_step_index])
                    + "\n<STEP>\n"
                }
            ]

            continuation_tokenized_chat = self.tokenize_chat(continuation_prompt)
            continuation_output = self.generate_output(continuation_tokenized_chat)

            result = self.decode_output(continuation_output, True).replace("assistant", "").strip()
            current_step_index += 1
        return result.split("<ANSWER>")[-1].replace("</ANSWER>", "").strip()

# Usage Example
if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    solver = MathProblemSolver(model_name=model_name)

    few_shot_examples_default = [
      {
        "role": "system",
        "content": "You are a helpful assistant that solves math problems step by step. Follow the examples provided and solve the given problem logically. Make sure to solve it within 5 steps."
      },
      {
        "role": "user",
        "content": "Find the derivative of the function f(x) = 3x^3 - 5x^2 + 4x - 7."
      },
      {
        "role": "assistant",
        "content": """
Reasoning step-by-step:
<STEP>
We begin by identifying the general rules for derivatives. For any term ax^n, the derivative is given by n * ax^(n-1).
</STEP>
<STEP>
Apply the rule to the first term, 3x^3. The derivative of 3x^3 is:
3 * 3x^2 = 9x^2.
</STEP>
<STEP>
Apply the rule to the second term, -5x^2. The derivative of -5x^2 is:
2 * (-5)x = -10x.
</STEP>
<STEP>
Apply the rule to the third term, 4x. The derivative of 4x is:
4 (since the derivative of x is 1).
</STEP>
<STEP>
The constant term, -7, has a derivative of 0 because the derivative of any constant is 0.
</STEP>
<ANSWER>
9x^2 - 10x + 4
</ANSWER>
"""
      },
      {
        "role": "user",
        "content": "Solve the equation: 2x + 3 = 7."
      },
      {
        "role": "assistant",
        "content": """
Reasoning step-by-step:
<STEP>
Subtract 3 from both sides of the equation to isolate the term with x:
2x + 3 - 3 = 7 - 3, 2x = 4.
</STEP>
<STEP>
Divide both sides by 2 to solve for x:
2x / 2 = 4 / 2, x = 2.
</STEP>
<ANSWER>
x = 2
</ANSWER>
"""
      }
    ]

    few_shot_examples_GSM8K = [
      {
        "role": "system",
        "content": "You are a helpful assistant that solves math problems step by step. Follow the examples provided and solve the given problem logically. Make sure to solve it within 10 steps."
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


    problem = "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?"
    solution = solver.solve_problem(problem, few_shot_examples_GSM8K)
    print(solution)
