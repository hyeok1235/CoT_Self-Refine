import re
from cot_refactored import MathProblemSolver
from datasets import load_dataset
import random
import json
from tqdm import tqdm
import argparse
import csv


# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
model_name = "microsoft/Phi-4-mini-reasoning"

# Function to load and preprocess datasets
def load_and_preprocess_datasets(dataset_name):
    print("Loading and preprocessing dataset...")
    if dataset_name == "gsm8k":
        dataset = load_dataset("gsm8k", "main")["test"]
        return [{"problem": item["question"], "solution": item["answer"]} for item in dataset]

    elif dataset_name == "math_qa":
        dataset = load_dataset("math_qa")["test"]
        return [{"problem": item["Problem"], "solution": item["Rationale"]} for item in dataset]

    elif dataset_name == "custom":
        # Example for a custom dataset
        return [
            {"problem": "Solve the equation: 2x + 3 = 7.", "solution": "x = 2"},
            {"problem": "Find the derivative of f(x) = 3x^3 - 5x^2 + 4x - 7.", "solution": "9x^2 - 10x + 4"}
        ]
    elif dataset_name == "math500":
        with open("MATH500_sampled.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
        # Process the dataset to extract problems and solutions
        processed_dataset = [{"problem": item["problem"], "solution": item["answer"]} for item in dataset]
        # Filter out problems whose solution is not an integer (only digits allowed)
        integer_dataset = [item for item in processed_dataset if re.match(r'^\d+$', item["solution"])]
        return integer_dataset
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

# Function to save the sampled dataset to a file
def save_sampled_dataset(sampled_dataset, filename="sampled_dataset.json"):
    with open(filename, "w") as f:
        json.dump(sampled_dataset, f)
    print(f"Sampled dataset saved to {filename}")

# Function to load the sampled dataset from a file
def load_sampled_dataset(filename="sampled_dataset.json"):
    with open(filename, "r") as f:
        return json.load(f)

def extract_final_value_from_ground_truth(solution):
    """
    Extract the value after #### from the ground truth solution using regex.

    Args:
        solution (str): The ground truth solution.

    Returns:
        str: The extracted value, or an empty string if no value is found.
    """
    match = re.search(r"####\s*(.+)", solution)  # Match anything after ####
    if match:
        return match.group(1).strip()  # Extract and strip any whitespace
    return "No Answer Found"

def extract_numeric_value(text):
    """
    Remove all non-numeric characters from the given string.

    Args:
        text (str): The input string containing numbers and other characters.

    Returns:
        str: A string containing only numeric characters.
    """
    return re.sub(r"[^\d]", "", text)

# Main evaluation script
if __name__ == "__main__":
    # Set up argument parsing so that dataset can be selected from the terminal
    parser = argparse.ArgumentParser(description="Evaluate math datasets with a Llama solver.")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Name of the dataset (gsm8k, math_qa, math500, custom)")
    args = parser.parse_args()
    dataset_name = args.dataset
    sampled_filename = f"sampled_dataset_{dataset_name}.json"
    
    # Define the model and load it
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

    few_shot_examples_MATH500 = [
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
                        "Compute [1/cos^2(10°) + 1/sin^2(20°) + 1/sin^2(40°)].")
        },
        {
        "role": "assistant",
        "content": """
Reasoning step-by-step:
<STEP>
We wish to simplify S = 1/cos²(10°) + 1/sin²(20°) + 1/sin²(40°).
Notice that cos(10°) = sin(80°) (since sin(90° - θ) = cos(θ)). Thus, we can rewrite the first term as:
1/cos²(10°) = 1/sin²(80°).
</STEP>
<STEP>
Now the sum becomes:
S = 1/sin²(20°) + 1/sin²(40°) + 1/sin²(80°).
</STEP>
<STEP>
Using the identity csc²(θ) = 1 + cot²(θ), we can rewrite each term:
S = [1 + cot²(20°)] + [1 + cot²(40°)] + [1 + cot²(80°)].
This simplifies to:
S = 3 + (cot²(20°) + cot²(40°) + cot²(80°)).
</STEP>
<STEP>
A numerical check or derivation shows that cot²(20°) + cot²(40°) + cot²(80°) = 9.
</STEP>
<STEP>
Finally, adding these values together gives:
S = 3 + 9 = 12.
</STEP>
<ANSWER>
12
</ANSWER>
"""
        },
        {
            "role": "user",
            "content": ("Solve this math problem step by step. Problem : \n"
                        "Evaluate the definite integral ∫[0 to 1] (3x^2 + 2x) dx.")
        },
        {
        "role": "assistant",
        "content": """
Reasoning step-by-step:
<STEP>
The given integral is ∫[0 to 1] (3x² + 2x) dx. We can split this into two separate integrals:
∫[0 to 1] (3x² dx) + ∫[0 to 1] (2x dx).
</STEP>
<STEP>
The integral of x^n is given by x^(n+1)/(n+1). Using this rule, we calculate each term:
∫[0 to 1] (3x² dx) = [3 * x³ / 3] from 0 to 1 = [x³] from 0 to 1.
</STEP>
<STEP>
For the second term:
∫[0 to 1] (2x dx) = [2 * x² / 2] from 0 to 1 = [x²] from 0 to 1.
</STEP>
<STEP>
Now evaluate both terms at the bounds of integration. For the first term, x³ from 0 to 1:
At x=1, x³ = 1³ = 1.
At x=0, x³ = 0³ = 0.
Thus, the first term evaluates to: 1 - 0 = 1.
</STEP>
<STEP>
For the second term, x² from 0 to 1:
At x=1, x² = 1² = 1.
At x=0, x² = 0² = 0.
Thus, the second term evaluates to: 1 - 0 = 1.
</STEP>
<STEP>
Adding both terms together gives:
Total integral value = (First term) + (Second term) = 1 + 1 = 2.
</STEP>
<ANSWER>
2
</ANSWER>
"""
        }
    ]

    # List of datasets to evaluate
    datasets = ["gsm8k", "math500"]  # Add or modify dataset names as needed

    # Number of samples to evaluate
    num_samples = 200

    # Check if the sampled dataset already exists
    try:
        sampled_dataset = load_sampled_dataset(sampled_filename)
        print("Loaded previously sampled dataset.")
    except FileNotFoundError:
        print("Sampled dataset not found, generating a new one.")
        
        # Load and preprocess dataset
        dataset = load_and_preprocess_datasets(dataset_name)

        # Ensure the dataset has enough samples
        total_problems = len(dataset)
        if total_problems < num_samples:
            print(f"Dataset contains only {total_problems} problems, which is less than the requested {num_samples} samples.")
        else:
            sampled_dataset = random.sample(dataset, num_samples)
            save_sampled_dataset(sampled_dataset, sampled_filename)

    correct = 0
    count = 0
    cor_to_cor = [0, 0, 0, 0] # 경우의 수, accept된 피드백의 개수, valid한 피드백의 총 개수, 전체 step의 개수
    wro_to_cor = [0, 0, 0, 0]

    cor_to_wro = [0, 0, 0, 0]
    wro_to_wro1 = [0, 0, 0, 0]
    wro_to_wro2 = [0, 0, 0, 0]

    csv_filename = f"answers_{dataset_name}.csv"
    csv_file = open(csv_filename, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['problem', 'formatted_initial_answer', 'formatted_answer', 'ground_truth'])

    # Initialize tqdm for progress bar
    for data in tqdm(sampled_dataset, desc="Processing problems", total=num_samples):
        problem = data["problem"]
        ground_truth = data["solution"]
        # print(f"data : {data}")
        # print("--------------------")
        # print(f"Problem: {problem}")
        # print(f"Ground Truth: {ground_truth}")

        # Generate solution using the model
        fewshot_examples = few_shot_examples_GSM8K if dataset_name == "gsm8k" else few_shot_examples_MATH500
        generated_solution = solver.solve_problem(problem, fewshot_examples)
        formatted_initial_answer = extract_numeric_value(generated_solution[0])
        formatted_answer = extract_numeric_value(generated_solution[1])
        valid_count = generated_solution[2]
        major_count = generated_solution[3]
        feedback_count = generated_solution[4]

        # Extract the value after #### from the ground truth
        if dataset_name == "gsm8k":
            ground_truth_value = extract_numeric_value(extract_final_value_from_ground_truth(ground_truth))
        elif dataset_name == "math500":
            ground_truth_value = extract_numeric_value(ground_truth)
        
        csv_writer.writerow([problem, formatted_initial_answer, formatted_answer, ground_truth_value])

        # Check if ground_truth_value is inside generated_solution
        if ground_truth_value in formatted_answer or formatted_answer in ground_truth_value: # b = c
            correct += 1
            if formatted_initial_answer in formatted_answer or formatted_answer in formatted_initial_answer: # a = b
                cor_to_cor[0] += 1
                cor_to_cor[1] += valid_count
                cor_to_cor[2] += major_count
                cor_to_cor[3] += feedback_count
            else:                                       # a != b
                wro_to_cor[0] += 1
                wro_to_cor[1] += valid_count
                wro_to_cor[2] += major_count
                wro_to_cor[3] += feedback_count
        else:                                               # b != c
            if formatted_initial_answer in ground_truth_value or ground_truth_value in formatted_initial_answer: # a = c
                cor_to_wro[0] += 1
                cor_to_wro[1] += valid_count
                cor_to_wro[2] += major_count
                cor_to_wro[3] += feedback_count
                print("cor_to_wro")
            elif formatted_initial_answer in formatted_answer or formatted_answer in formatted_initial_answer:  # a = b and a != c
                wro_to_wro1[0] += 1
                wro_to_wro1[1] += valid_count
                wro_to_wro1[2] += major_count
                wro_to_wro1[3] += feedback_count
                print("wro_to_wro1")
            else:                                                                                               # a != b and a != c
                wro_to_wro2[0] += 1
                wro_to_wro2[1] += valid_count
                wro_to_wro2[2] += major_count
                wro_to_wro2[3] += feedback_count
                print("wro_to_wro2")
            print(f"Problem: {problem}")
            print(f"Initial Answer: {formatted_initial_answer}")
            print(f"Generated Solution: {formatted_answer}")
            print(f"Ground Truth: {ground_truth_value}")
            print("--------------------")

        count += 1

        # Calculate accuracy
        if count % 50 == 0:
            accuracy = (correct / count) * 100
            print("\n************************************\n")
            print(f"Accuracy after {count} problems: {accuracy:.2f}%")
            print("Correct to Correct: ", cor_to_cor)
            print("Wrong to Correct: ", wro_to_cor)
            print("Correct to Wrong: ", cor_to_wro)
            print("Wrong to Wrong (Initial Answer == Generated Answer): ", wro_to_wro1)
            print("Wrong to Wrong (Initial Answer != Generated Answer): ", wro_to_wro2)
            print("\n************************************\n")

    # Final accuracy calculation
    accuracy = (correct / count) * 100
    print(f"Total Accuracy: {accuracy:.2f}%")
    print("Correct to Correct: ", cor_to_cor)
    print("Wrong to Correct: ", wro_to_cor)
    print("Correct to Wrong: ", cor_to_wro)
    print("Wrong to Wrong (Initial Answer == Generated Answer): ", wro_to_wro1)
    print("Wrong to Wrong (Initial Answer != Generated Answer): ", wro_to_wro2)
