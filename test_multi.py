import re
from cot_refactored import MathProblemSolver
from datasets import load_dataset
import random
import json
from tqdm import tqdm

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

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
    # Define the model and load it
    solver = MathProblemSolver(model_name=model_name)

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

    # List of datasets to evaluate
    datasets = ["gsm8k"]  # Add or modify dataset names as needed

    # Number of samples to evaluate
    num_samples = 200

    # Check if the sampled dataset already exists
    try:
        sampled_dataset = load_sampled_dataset()
        print("Loaded previously sampled dataset.")
    except FileNotFoundError:
        print("Sampled dataset not found, generating a new one.")
        
        # Load and preprocess dataset
        dataset = load_and_preprocess_datasets("gsm8k")

        # Ensure the dataset has enough samples
        total_problems = len(dataset)
        if total_problems < num_samples:
            print(f"Dataset contains only {total_problems} problems, which is less than the requested {num_samples} samples.")
        else:
            sampled_dataset = random.sample(dataset, num_samples)
            save_sampled_dataset(sampled_dataset)

    correct = 0
    count = 0
    cor_to_cor = [0, 0, 0]
    wro_to_cor = [0, 0, 0]

    cor_to_wro = [0, 0, 0]
    wro_to_wro1 = [0, 0, 0]
    wro_to_wro2 = [0, 0, 0]

    # Initialize tqdm for progress bar
    for data in tqdm(sampled_dataset, desc="Processing problems", total=num_samples):
        problem = data["problem"]
        ground_truth = data["solution"]

        # Generate solution using the model
        generated_solution = solver.solve_problem(problem, few_shot_examples_GSM8K)
        formatted_initial_answer = extract_numeric_value(generated_solution[0])
        formatted_answer = extract_numeric_value(generated_solution[1])
        pass_count = generated_solution[2]
        step_count = generated_solution[3]

        # Extract the value after #### from the ground truth
        ground_truth_value = extract_numeric_value(extract_final_value_from_ground_truth(ground_truth))

        # Check if ground_truth_value is inside generated_solution
        if ground_truth_value in formatted_answer or formatted_answer in ground_truth_value:
            correct += 1
            if formatted_initial_answer == formatted_answer:
                cor_to_cor[0] += 1
                cor_to_cor[1] += pass_count
                cor_to_cor[2] += step_count
            else:
                wro_to_cor[0] += 1
                wro_to_cor[1] += pass_count
                wro_to_cor[2] += step_count
        else:
            if formatted_initial_answer in ground_truth_value or ground_truth_value in formatted_initial_answer:
                cor_to_wro[0] += 1
                cor_to_wro[1] += pass_count
                cor_to_wro[2] += step_count
                print("cor_to_wro")
            elif formatted_initial_answer == formatted_answer:
                wro_to_wro1[0] += 1
                wro_to_wro1[1] += pass_count
                wro_to_wro1[2] += step_count
                print("wro_to_wro1")
            else:
                wro_to_wro2[0] += 1
                wro_to_wro2[1] += pass_count
                wro_to_wro2[2] += step_count
                print("wro_to_wro2")
            print(f"Problem: {problem}")
            print(f"Initial Answer: {generated_solution[0]}")
            print(f"Generated Solution: {formatted_answer}")
            print(f"Ground Truth: {ground_truth_value}")
            print("--------------------")

        count += 1

        # Calculate accuracy
        if count % 100 == 0:
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
