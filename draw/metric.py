import os
import json
import pandas as pd

# Define function to calculate metrics (same as before)
def drawculate_metrics(all_results):
    total_samples = len(all_results)
    correct_samples = [sample for sample in all_results if sample["first_answer"] == "Correct"]
    incorrect_samples = [sample for sample in all_results if sample["first_answer"] == "Incorrect"]
    initial_accuracy = len(correct_samples) / total_samples if total_samples else 0

    # Initialize result storage, partitioned by attack method
    attack_metrics = {}

    for sample in all_results:
        initial_answer = sample["first_answer"]  # 'Correct' or 'Incorrect'
        # Get all attack methods
        attack_methods = list(sample["attacks"].keys())

        for method in attack_methods:
            if method not in attack_metrics:
                attack_metrics[method] = {
                    "correct_then_incorrect_in_first_round": 0,
                    "final_correct_samples": 0
                }

            attack_rounds = sample["attacks"][method]
            # Get the result of the first attack (round == 1)
            first_attack = next((r for r in attack_rounds if r["round"] == 1), None)
            if first_attack:
                first_attack_answer = first_attack["answer"]
                # Check if the first attack changed from correct to incorrect or from incorrect to correct
                if initial_answer == "Correct" and first_attack_answer == "Incorrect":
                    attack_metrics[method]["correct_then_incorrect_in_first_round"] += 1

                # Update final_correct_samples to whether the first attack was correct
                if first_attack_answer == "Correct":
                    attack_metrics[method]["final_correct_samples"] += 1

    # Calculate final results by attack method
    metrics_by_method = {}
    for method, values in attack_metrics.items():
        final_accuracy = values["final_correct_samples"] / total_samples if total_samples else 0
        correct_then_incorrect_in_first_round_ratio = values["correct_then_incorrect_in_first_round"] / len(correct_samples) if correct_samples else 0

        metrics_by_method[method] = {
            "initial_accuracy": initial_accuracy,
            "final_accuracy": final_accuracy,
            "accuracy_change": final_accuracy - initial_accuracy,
            "correct_to_incorrect_ratio_first_round": correct_then_incorrect_in_first_round_ratio
        }

    return metrics_by_method

def find_latest_result(model_dir):
    """Find the latest result folder in the model directory"""
    base_path = os.path.join('results', model_dir)
    if not os.path.exists(base_path):
        return None
    
    # Get all relevant directories
    dirs = [d for d in os.listdir(base_path) if d.startswith('self_correction')]
    if not dirs:
        return None
    
    # Split directory names into base name and numeric suffix
    def get_suffix_num(d):
        parts = d.split('_')
        if len(parts) > 2 and parts[-1].isdigit():
            return int(parts[-1])
        return 0
    
    # Find the latest directory
    latest_dir = max(dirs, key=get_suffix_num)
    return os.path.join(base_path, latest_dir)

def process_all_models():
    """Process results for all models and generate CSV"""
    models = [
        "llama2-7b-instruct", "llama3-8b-instruct", "llama3.1-8b-instruct",
        "gpt3.5-turbo", "gpt3.5-turbo-instruct", "gpt4-turbo", "gpt4", 
        "gpt4o", "gpt4o-mini", "o1-preview", "o1-mini"
    ]
    
    results = []
    for model in models:
        latest_dir = find_latest_result(model)
        if not latest_dir:
            print(f"No results found for model {model}")
            continue
            
        results_file = os.path.join(latest_dir, 'output.json')
        if not os.path.exists(results_file):
            print(f"File not found: {results_file}")
            continue
            
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            metrics = drawculate_metrics(data)
            
            # Add a row for each attack method
            for attack_method, attack_metrics in metrics.items():
                row = {
                    'model': model,
                    'attack_method': attack_method,
                    'initial_accuracy': attack_metrics['initial_accuracy'],
                    'final_accuracy': attack_metrics['final_accuracy'],
                    'accuracy_change': attack_metrics['accuracy_change'],
                    'correct_to_incorrect_ratio': attack_metrics['correct_to_incorrect_ratio_first_round']
                }
                results.append(row)
                
        except Exception as e:
            print(f"Error processing model {model}: {str(e)}")
    
    # Create DataFrame and save as CSV
    if results:
        df = pd.DataFrame(results)
        csv_path = 'metrics_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
    else:
        print("No results found to process")

if __name__ == "__main__":
    process_all_models()
