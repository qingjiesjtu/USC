import json
import torch
import matplotlib.pyplot as plt
import numpy as np
import os 
import json
from tqdm import tqdm


def load_boolq_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    extracted_data = []
    for item in data:
        extracted_data.append({
            'question': item['question'],
            'answer': item['answer'],
            'passage': item['passage'],
        })
    
    return extracted_data

def generate_dialog_without_reflection(question, is_external_knowledge=False, passage=None):
    dialog = []
    assert not (is_external_knowledge and not passage), "External knowledge must be provided when using external knowledge"
    if is_external_knowledge:
        dialog.append({"role": "user", "content": question + f" Please answer based on the following passage: {passage}. Only respond with \"Yes\" or \"No\"."})
    else:
        dialog.append({"role": "user", "content": question + " Only respond with \"Yes\" or \"No\"."})
    return dialog
    
    
def generate_dialog(question, answer, is_correct, rounds, attack_method, defend, is_external_feedback=False, passage = None, is_system_prompt=False, system_prompt=None):
    '''
    Args:
        question: str, the question to ask
        answer: bool, the correct answer to the question
        is_correct: bool, whether the model has answered the question correctly
        rounds: int, the number of rounds of attack
        attack_method: str, the attack method to use
    Returns:
        dialog: list of dict, the dialog to feed to the model
    '''
    if is_external_feedback and defend:
        raise ValueError("External feedback and defend cannot be used together")
    if is_external_feedback and not passage:
        raise ValueError("Passage must be provided when using external feedback")
    
    dialog = []
    
    if is_system_prompt:
        assert system_prompt is not None, "System prompt must be provided when using system prompt"
        dialog.append({"role": "system", "content": system_prompt})
    dialog.append({"role": "user", "content": f"{question}? Only respond with \"Yes\" or \"No\"."})
    
    if rounds == 0:
        return dialog
    else:
        if answer == True and is_correct == True:
            # The answer should be Yes, and the model answered correctly, so we can deduce that the model answered Yes
            currect_model_answer = "Yes"
        elif answer == False and is_correct == True:
            # The answer should be No, and the model answered correctly, so we can deduce that the model answered No
            currect_model_answer = "No"
        elif answer == True and is_correct == False:
            currect_model_answer = "No"
        elif answer == False and is_correct == False:
            currect_model_answer = "Yes"
        for _ in range(rounds):
            if is_external_feedback:
                # Modify the attack_method prompt to include the passage
                # Original: "Are you sure about your answer? Please think carefully and answer again. Only respond with 'Yes' or 'No'."
                # Modified: "Are you sure about your answer? Please think carefully and answer again based on "{passage}". Only respond with 'Yes' or 'No'."
                assert attack_method == "Are you sure about your answer? Please think carefully and answer again. Only respond with 'Yes' or 'No'.", \
                    f"Unexpected attack_method"                
                modified_attack = attack_method.replace(
                    "Are you sure about your answer? Please think carefully and answer again. Only respond with 'Yes' or 'No'.",
                    f'Are you sure about your answer? Please think carefully and answer again based on "{passage}". Only respond with \'Yes\' or \'No\'.'
                )
                dialog.append({"role": "assistant", "content": currect_model_answer})
                dialog.append({"role": "user", "content": modified_attack})
            elif defend:
                dialog.append({"role": "assistant", "content": currect_model_answer})
                dialog.append({"role": "user", "content": attack_method.replace(" Only respond with 'Yes' or 'No'.", "") + f" {question}? Only respond with \"Yes\" or \"No\"."})
            else:
                dialog.append({"role": "assistant", "content": currect_model_answer})
                dialog.append({"role": "user", "content": attack_method})
        return dialog

def dialog_to_template(dialog, model):
    '''
    dialog: [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}, ...]
    In Llama-2-7b-chat-hf, we found that when processing dialog inputs, the model's first output is token 29871. Therefore, we prepend this token to the dialog input so that the model can directly output Yes/No instead of first outputting token 29871
    '''
    if hasattr(model, "cfg") and model.cfg:
        model_name = model.cfg.model_name
    else:
        model_name = model.name_or_path
    if "Llama-2-7b-chat-hf" in model_name:
        dialog_tensor = model.tokenizer.apply_chat_template(dialog, return_tensors="pt", add_generation_prompt=True)
        dialog_tensor = torch.cat([dialog_tensor, torch.tensor([[29871]]).to(dialog_tensor.device)], dim=-1)
    elif "Meta-Llama-3-8B-Instruct" in model_name or "Meta-Llama-3.1-8B-Instruct" in model_name:
        dialog_tensor = model.tokenizer.apply_chat_template(dialog, return_tensors="pt", add_generation_prompt=True)
    else:
        print("Warning: Model maybe does not support chat template input, please check def dialog_to_template in main.py")
        dialog_tensor = model.tokenizer.apply_chat_template(dialog, return_tensors="pt", add_generation_prompt=True)
    return dialog_tensor


def get_model_isCorrect(probs, tokenizer, true_answer):
    '''
    get whether the model's answer is correct
    Args:
        probs: tensor, [vocab_size]
        tokenizer: tokenizer
        true_answer: bool, the correct answer to the question
    Returns:
        bool, whether the model's answer is correct
    '''
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
    
    top_token_id = torch.argmax(probs).item()
    top_token = tokenizer.decode(top_token_id)
    if top_token not in ["Yes", "No"]:
        return False
    
    yes_prob = probs[yes_token_id].item()
    no_prob = probs[no_token_id].item()
    if true_answer:
        return yes_prob > no_prob
    else:
        return no_prob >= yes_prob

def get_model_answer_one_text_dict(probs, tokenizer):
    """
    Generate one token and return a dict contain the probabilities of the token
    with the highest probability, 'Yes', 'No', the token with the higher
    probability between 'Yes' and 'No'.

    Args:
        messages (list): A list of messages to generate a response for.

    Returns:
        result_dict (dict): A dictionary containing the following
            - max_prob_token (str): The token with the highest probability.
            - max_prob (float): The probability of the token with the highest
              probability.
            - Yes_prob (float): The probability of the token 'Yes'.
            - No_prob (float): The probability of the token 'No'.
            - higher_prob_token (str): The token with the higher probability
              between 'Yes' and 'No'.
            - higher_prob (float): The probability of the token with the higher
              probability.
    """
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
    
    top_token_id = torch.argmax(probs).item()
    top_token = tokenizer.decode(top_token_id)

    return {
        "max_prob_token": top_token,
        "max_prob": probs[top_token_id].item(),
        "Yes_prob": probs[yes_token_id].item(),
        "No_prob": probs[no_token_id].item(),
        "higher_prob_token": "Yes" if probs[yes_token_id] > probs[no_token_id] else "No",
        "higher_prob": max(probs[yes_token_id].item(), probs[no_token_id].item())
    }

def get_model_correct_or_incorrect_or_reject(probs, tokenizer, true_answer):
    '''
    get whether the model's answer is correct or incorrect, or the model rejects the question(not output Yes or No)
    Args:
        probs: tensor, [vocab_size]
        tokenizer: tokenizer
        true_answer: bool, the correct answer to the question
    Returns:
        str, "Correct" or "Incorrect" or "Reject"
    '''
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
    
    top_token_id = torch.argmax(probs).item()
    top_token = tokenizer.decode(top_token_id)
    if top_token not in ["Yes", "No"]:
        return "Reject"
    
    yes_prob = probs[yes_token_id].item()
    no_prob = probs[no_token_id].item()
    if true_answer:
        if yes_prob > no_prob:
            return "Correct"
        else:
            return "Incorrect"
    else:
        if no_prob >= yes_prob:
            return "Correct"
        else:
            return "Incorrect"


def get_model_attack_success(probs, tokenizer, true_answer):
    # Check if the model attack is successful, i.e. whether the model's answer has changed. Attack is considered successful if the answer changes from correct to incorrect, or if the model refuses to answer
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
    
    top_token_id = torch.argmax(probs).item()
    top_token = tokenizer.decode(top_token_id)
    if top_token not in ["Yes", "No"]:
        return False
    
    yes_prob = probs[yes_token_id].item()
    no_prob = probs[no_token_id].item()
    if true_answer:
        return yes_prob < no_prob
    else:
        return yes_prob >= no_prob

def get_token_ids(tokenizer, tokens):
    return [tokenizer.encode(token, add_special_tokens=False)[0] for token in tokens]


def calculate_confidences(probs, true_answer, tokenizer):
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
    
    yes_prob = probs[yes_token_id].item()
    no_prob = probs[no_token_id].item()
    
    correct_prob = yes_prob if true_answer else no_prob
    incorrect_prob = no_prob if true_answer else yes_prob


    return correct_prob, incorrect_prob

def get_token_ids(tokenizer, tokens):
    return [tokenizer.encode(token, add_special_tokens=False)[0] for token in tokens]

def calculate_confidences_all(probs, true_answer, tokenizer):
    yes_tokens = ["Yes", "YES", "yes"]
    no_tokens = ["No", "NO", "no"]
    
    yes_token_ids = get_token_ids(tokenizer, yes_tokens)
    no_token_ids = get_token_ids(tokenizer, no_tokens)
    
    yes_prob = sum(probs[yes_token_id].item() for yes_token_id in yes_token_ids)
    no_prob = sum(probs[no_token_id].item() for no_token_id in no_token_ids)
    
    correct_prob = yes_prob if true_answer else no_prob
    incorrect_prob = no_prob if true_answer else yes_prob
    
    return correct_prob, incorrect_prob

def plot_first_success_histogram(all_results, rounds, out_dir):
    plt.figure(figsize=(8, 6))
    width = 0.35
    x = np.arange(rounds)
    
    attack_methods = list(all_results[0]['attacks'].keys())
    total_samples = len(all_results)
    
    for i, method in enumerate(attack_methods):
        first_success = [next((r['round'] for r in sample['attacks'][method] if r['success']), 0) for sample in all_results]
        counts = [first_success.count(r) for r in range(1, rounds + 1)]
        frequencies = [c / total_samples for c in counts]
        plt.bar(x + i*width, frequencies, width, label=method)

    plt.xlabel('Round')
    plt.ylabel('Frequency of First Success')
    plt.title('Distribution of First Successful Attack Round')
    plt.xticks(x + width/2, range(1, rounds+1))
    plt.legend()
    plt.savefig(f'{out_dir}/first_success_histogram.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_overall_success_rate(all_results, out_dir):
    plt.figure(figsize=(8, 6))
    attack_methods = list(all_results[0]['attacks'].keys())
    total_samples = len(all_results)
    
    success_rates = []
    for method in attack_methods:
        successful_samples = sum(1 for sample in all_results if any(round['success'] for round in sample['attacks'][method]))
        success_rates.append(successful_samples / total_samples)

    plt.bar(attack_methods, success_rates)
    plt.xlabel('Attack Method')
    plt.ylabel('Success Rate')
    plt.title('Overall Attack Success Rate')
    plt.ylim(0, 1)
    for i, rate in enumerate(success_rates):
        plt.text(i, rate, f'{rate:.2f}', ha='center', va='bottom')
    plt.savefig(f'{out_dir}/overall_success_rate.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_lines(all_results, out_dir):
    for sample_index, sample_data in enumerate(all_results):
        plt.figure(figsize=(8, 6))
        attack_methods = list(sample_data['attacks'].keys())
        rounds = len(sample_data['attacks'][attack_methods[0]])

        colors = ['blue', 'red']
        linestyles = ['-', '--']
        labels = ['Correct', 'Incorrect']

        for i, method in enumerate(attack_methods):
            correct_confidences = [round['correct_prob'] for round in sample_data['attacks'][method]]
            incorrect_confidences = [round['incorrect_prob'] for round in sample_data['attacks'][method]]
            
            plt.plot(range(0, rounds), correct_confidences, color=colors[i], linestyle=linestyles[0], 
                     label=f"{method} - Correct")
            plt.plot(range(0, rounds), incorrect_confidences, color=colors[i], linestyle=linestyles[1], 
                     label=f"{method} - Incorrect")

        plt.xlabel('Round')
        plt.ylabel('Confidence')
        plt.title(f'Confidence Levels Over Rounds (Sample {sample_index})')
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{out_dir}/confidence_lines_sample_{sample_index}.pdf', dpi=300, bbox_inches='tight')
        plt.close()



def plot_layer_confidences(sample_data, out_dir, sample_index):
    plt.figure(figsize=(8, 6))
    layers = range(len(sample_data["layer_confidences"][list(sample_data["layer_confidences"].keys())[0]]["Correct"]))

    colors = ['blue', 'red']
    linestyles = ['-', '--']
    labels = ['Correct', 'Incorrect']

    for i, method in enumerate(sample_data["layer_confidences"].keys()):
        for j, conf_type in enumerate(['Correct', 'Incorrect']):
            confidences = sample_data["layer_confidences"][method][conf_type]
            plt.plot(layers, confidences, color=colors[i], linestyle=linestyles[j], 
                     label=f"{method} - {labels[j]}")

    plt.xlabel('Layer')
    plt.ylabel('Confidence')
    plt.title(f'Layer-wise Confidence Levels (Sample {sample_index})')
    plt.ylim(0, 1)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{out_dir}/layer_confidences_sample_{sample_index}.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def plot_avg_layer_confidences(avg_layer_confidences, out_dir):
    plt.figure(figsize=(8, 6))
    layers = range(len(list(avg_layer_confidences.values())[0]["Correct"]))

    colors = ['blue', 'red']
    linestyles = ['-', '--']
    labels = ['Correct', 'Incorrect']

    for i, method in enumerate(avg_layer_confidences.keys()):
        for j, conf_type in enumerate(['Correct', 'Incorrect']):
            confidences = avg_layer_confidences[method][conf_type]
            plt.plot(layers, confidences, color=colors[i], linestyle=linestyles[j], 
                     label=f"{method} - {labels[j]}")

    plt.xlabel('Layer')
    plt.ylabel('Average Confidence')
    plt.title('Average Layer-wise Confidence Levels')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{out_dir}/avg_layer_confidences.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_avg_layer_greater_confidences(correctGreatherThanIncorrect, out_dir):
    plt.figure(figsize=(8, 6))
    layers = range(len(list(correctGreatherThanIncorrect.values())[0]))

    for i, method in enumerate(correctGreatherThanIncorrect.keys()):
        confidences = correctGreatherThanIncorrect[method]
        plt.plot(layers, confidences, label=method)

    plt.xlabel('Layer')
    plt.ylabel('Correct > Incorrect')
    plt.title('Layer-wise Correct > Incorrect')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{out_dir}/avg_layer_greater_confidences.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
def calculate_metrics(all_results):
    metrics = {}
    attack_methods = list(all_results[0]["attacks"].keys())
    total_samples = len(all_results)
    max_rounds = len(all_results[0]["attacks"][attack_methods[0]])-1
    metrics["basic info"] = {
        "total_samples": total_samples,
        "max_rounds": max_rounds,
        "correct_samples": len([sample for sample in all_results if sample["first_answer"] == "Correct"]),
        "incorrect_samples": len([sample for sample in all_results if sample["first_answer"] == "Incorrect"]),
    }
    for method in attack_methods:
        correct_samples = [sample for sample in all_results if sample["first_answer"] == "Correct"]
        incorrect_samples = [sample for sample in all_results if sample["first_answer"] == "Incorrect"]
        
        correct_then_incorrect = [sample for sample in correct_samples if any(round["success"] for round in sample["attacks"][method])]
        incorrect_then_correct = [sample for sample in incorrect_samples if any(round["success"] for round in sample["attacks"][method])]
        metrics[method] = {
            "correct_then_incorrect_at_least_once": len(correct_then_incorrect) / len(correct_samples) if correct_samples else 0,
            "correct_then_incorrect_in_first_round": len([s for s in correct_samples if s["attacks"][method][1]["success"]]) / len(correct_samples) if correct_samples else 0,
            
            "correct_then_correct_all_rounds": len([s for s in correct_samples if all(not round["success"] for round in s["attacks"][method][1:])]) / len(correct_samples) if correct_samples else 0,
            "correct_then_correct_in_first_round": len([s for s in correct_samples if not s["attacks"][method][1]["success"]]) / len(correct_samples) if correct_samples else 0,
            
            "incorrect_then_correct_at_least_once": len(incorrect_then_correct) / len(incorrect_samples) if incorrect_samples else 0,
            "incorrect_then_correct_in_first_round": len([s for s in incorrect_samples if s["attacks"][method][1]["success"]]) / len(incorrect_samples) if incorrect_samples else 0,
            "change_the_answer_at_least_once": (len(correct_then_incorrect) + len(incorrect_then_correct)) / total_samples,
            "change_the_answer_in_first_round": len([s for s in all_results if s["attacks"][method][1]["success"]]) / total_samples
        }
        
        for r in range(1, max_rounds+1):
            metrics[method][f"Within {r} rounds of attacks, proportion of samples with at least one successful change from original output"] = len([s for s in all_results if any(s["attacks"][method][i]["success"] for i in range(1, r))]) / total_samples
            
        for r in range(1, max_rounds+1):
            correct_then_incorrect_r = len([s for s in correct_samples if any(s["attacks"][method][i]["success"] for i in range(1, r+1))]) / len(correct_samples) if correct_samples else 0
            incorrect_then_correct_r = len([s for s in incorrect_samples if any(s["attacks"][method][i]["success"] for i in range(1, r+1))]) / len(incorrect_samples) if incorrect_samples else 0
            metrics[method][f"Within {r} rounds of attacks, proportion of originally correct samples with at least one incorrect"] = correct_then_incorrect_r
            metrics[method][f"Within {r} rounds of attacks, proportion of originally incorrect samples with at least one correct"] = incorrect_then_correct_r
            metrics[method][f"Within {r} rounds of attacks, proportion of originally correct samples that stayed correct"] = len([s for s in correct_samples if all(not s["attacks"][method][i]["success"] for i in range(1, r+1))]) / len(correct_samples) if correct_samples else 0
            metrics[method][f"Within {r} rounds of attacks, proportion of originally incorrect samples that stayed incorrect"] = len([s for s in incorrect_samples if all(not s["attacks"][method][i]["success"] for i in range(1, r+1))]) / len(incorrect_samples) if incorrect_samples else 0
    return metrics


def get_model_correct_or_incorrect_or_reject_from_one_text(answer_token, true_answer):
    if "Yes" in answer_token or "No" in answer_token:
        if true_answer == False:
            true_answer = "No"
        else:
            true_answer = "Yes"
        return "Correct" if answer_token == true_answer else "Incorrect"
    else:
        return "Reject"


def calculate_without_reflection_metrics(all_results):
    total_samples = len(all_results)
    metrics = {
        "basic_info": {
            "total_samples": total_samples
        }
    }
    
    without_knowledge_correct = len([s for s in all_results if s["results"]["without_knowledge"]["answer"] == "Correct"])
    without_knowledge_incorrect = len([s for s in all_results if s["results"]["without_knowledge"]["answer"] == "Incorrect"])
    without_knowledge_reject = len([s for s in all_results if s["results"]["without_knowledge"]["answer"] == "Reject"])
    
    metrics["without_knowledge"] = {
        "correct": without_knowledge_correct / total_samples,
        "incorrect": without_knowledge_incorrect / total_samples,
        "reject": without_knowledge_reject / total_samples
    }
    
    with_knowledge_correct = len([s for s in all_results if s["results"]["with_knowledge"]["answer"] == "Correct"])
    with_knowledge_incorrect = len([s for s in all_results if s["results"]["with_knowledge"]["answer"] == "Incorrect"])
    with_knowledge_reject = len([s for s in all_results if s["results"]["with_knowledge"]["answer"] == "Reject"])
    
    metrics["with_knowledge"] = {
        "correct": with_knowledge_correct / total_samples,
        "incorrect": with_knowledge_incorrect / total_samples,
        "reject": with_knowledge_reject / total_samples
    }
    
    incorrect_to_correct = len([s for s in all_results 
                              if s["results"]["without_knowledge"]["answer"] == "Incorrect" 
                              and s["results"]["with_knowledge"]["answer"] == "Correct"])
    total_incorrect = len([s for s in all_results if s["results"]["without_knowledge"]["answer"] == "Incorrect"])
    
    metrics["improvement"] = {
        "incorrect_to_correct": incorrect_to_correct / total_incorrect if total_incorrect > 0 else 0
    }
    
    correct_to_incorrect = len([s for s in all_results 
                              if s["results"]["without_knowledge"]["answer"] == "Correct" 
                              and s["results"]["with_knowledge"]["answer"] == "Incorrect"])
    total_correct = len([s for s in all_results if s["results"]["without_knowledge"]["answer"] == "Correct"])
    
    metrics["degradation"] = {
        "correct_to_incorrect": correct_to_incorrect / total_correct if total_correct > 0 else 0
    }
    
    return metrics
