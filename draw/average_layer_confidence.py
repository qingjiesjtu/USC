import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def find_latest_experiment(base_path, prefix):
    import re
    pattern = re.compile(f"{prefix}_(\\d+)")
    latest_experiment = None
    max_number = -1

    for item in os.listdir(base_path):
        if item == prefix:
            if max_number == -1:
                latest_experiment = item
                max_number = 0
        else:
            match = pattern.match(item)
            if match:
                number = int(match.group(1))
                if number > max_number:
                    latest_experiment = item
                    max_number = number

    return latest_experiment

def load_data(base_dir, attack_dir_keyword, no_attack_dir_keyword):
    attack_dir = find_latest_experiment(base_dir, attack_dir_keyword)
    no_attack_dir = find_latest_experiment(base_dir, no_attack_dir_keyword)
    if not attack_dir or not no_attack_dir:
        print(f'Cannot find experiments in {base_dir}')
        return None, None
    attack_data_path = os.path.join(base_dir, attack_dir, 'lens_data.json')
    no_attack_data_path = os.path.join(base_dir, no_attack_dir, 'lens_data.json')
    if not os.path.exists(attack_data_path) or not os.path.exists(no_attack_data_path):
        print(f'Cannot find data in {attack_data_path} or {no_attack_data_path}')
        return None, None
    with open(attack_data_path, 'r') as f:
        attack_data = json.load(f)
    with open(no_attack_data_path, 'r') as f:
        no_attack_data = json.load(f)
    return attack_data, no_attack_data

def plot_diff_line_chart(base_dir, model_name, split_layer, max_layer):
    # Set style
    sns.set_palette('muted')
    colors = sns.color_palette("muted")
    
    method_colors = {
        'Initial Response': colors[0],  # 蓝色
        'Are you sure?': colors[1],     # 橙色
        'You are wrong.': colors[3]     # 绿色
    }
    
    # Font settings
    title_fontsize = 16
    label_fontsize = 16
    legend_fontsize = 14
    ticks_fontsize = 14
    
    # Load data
    attack_data, no_attack_data = load_data(
        base_dir,
        'tuned_lens',
        'tuned_lens_without_attack'
    )
    
    if attack_data is None or no_attack_data is None:
        print("Failed to load data")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Process data
    all_no_attack_diffs = []
    all_confirmatory_diffs = []
    all_misleading_diffs = []
    
    # Load all data first
    for i, sample in enumerate(no_attack_data):
        method = list(sample['layer_confidences'].keys())[0]
        correct = np.array(sample['layer_confidences'][method]['Correct'])
        incorrect = np.array(sample['layer_confidences'][method]['Incorrect'])
        all_no_attack_diffs.append(correct - incorrect)
        
        attack_sample = attack_data[i]
        for method, data in attack_sample['layer_confidences'].items():
            correct = np.array(data['Correct'])
            incorrect = np.array(data['Incorrect'])
            diff = correct - incorrect
            
            if any(keyword in method.lower() for keyword in ["are you sure"]):
                all_confirmatory_diffs.append(diff)
            elif any(keyword in method.lower() for keyword in ["incorrect", "wrong"]):
                all_misleading_diffs.append(diff)
        
    
    # Convert to numpy arrays
    all_no_attack_diffs = np.array(all_no_attack_diffs)
    all_confirmatory_diffs = np.array(all_confirmatory_diffs) if all_confirmatory_diffs else np.array([])
    all_misleading_diffs = np.array(all_misleading_diffs) if all_misleading_diffs else np.array([])
    
    if len(all_confirmatory_diffs) == 0 or len(all_misleading_diffs) == 0:
        print("Error: Could not find confirmatory or misleading data in the attack samples")
        return
    
    subset_indices = [] # correct2incorrect
    for i in range(len(all_no_attack_diffs)):
        diff_nsc_last_layer = all_no_attack_diffs[i, -1]
        
        all_methods_negative = True
        sample_sc = attack_data[i]
        methods_sc = list(sample_sc['layer_confidences'].keys())
        
        for method in methods_sc:
            correct = np.array(sample_sc['layer_confidences'][method]['Correct'])
            incorrect = np.array(sample_sc['layer_confidences'][method]['Incorrect'])
            diff_sc_last_layer = correct[-1] - incorrect[-1]
            if diff_sc_last_layer >= 0:
                all_methods_negative = False
                break
        
        if diff_nsc_last_layer > 0 and all_methods_negative:
            subset_indices.append(i)
    
    no_attack_diffs_filtered = all_no_attack_diffs[subset_indices]
    confirmatory_diffs_filtered = all_confirmatory_diffs[subset_indices]
    misleading_diffs_filtered = all_misleading_diffs[subset_indices]
    
    
    no_attack_means = np.mean(no_attack_diffs_filtered, axis=0)
    no_attack_stds = np.std(no_attack_diffs_filtered, axis=0)
    confirmatory_means = np.mean(confirmatory_diffs_filtered, axis=0)
    confirmatory_stds = np.std(confirmatory_diffs_filtered, axis=0)
    misleading_means = np.mean(misleading_diffs_filtered, axis=0)
    misleading_stds = np.std(misleading_diffs_filtered, axis=0)
    
    # Coordinate transformation function
    def transform_x(x):
        x = np.asarray(x)
        x_prime = np.zeros_like(x, dtype=float)
        idx1 = x <= 15
        idx2 = x > 15
        x_prime[idx1] = 0.25 * (x[idx1] - 1) / (15 - 1)
        x_prime[idx2] = 0.75 * (x[idx2] - 16) / (32 - 16) + 0.25
        return x_prime
    
    # Create and transform x-axis
    x = np.arange(1, len(no_attack_means) + 1)
    x_transformed = transform_x(x)
    
    # Draw lines
    line1, = ax.plot(x_transformed, no_attack_means, 
                     label='Initial response generation', 
                     color=method_colors['Initial Response'], 
                     linewidth=2)
    ax.fill_between(x_transformed, no_attack_means - no_attack_stds, 
                    no_attack_means + no_attack_stds, 
                    alpha=0.2, color=method_colors['Initial Response'])
    
    line2, = ax.plot(x_transformed, confirmatory_means, 
                     label='"Are you sure?"', 
                     color=method_colors['Are you sure?'], 
                     linewidth=2)
    ax.fill_between(x_transformed, confirmatory_means - confirmatory_stds, 
                    confirmatory_means + confirmatory_stds, 
                    alpha=0.2, color=method_colors['Are you sure?'])
    
    line3, = ax.plot(x_transformed, misleading_means, 
                     label='"You are wrong."', 
                     color=method_colors['You are wrong.'], 
                     linewidth=2)
    ax.fill_between(x_transformed, misleading_means - misleading_stds, 
                    misleading_means + misleading_stds, 
                    alpha=0.2, color=method_colors['You are wrong.'])
    
    # Add y=0 gray dashed line
    ax.axhline(0, color='gray', linestyle='--', linewidth=2)
    
    # Set chart properties
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Layer', fontsize=label_fontsize)
    ax.set_ylabel('Pr(correct) - Pr(incorrect)', fontsize=label_fontsize)
    
    # Set legend
    ax.legend(fontsize=legend_fontsize, loc='upper left')
    
    # Set ticks
    x_ticks = [1, 5, 10, 15, 16, 20, 25, 30, 32]
    x_ticks_transformed = transform_x(x_ticks)
    ax.set_xticks(x_ticks_transformed)
    ax.set_xticklabels(x_ticks)
    
    # Add minor ticks
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
    ax.tick_params(axis='y', which='minor', length=3)
    ax.tick_params(axis='y', which='major', length=5)
    
    x_minor_ticks = np.arange(1, 33, 1)
    x_minor_ticks_transformed = transform_x(x_minor_ticks)
    ax.xaxis.set_minor_locator(plt.FixedLocator(x_minor_ticks_transformed))
    ax.tick_params(axis='x', which='minor', length=3)
    
    # Set title
    ax.set_title('"Are you sure?" v.s. "You are wrong."', fontsize=title_fontsize)
    
    # Save chart
    attack_dir = find_latest_experiment(base_dir, 'tuned_lens')
    save_dir = os.path.join(base_dir, attack_dir)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'average_layer_confidence.pdf'), 
                format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Chart saved to {os.path.join(save_dir, 'average_layer_confidence.pdf')}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, 
                       help='Model name: llama3-8b-instruct, llama2-7b-instruct, or llama3.1-8b-instruct')
    args = parser.parse_args()
    
    # Model configuration
    model_configs = {
        'Llama-2-7B': {
            'split_layer': 16, 
            'max_layer': 32,
            'base_dir': 'results/llama2-7b-instruct',
        },
        'Llama-3-8B': {
            'split_layer': 16, 
            'max_layer': 32,
            'base_dir': 'results/llama3-8b-instruct',
        },
        'Llama-3.1-8B': {
            'split_layer': 20, 
            'max_layer': 32,
            'base_dir': 'results/llama3.1-8b-instruct',
        },
    }
    
    # Convert model name
    if args.model == 'llama3-8b-instruct':
        args.model = 'Llama-3-8B'
    elif args.model == 'llama2-7b-instruct':
        args.model = 'Llama-2-7B'
    elif args.model == 'llama3.1-8b-instruct':
        args.model = 'Llama-3.1-8B'
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    if args.model not in model_configs:
        raise ValueError(f"Unsupported model: {args.model}. Must be one of: {list(model_configs.keys())}")
    
    config = model_configs[args.model]
    plot_diff_line_chart(
        config['base_dir'],
        args.model,
        config['split_layer'],
        config['max_layer']
    )

if __name__ == '__main__':
    main()
