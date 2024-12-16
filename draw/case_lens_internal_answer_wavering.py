import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

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
    if not os.path.exists(attack_data_path):
        print(f'Cannot find data in {attack_data_path} or {no_attack_data_path}')
        return None, None
    with open(attack_data_path, 'r') as f:
        attack_data = json.load(f)
    with open(no_attack_data_path, 'r') as f:
        no_attack_data = json.load(f)
        

    return attack_data, no_attack_data

def build_idx_to_sample(data):
    idx_to_sample = {}
    for i, sample in enumerate(data):
        sample_idx = sample.get('idx', i)  # Use 'i' if 'idx' is missing
        idx_to_sample[sample_idx] = sample
    return idx_to_sample

def plot_sample_data(ax, data_attack, data_no_attack, model_name, split_layer, max_layer, question_text):
    sns.set_palette('muted')
    colors = sns.color_palette("muted")

    # Font settings
    title_fontsize = 16
    label_fontsize = 16
    legend_fontsize = 16
    ticks_fontsize = 16
    question_fontsize = 16

    # Extract method names
    method_attack = list(data_attack['layer_confidences'].keys())[0]
    method_no_attack = list(data_no_attack['layer_confidences'].keys())[0]

    # Extract data
    correct_with_attack = np.array(data_attack['layer_confidences'][method_attack]['Correct'])
    incorrect_with_attack = np.array(data_attack['layer_confidences'][method_attack]['Incorrect'])
    correct_without_attack = np.array(data_no_attack['layer_confidences'][method_no_attack]['Correct'])
    incorrect_without_attack = np.array(data_no_attack['layer_confidences'][method_no_attack]['Incorrect'])

    # Compute Pr(correct) - Pr(incorrect)
    diff_with_attack = correct_with_attack - incorrect_with_attack
    diff_without_attack = correct_without_attack - incorrect_without_attack

    # Create x-axis
    num_layers = len(diff_with_attack)
    x = np.arange(1, num_layers + 1)

    # Define x-axis transformation function based on the model
    def transform_x(x):
        x = np.asarray(x)
        x_prime = np.zeros_like(x, dtype=float)
        idx1 = x <= split_layer
        idx2 = x > split_layer
        x_prime[idx1] = 0.25 * (x[idx1] - 1) / (split_layer - 1)
        x_prime[idx2] = 0.75 * (x[idx2] - split_layer) / (max_layer - split_layer) + 0.25
        return x_prime

    # Transform x-axis
    x_transformed = transform_x(x)

    # Plotting on the provided axes 'ax'
    ax.plot(x_transformed, diff_without_attack, label='Initial response generation', color=colors[0], linewidth=2)
    ax.plot(x_transformed, diff_with_attack, label='"Are you sure?"', color=colors[1], linewidth=2)
    # Add y=0 horizontal grey dashed line
    ax.axhline(0, color='gray', linestyle='--', linewidth=2)

    # Set y-axis limits
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, 1)
    ax.set_ylabel('Pr(correct) - Pr(incorrect)', fontsize=label_fontsize, labelpad=-2)

    # Set x-axis label
    # ax.set_xlabel('Layer', fontsize=label_fontsize, labelpad=-2)

    # Add legend
    legend = ax.legend(fontsize=legend_fontsize, loc='upper left')

    # Get legend frame properties
    legend_frame = legend.get_frame()
    frame_color = legend_frame.get_facecolor()
    frame_edge_color = legend_frame.get_edgecolor()

    # Add question text as annotation
    # Truncate the question text if too long
    max_question_length = 90
    if len(question_text) > max_question_length:
        question_text = question_text[:max_question_length] + '...'

    ax.text(0.04, -0.82, f"Question:\n{question_text}", fontsize=question_fontsize, 
            bbox=dict(facecolor=frame_color, edgecolor=frame_edge_color))

    # Set tick font sizes
    ax.tick_params(axis='x', labelsize=ticks_fontsize)
    ax.tick_params(axis='y', labelsize=ticks_fontsize)

    # Customize x-axis ticks
    x_ticks = [1, split_layer, max_layer]
    x_ticks_transformed = transform_x(x_ticks)
    ax.set_xticks(x_ticks_transformed)
    ax.set_xticklabels(x_ticks)

    # Add minor ticks
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
    ax.tick_params(axis='y', which='minor', length=3)
    ax.tick_params(axis='y', which='major', length=5)

    # Add x-axis minor ticks
    x_minor_ticks = np.arange(1, max_layer + 1, 1)
    x_minor_ticks_transformed = transform_x(x_minor_ticks)
    ax.xaxis.set_minor_locator(plt.FixedLocator(x_minor_ticks_transformed))
    ax.tick_params(axis='x', which='minor', length=3)

    # Set the title for individual plot
    ax.set_title(f'Model: {model_name}', fontsize=title_fontsize)

def main():
    import argparse
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model name: llama3-8b-instruct, llama2-7b-instruct, or llama3.1-8b-instruct')
    args = parser.parse_args()
    
    # Model specific configurations
    model_configs = {
        'Llama-2-7B': {
            'split_layer': 16, 
            'max_layer': 32,
            'base_dir': 'results/llama2-7b-instruct',
            'attack_dir_keyword': 'tuned_lens',
            'no_attack_dir_keyword': 'tuned_lens_without_attack',
        },
        'Llama-3-8B': {
            'split_layer': 16, 
            'max_layer': 32,
            'base_dir': 'results/llama3-8b-instruct',
            'attack_dir_keyword': 'tuned_lens',
            'no_attack_dir_keyword': 'tuned_lens_without_attack',
        },
        'Llama-3.1-8B': {
            'split_layer': 20, 
            'max_layer': 32,
            'base_dir': 'results/llama3.1-8b-instruct',
            'attack_dir_keyword': 'tuned_lens',
            'no_attack_dir_keyword': 'tuned_lens_without_attack',
        },
    }
    
    # change model name
    if args.model == 'llama3-8b-instruct':
        args.model = 'Llama-3-8B'
    elif args.model == 'llama2-7b-instruct':
        args.model = 'Llama-2-7B'
    elif args.model == 'llama3.1-8b-instruct':
        args.model = 'Llama-3.1-8B'
    else:
        raise ValueError(f"Unsupported model: {args.model}.")
    
    if args.model not in model_configs:
        raise ValueError(f"Unsupported model: {args.model}. Must be one of {list(model_configs.keys())}")
    
    # Get model configuration
    config = model_configs[args.model]
    base_dir = config['base_dir']
    
    # Load both attack and no-attack data
    attack_data, no_attack_data = load_data(
        base_dir,
        config['attack_dir_keyword'],
        config['no_attack_dir_keyword']
    )
    
    if attack_data is None or no_attack_data is None:
        print("Failed to load data")
        return
    
    # Find latest experiment directories
    attack_dir = find_latest_experiment(base_dir, config['attack_dir_keyword'])
    no_attack_dir = find_latest_experiment(base_dir, config['no_attack_dir_keyword'])
    
    if not attack_dir or not no_attack_dir:
        print(f'Cannot find experiments in {base_dir}')
        return
    
    # Create output directory
    plots_dir = os.path.join(base_dir, attack_dir, 'individual_plots')
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Created output directory: {plots_dir}")
    
    # Set style
    sns.set_palette('muted')
    plt.rcParams['font.family'] = 'Arial'
    
    # Build index to sample mappings
    attack_idx_to_sample = build_idx_to_sample(attack_data)
    no_attack_idx_to_sample = build_idx_to_sample(no_attack_data)
    
    print(f"Starting to plot {len(attack_data)} samples...")
    
    # Plot each sample in a separate figure
    for idx, attack_sample in enumerate(tqdm(attack_data)):
        sample_idx = attack_sample.get('idx', idx)
        no_attack_sample = no_attack_idx_to_sample.get(sample_idx)
        
        if no_attack_sample is None:
            print(f"Warning: No matching no-attack sample for idx {sample_idx}")
            continue
        
        # Create new figure for each sample
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        
        # Get question text
        question_text = attack_sample.get('question', 'Unknown Question')
        
        # Plot the sample
        plot_sample_data(ax, attack_sample, no_attack_sample, args.model, 
                        config['split_layer'], config['max_layer'], question_text)
        
        # Save individual plot
        save_path = os.path.join(plots_dir, f'sample_{sample_idx}.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()
    
    print(f"All plots saved in directory: {plots_dir}")

if __name__ == '__main__':
    main()
