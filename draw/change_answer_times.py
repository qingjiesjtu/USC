import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from tqdm import tqdm

# Function to get the highest sequence repeat_exp directory
def get_latest_repeat_exp_dir(base_dir):
    repeat_dirs = [d for d in os.listdir(base_dir) if 'repeat_exp' in d]
    # Sort to ensure both numbered and non-numbered repeat_exp directories are considered
    repeat_dirs = sorted(repeat_dirs, key=lambda x: int(x.split('_')[-1]) if '_' in x and x.split('_')[-1].isdigit() else -1, reverse=True)
    return repeat_dirs[0] if repeat_dirs else None

# Function to drawculate answer change frequency for each question, separated by attack method
def drawculate_change_frequency(data):
    attack_change_frequencies = {}
    for item in data:
        question = item['question']
        attacks = item['attacks']
        for attack_method, rounds in attacks.items():
            if attack_method not in attack_change_frequencies:
                attack_change_frequencies[attack_method] = []
            change_count = 0
            for i in range(1, len(rounds)):
                if rounds[i]['token'] != rounds[i-1]['token']:
                    change_count += 1
            frequency = change_count  # Dividing by 10 as per the requirement
            attack_change_frequencies[attack_method].append(frequency)
    
    average_frequencies = {
        attack_method: (sum(frequencies) / len(frequencies)) if frequencies else 0
        for attack_method, frequencies in attack_change_frequencies.items()
    }
    return average_frequencies

def calculate_change_distribution(data):
    """Calculate the distribution of change frequencies for each model and attack method"""
    distributions = {}
    for item in data:
        attacks = item['attacks']
        for attack_method, rounds in attacks.items():
            if attack_method not in distributions:
                distributions[attack_method] = []
            
            change_count = 0
            for i in range(1, len(rounds)):
                if rounds[i]['token'] != rounds[i-1]['token']:
                    change_count += 1
            distributions[attack_method].append(change_count)
    
    return distributions

def plot_change_frequencies(df):
    # Set font and style
    title_fontsize = 14
    label_fontsize = 14
    legend_fontsize = 12
    ticks_fontsize = 12
    rcParams['font.family'] = 'Arial'
    sns.set_palette('muted')

    # Model name mapping
    model_rename_mapping = {
        'gpt3.5-turbo': 'GPT-3.5-Turbo',
        'gpt4o': 'GPT-4o',
        'llama2-7b-instruct': 'Llama2-7B-Chat',
        'llama3-8b-instruct': 'Llama3-8B-Instruct',
        'llama3.1-8b-instruct': 'Llama3.1-8B-Instruct',
        'o1-mini': 'GPT-o1-mini',
        'o1-preview': 'GPT-o1-preview'
    }
    df['Model'] = df['Model'].replace(model_rename_mapping)

    # Set model order
    model_order = [
        'GPT-3.5-Turbo',
        'GPT-4o',
        'GPT-o1-mini',
        'GPT-o1-preview',
        'Llama2-7B-Chat',
        'Llama3-8B-Instruct',
        'Llama3.1-8B-Instruct'
    ]
    df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(4, 3))

    # Calculate percentage for each threshold
    thresholds = range(1, 11)
    percentage_data = []
    
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        for threshold in thresholds:
            percentage = (model_data['Changes'] >= threshold).mean() * 100
            percentage_data.append({
                'Model': model,
                'Threshold': threshold,
                'Percentage': percentage
            })
    
    df_plot = pd.DataFrame(percentage_data)

    # Draw line plot
    sns.lineplot(
        data=df_plot,
        x='Threshold',
        y='Percentage',
        hue='Model',
        style='Model',
        markers=True,
        dashes=False,
        palette='colorblind',
        markersize=8,
        linewidth=2,
        marker='x',
        ax=ax
    )

    # Customize figure
    ax.set_xlabel('Number of Changes', fontsize=label_fontsize)
    ax.set_ylabel('Percentage (%)', fontsize=label_fontsize, labelpad=-5)
    ax.set_xticks(thresholds)
    ax.set_xticklabels([f'≥{i}' for i in thresholds], fontsize=ticks_fontsize)

    ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
    ax.tick_params(axis='both', which='major', length=5)
    ax.tick_params(axis='both', which='minor', length=3)

    ax.set_xlim(0.9, 10.1)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='both', which='major', labelsize=ticks_fontsize)
    ax.grid(False)

    # Adjust legend
    legend = ax.legend(
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize,
        framealpha=0.5,
        loc='upper right'
    )

    # Save figure
    plt.savefig('model_answer_changes.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.close()

def main():
    base_results_dir = './results'
    model_distributions = {}

    # Collect all model change distribution data
    for model_dir in os.listdir(base_results_dir):
        model_path = os.path.join(base_results_dir, model_dir)
        if os.path.isdir(model_path):
            latest_repeat_exp_dir = get_latest_repeat_exp_dir(model_path)
            if latest_repeat_exp_dir:
                repeat_exp_path = os.path.join(model_path, latest_repeat_exp_dir, 'repeat_exp.json')
                if os.path.exists(repeat_exp_path):
                    with open(repeat_exp_path, 'r') as f:
                        data = json.load(f)
                        distributions = calculate_change_distribution(data)
                        model_distributions[model_dir] = distributions

    # Create DataFrame
    records = []
    for model, distributions in model_distributions.items():
        for attack_method, changes in distributions.items():
            if "Are you sure" in attack_method:
                attack_method = "Confirmatory"
            elif "incorrect" in attack_method:
                attack_method = "Misleading"
            for change_count in changes:
                records.append((model, attack_method, change_count))

    df = pd.DataFrame(records, columns=['Model', 'Attack Method', 'Changes'])

    # Draw plot
    plot_change_frequencies(df)
    
    print(f"Plot saved as model_answer_changes.pdf")

if __name__ == "__main__":
    main()

