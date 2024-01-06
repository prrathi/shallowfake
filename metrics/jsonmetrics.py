import json
import matplotlib.pyplot as plt
import numpy as np
import sys


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from collections import Counter


def plot_pairwise_scatter(file_path,output_file_path):
    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Initialize lists to store scores
    stoi_scores = []
    waveform_compare_scores = []
    text_compare_scores = []

    # Extract scores from the data
    for key, value in data.items():
        for subkey, metrics in value.items():
            stoi_scores.append(metrics['stoi'])
            waveform_compare_scores.append(metrics['waveformCompare'])
            text_compare_scores.append(metrics['textCompare'])

    # Create a DataFrame from the scores
    df = pd.DataFrame({
        'STOI': stoi_scores,
        'SpectralCompare': waveform_compare_scores,
        'TextCompare': text_compare_scores
    })

    # Plot pairwise scatter plots
    plt.figure(figsize=(15, 5))

    # Scatter plot for STOI vs SpectralCompare
    plt.subplot(1, 3, 1)
    plt.scatter(df['STOI'], df['SpectralCompare'], alpha=0.5)
    plt.title('STOI vs SpectralCompare')
    plt.xlabel('STOI Score')
    plt.ylabel('SpectralCompare Score')

    # Scatter plot for STOI vs TextCompare
    plt.subplot(1, 3, 2)
    plt.scatter(df['STOI'], df['TextCompare'], alpha=0.5)
    plt.title('STOI vs TextCompare')
    plt.xlabel('STOI Score')
    plt.ylabel('TextCompare Score')

    # Scatter plot for SpectralCompare vs TextCompare
    plt.subplot(1, 3, 3)
    plt.scatter(df['SpectralCompare'], df['TextCompare'], alpha=0.5)
    plt.title('SpectralCompare vs TextCompare')
    plt.xlabel('SpectralCompare Score')
    plt.ylabel('TextCompare Score')

    plt.tight_layout()
    plt.savefig(f"{output_file_path}_scatter.png")

# Replace 'your_file.json' with the path to your JSON file
#plot_pairwise_scatter('valid.json')

def plot_pairwise_correlation(file_path,output_file_path):
    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Initialize lists to store scores
    stoi_scores = []
    waveform_compare_scores = []
    text_compare_scores = []

    # Extract scores from the data
    for key, value in data.items():
        for subkey, metrics in value.items():
            stoi_scores.append(metrics['stoi'])
            waveform_compare_scores.append(metrics['waveformCompare'])
            text_compare_scores.append(metrics['textCompare'])

    # Create a DataFrame from the scores
    df = pd.DataFrame({
        'STOI': stoi_scores,
        'SpectralCompare': waveform_compare_scores,
        'TextCompare': text_compare_scores
    })

    # Calculate the correlation matrix
    corr = df.corr()

    # Plot the heatmap of the correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Pairwise Correlation of Metrics')
    plt.savefig(f"{output_file_path}_crosscorr.png")

# Replace 'your_file.json' with the path to your JSON file
#plot_pairwise_correlation('valid.json')

def getLabel(text):
    if text=='clean_2':
        return 'Clean Baseline'
    elif 'whisper_pgd' in text:
        return "Whisper PGD"
    elif 'whisper_gan' in text:
        return "Whisper GAN"
    elif 'DF' in text and 'gan' not in text:
        return "DF PGD"
    elif 'df' in text:
        return "DF GAN"
    elif text == 'stoi':
        return 'STOI'
    elif text == 'textCompare':
        return "Text Compare"
    elif text == 'waveformCompare':
        return 'Spectral Comparison'
    return text

def generate_and_save_histograms_and_keys(file_path, output_histogram_file, output_keys_file):
    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    attacks = list(data[next(iter(data))].keys())
    print(attacks)
    attacks.remove('df_gan_wp200_clean')

    # Initialize lists to store scores and a dictionary for key mapping    
    stoi_scores = { attack : [] for attack in attacks} # attack to array
    waveform_compare_scores = { attack : [] for attack in attacks} # attack to array
    text_compare_scores = { attack : [] for attack in attacks} # attack to array
    key_to_text_compare = { attack : {} for attack in attacks} # attack to dictionary

    # Extract scores from the data
    for key, value in data.items():
        for attack, metrics in value.items():
            if attack not in attacks:
                continue
            stoi_score = metrics['stoi']
            waveform_compare_score = metrics['waveformCompare']
            text_compare_score = metrics['textCompare']

            stoi_scores[attack].append(stoi_score)
            waveform_compare_scores[attack].append(waveform_compare_score)
            text_compare_scores[attack].append(text_compare_score)

            key_to_text_compare[attack][key] = text_compare_score

    # TODO - uncomment this?
    # # Identify keys in the top 90th percentile for textCompare scores
    # percentile_90th = np.percentile(text_compare_scores, 90)
    # top_90th_keys = [key for key, score in key_to_text_compare.items() if score >= percentile_90th]

    # # Save the top keys to a file
    # with open(output_keys_file, 'w') as file:
    #     for key in top_90th_keys:
    #         file.write(key + '\n')

    # Plot histograms
    # plt.figure(figsize=(15, 5))

    cols = ['blue','green','red','brown', 'purple']
    attack_cols = {}
    for i in range(len(attacks)):
        attack_cols[attacks[i]] = cols[i]


    plt.figure(figsize=(5,5))

    # attacks = ['df_gan_wp200_clean','whisper_gan_wp200_clean']
    for attack in attacks:
        # Histogram for STOI scores
        # plt.subplot(1, 3, 1)
        plt.hist(stoi_scores[attack],bins=20, color=attack_cols[attack], alpha=0.2,label = getLabel(attack))
        print(attack,'stoi',np.average(stoi_scores[attack]))
        # plt.axvline(np.average(stoi_scores[attack]), color=attack_cols[attack], linestyle='solid', linewidth=3)
        plt.title('Histogram of STOI Scores')
        plt.xlabel('STOI Score')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.legend()

    # Save the histograms
    plt.savefig('stoi_'+output_histogram_file)
    plt.show()
    plt.clf()
    
    for attack in attacks:

        # Histogram for Spectral Compare scores
        # plt.subplot(1, 3, 2)
        plt.hist(waveform_compare_scores[attack], bins=20, color=attack_cols[attack], alpha=0.2,label = getLabel(attack))
        print(attack,'spectral',np.average(waveform_compare_scores[attack]))
        plt.title('Histogram of Spectral Compare Scores')
        plt.xlabel('Spectral Compare Score')
    plt.tight_layout()
    plt.legend()

    # Save the histograms
    plt.savefig('spectral_'+output_histogram_file)
    plt.show()
    plt.clf()

    
    for attack in attacks:

        # Histogram for Text Compare scores
        # plt.subplot(1, 3, 3)
        plt.hist(text_compare_scores[attack],bins=20, color=attack_cols[attack], alpha=0.2,label = getLabel(attack))
        print(attack,'text',np.average(text_compare_scores[attack]))
        plt.title('Histogram of Text Compare Scores')
        plt.xlabel('Text Compare Score')
    plt.tight_layout()
    plt.legend()

    # Save the histograms
    plt.savefig('text_'+output_histogram_file)
    plt.show()
    plt.clf()
    


# Replace with the path to your JSON file and desired output files
# generate_and_save_histograms_and_keys('valid.json', 'output_histograms.png', 'top_90th_keys.txt')

def generate_and_save_win_charts(file_path):
    # metrics = ['waveformCompare']
    metrics = ['stoi','waveformCompare','textCompare']
    with open(file_path, 'r') as file:
        data = json.load(file)

    attacks = list(data[next(iter(data))].keys())
    print(attacks)
    attacks.remove('df_gan_wp200_clean')


    # Initialize lists to store scores and a dictionary for key mapping    
    # dom = { attack : [] for attack in attacks} # attack to array
    all_dom = {metric : { attack : [] for attack in attacks} for metric in metrics}

    for id, scores in data.items():
        for metric in metrics:
            for attack in attacks:
                for attack2 in attacks:
                    if attack2 != attack and scores[attack][metric] > scores[attack2][metric]:
                        # if 'clean_2' == attack and 'DF_audio' == attack2 and metric=='waveformCompare':
                        #     print(id , scores[attack][metric], scores[attack2][metric])
                        all_dom[metric][attack].append(attack2)

    # now all_dom[metric][attack] has count of attacks being beat
    num_candidates = len(attacks)
    plt.figure(figsize=(6, 5))

    for i, metric in enumerate(metrics):
        # plt.subplot(1, 3, i+1)

        votes_matrix = np.zeros((num_candidates, num_candidates), dtype=int)

        # Count the number of times one candidate beats another
        for i, candidate in enumerate(attacks):
            votes_counter = Counter(all_dom[metric][candidate])
            for voted_over_candidate, count in votes_counter.items():
                j = attacks.index(voted_over_candidate)
                votes_matrix[i, j] = count*100/175


        plt.imshow(votes_matrix, cmap='Blues', interpolation='nearest')

        # Customize the plot
        plt.xticks(np.arange(num_candidates), [getLabel(a) for a in attacks], rotation=25)
        plt.yticks(np.arange(num_candidates), [getLabel(a) for a in attacks])
        plt.xlabel('Attacks That Got Beat')
        plt.ylabel('Attacks')
        plt.title(f'Method Comparison for {getLabel(metric)}')

        # Display the values in each cell
        for i in range(num_candidates):
            for j in range(num_candidates):
                plt.text(j, i, f'{round(votes_matrix[i, j], 2)}%', ha='center', va='center', color='orange')


        plt.colorbar(label='Victory %')
        plt.tight_layout()
        # plt.legend()

        # Save the histograms
        plt.savefig(f'win_{metric}_{file_path}.png')
        plt.show()
        plt.clf()

if __name__ == "__main__":
    # Check if the file path is provided as a command line argument
    if len(sys.argv) < 3:
        print("Usage: python script_name.py <json_file_path>")
        sys.exit(1)

    json_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    # Call your functions with the provided JSON file path
    plot_pairwise_scatter(json_file_path,output_file_path)
    plot_pairwise_correlation(json_file_path,output_file_path)
    generate_and_save_histograms_and_keys(json_file_path, f'output_histograms_{output_file_path}.png', f'{output_file_path}_top_90th_keys.txt')
    generate_and_save_win_charts(json_file_path)
