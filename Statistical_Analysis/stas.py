import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Load the FN, FP, and dataset files
fn_file_path = 'false_negatives.csv'
fp_file_path = 'false_positives.csv'
dataset_file_path = 'dataset.csv'

false_negatives_df = pd.read_csv(fn_file_path)
false_positives_df = pd.read_csv(fp_file_path)
dataset_df = pd.read_csv(dataset_file_path)

# Extract the patient ID from the path in FN and FP datasets
def extract_patient_id(path):
    match = re.search(r'(\d+)\.csv$', path)  # Look for a number before ".csv"
    if match:
        return int(match.group(1))
    else:
        return None

false_negatives_df['patient_id'] = false_negatives_df['path'].apply(extract_patient_id)
false_positives_df['patient_id'] = false_positives_df['path'].apply(extract_patient_id)

# Merge the FN and FP dataframes with the dataset based on patient_id
fn_matches = pd.merge(false_negatives_df, dataset_df, on='patient_id', how='inner')
fp_matches = pd.merge(false_positives_df, dataset_df, on='patient_id', how='inner')

# Correlation for numerical features in FN and FP datasets
fn_corr_matrix = fn_matches.select_dtypes(include=['number']).dropna(axis=1, how='all').corr()
fp_corr_matrix = fp_matches.select_dtypes(include=['number']).dropna(axis=1, how='all').corr()

# Plot heatmap for FN correlations
plt.figure(figsize=(10, 8))
sns.heatmap(fn_corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap for False Negatives")
plt.show()

# Plot heatmap for FP correlations
plt.figure(figsize=(10, 8))
sns.heatmap(fp_corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap for False Positives")
plt.show()

# Save the correlation matrices to CSV files
fn_corr_matrix.to_csv('fn_correlation_filtered.csv')
fp_corr_matrix.to_csv('fp_correlation_filtered.csv')

print("Analysis complete. Heatmaps plotted and correlation matrices saved.")
