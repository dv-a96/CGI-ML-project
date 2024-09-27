import pandas as pd
import numpy as np

# Load the original dataset and false negative/positive files
dataset = pd.read_csv('new_dataset_1024_64_128.csv')
false_negatives = pd.read_csv('../Wrist_Transfer_Learning/false_negatives.csv', header=None)
false_positives = pd.read_csv('../Wrist_Transfer_Learning/false_positives.csv', header=None)

# Function to crop or shuffle vector
def augment_vector(data):
    # Shuffle
    np.random.shuffle(data)
    return data

# Create a new CSV file for augmented data
with open('augmantation_data.csv', 'w') as out_file:
    out_file.write('path,label\n')

    # Iterate over the original dataset
    for _, row in dataset.iterrows():
        path, label = row['path'], row['label']

        # Check if the path exists in false negatives or positives (without turning them into sets)
        if path in false_negatives[0].values or path in false_positives[0].values:
            data = np.loadtxt(path, delimiter=',')
            augmented_data = augment_vector(data)

            # Save augmented data to a new path (e.g., same directory with '_aug' added to filename)
            new_path = path.replace('.csv', '_aug.csv')
            np.savetxt(new_path, augmented_data, delimiter=',')

            # Write new path and label to the output file
            out_file.write(f'{new_path},{label}\n')
        else:
            # If no augmentation, write the original path and label
            out_file.write(f'{path},{label}\n')

