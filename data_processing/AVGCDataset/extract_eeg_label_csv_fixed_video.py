import scipy.io
import pandas as pd
import os

def extract_and_save_eeg_data(subject_number, mat_file_path, output_folder, ConType="No"):
    # Load .mat file
    mat_data = scipy.io.loadmat(mat_file_path)
    ID = mat_data['conditionID'][0]
    trail_ID = []
    for index, condition in enumerate(ID):
        if "FixedVideo" in condition[0]:
            trail_ID.append(index)
    print(trail_ID)
    trials_label = mat_data['initAttention'][0]
    # print(trials.size)
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    labels = []  # To store labels for all trials

    # Iterate through each trial and save the EEG data and label
    for i in trail_ID:
        eeg_data_label = trials_label[i][0]

        # Convert 'R' to 1 and 'L' to 0
        if eeg_data_label == 'right':
            label = 1
        elif eeg_data_label == 'left':
            label = 0
        else:
            label = None  # Handle unexpected cases

        labels.append(label)  # Store the label

    # Ensure the output directory for CSV files exists
    csv_output_folder = os.path.join(output_folder, 'csv')
    if not os.path.exists(csv_output_folder):
        os.makedirs(csv_output_folder)

    # Save all labels to a CSV file
    label_filename = os.path.join(csv_output_folder, f"{subject_number}{ConType}.csv")
    labels_df = pd.DataFrame(labels, columns=["Label"])
    labels_df.to_csv(label_filename, index=False)

    print(f"EEG data and labels extracted and saved for {subject_number}")


def main():
    # 2024-AV-GC-AAD-sub10_preprocessed.mat
    # 由于Sub受试者只有1个trials的Fixed Video
    subjects = [1, 3, 4, 7, 8, 9, 10, 11, 12, 13,  15, 16]  # S1 to S16
    data_document_path = "/data/wjq/AAD/AVGCDataset/preprocessed_data"
    output_data_document_path = "/data/wjq/AAD/AVGCDataset_fixed_video"
    ConType = "No_vanilla_128"
    for subject_number in subjects:
        name = f"0{subject_number}" if subject_number < 10 else f"{subject_number}"
        mat_file_path = os.path.join(data_document_path, f"2024-AV-GC-AAD-sub{name}_preprocessed.mat")

        output_name = f"S{subject_number}"
        output_folder = os.path.join(output_data_document_path, ConType, output_name)
        extract_and_save_eeg_data(output_name, mat_file_path, output_folder, ConType)


if __name__ == "__main__":
    main()