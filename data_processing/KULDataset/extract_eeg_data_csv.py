import scipy.io
import pandas as pd
import os


def extract_and_save_eeg_data(subject_number, mat_file_path, output_folder, ConType="No"):
    # Load .mat file
    mat_data = scipy.io.loadmat(mat_file_path)
    trials = mat_data['preproc_trials']

    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each trial and save the EEG data as .csv
    for i in range(trials.size):
        eeg_data_struct = trials[0, i]
        eeg_data_label = eeg_data_struct[0][0][3]
        eeg_data = eeg_data_struct[0][0][0][0][0][1]
        df = pd.DataFrame(eeg_data)

        # Construct the filename according to the naming convention
        filename = f"{output_folder}/{subject_number}Tra{i + 1}.csv"

        # Save the DataFrame as a CSV file
        df.to_csv(filename, index=False, header=False)

    print(f"EEG data extracted and saved for {subject_number}")


def main():
    subjects = range(1, 17)  # S1 to S16
    data_document_path = "/data/wjq/AAD/KULDataset/preprocessed_data_0204_Cz_0220"
    output_data_document_path = "/data/wjq/AAD/KULDataset"
    ConType = "No_vanilla_250204_Cz_0220"
    for subject_number in subjects:
        name = f"S{subject_number}"
        mat_file_path = os.path.join(data_document_path, f"{name}.mat")
        output_folder = os.path.join(output_data_document_path, ConType, name)

        extract_and_save_eeg_data(name, mat_file_path, output_folder, ConType)


if __name__ == "__main__":
    main()
