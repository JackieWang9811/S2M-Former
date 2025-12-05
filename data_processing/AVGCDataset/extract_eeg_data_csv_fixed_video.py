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
            trail_ID.append(index+1)
    print(trail_ID)
    trials = mat_data['data']

    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    #
    # # Iterate through each trial and save the EEG data as .csv
    # # for i in range(trials.size):
    # for indix, ID in enumerate(trail_ID):
    #
    #     eeg_data_struct = trials[0, ID]
    #     df = pd.DataFrame(eeg_data_struct)
    #     # Construct the filename according to the naming convention
    #     filename = f"{output_folder}/{subject_number}Tra{indix + 1}.csv"
    #     # Save the DataFrame as a CSV file
    #     df.to_csv(filename, index=False, header=False)

    # print(f"EEG data extracted and saved for {subject_number}")


def main():
    # 2024-AV-GC-AAD-sub10_preprocessed.mat
    # subjects = range(1, 17)  # S1 to S16
    # 由于Sub14受试者只有1个trials的Fixed Video,所以去掉
    subjects = [1, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # S1 to S16
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
