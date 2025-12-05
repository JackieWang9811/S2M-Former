import scipy.io
import pandas as pd
import os
import numpy as np


def extract_labels(data):
    results = []
    # 遍历每个元组
    for item in data:
        # 第一个元素，提取1
        for i in item:
            number = i[1][0][0][0][0]
            results.append(number)
        # first_number = item[0][0][0][0]
        # results.append(first_number)
        #
        # # 第二个元素，可能需要进一步提取
        # second_number = item[1][0][0][0][0]
        # results.append(second_number)

    return results


def extract_and_save_eeg_data(subject_number, mat_file_path, output_folder, ConType="No"):
    # Load .mat file
    mat_data = scipy.io.loadmat(mat_file_path)
    data = mat_data['data']    # 提取eeg数据，假设它存储在名为'eeg'的单元格数组中    # 访问数组中的第一个元素，这通常是实际的结构体
    data_struct = data.item()  # 使用 .item() 来访问结构体中的数据

    # 提取eeg数据，假设它存储在名为'eeg'的字段中
    eeg_data_trials = data_struct[3]  # 使用字典式访问

    # 提取event数据，假设它存储在名为'event'的字段中
    event_data = data_struct[2]  # 使用字典式访问
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(eeg_data_trials.size)

    labels = event_data.item()
    labels = labels[0]
    labels = extract_labels(labels)


    # Iterate through each trial and save the EEG data as .csv
    for i in range(eeg_data_trials.size):
        eeg_data_struct = eeg_data_trials[0, i][:,:64]
        # eeg_data_label = eeg_data_struct[0][0][3]
        # eeg_data = eeg_data_struct[0][0][0][0][0][1]
        df = pd.DataFrame(eeg_data_struct)

        # Construct the filename according to the naming convention
        filename = f"{output_folder}/{subject_number}Tra{i + 1}.csv"

        # Save the DataFrame as a CSV file
        df.to_csv(filename, index=False, header=False)

    print(f"EEG data extracted and saved for {subject_number}")


def main():
    subjects = range(18, 19)  # S1 to S18
    data_document_path = "/data/wjq/AAD/DTUDataset/DTU_vanilla_dataset+preproc+128/"
    output_data_document_path = "/data/wjq/AAD/DTUDataset"
    ConType = "No_vanilla_128"
    for subject_number in subjects:
        name = f"S{subject_number}"
        mat_file_path = os.path.join(data_document_path, f"{name}_data_preproc.mat")
        output_folder = os.path.join(output_data_document_path, ConType, name)

        extract_and_save_eeg_data(name, mat_file_path, output_folder, ConType)


if __name__ == "__main__":
    main()
