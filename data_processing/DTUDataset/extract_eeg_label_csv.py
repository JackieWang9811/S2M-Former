import scipy.io
import pandas as pd
import os


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
    labels = [x - 1 for x in labels]

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