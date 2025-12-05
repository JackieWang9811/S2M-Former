import pickle
import math
import pandas as pd
from sklearn.preprocessing import scale
from scipy.interpolate import griddata
from importlib import reload
from scipy.io import loadmat
from utils import makePath, cart2sph, pol2cart
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.model_selection import KFold
from mne.decoding import CSP

def get_logger(name, log_path, length):
    import logging
    reload(logging)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logfile = makePath(log_path) + "/Train"+str(length)+"s_" + name + ".log"
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    if log_path == "./result/test":
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, math.pi / 2 - elev)


def gen_images(data, args):
    locs = loadmat('/data/wjq/AAD/OpenAAD/tools/locs_orig.mat')
    locs_3d = locs['data']
    locs_2d = []
    for e in locs_3d:
        locs_2d.append(azim_proj(e))

    data = np.nan_to_num(data, nan=np.nan, posinf=np.nan, neginf=np.min(data[np.isfinite(data)]))

    locs_2d_final = np.array(locs_2d)
    grid_x, grid_y = np.mgrid[
                     min(np.array(locs_2d)[:, 0]):max(np.array(locs_2d)[:, 0]):args.image_size * 1j,
                     min(np.array(locs_2d)[:, 1]):max(np.array(locs_2d)[:, 1]):args.image_size * 1j]

    images = []
    for i in range(data.shape[0]):
        images.append(griddata(locs_2d_final, data[i, :], (grid_x, grid_y), method='cubic', fill_value=np.nan))
    images = np.stack(images, axis=0)

    images[~np.isnan(images)] = scale(images[~np.isnan(images)])
    images = np.nan_to_num(images)
    return images


def visualize_gen_images(images, title, save_path="visualizations"):
    """
    可视化并保存生成的特征图
    :param images: 输入的图像数据，形状为 (N, 32, 32)
    :param title: 保存的图像标题，例如 "Delta", "Theta" 等
    :param save_path: 保存图像的路径，默认为 "visualizations"
    """
    # 创建保存文件夹
    os.makedirs(save_path, exist_ok=True)

    num_images = min(5, images.shape[0])  # 限制可视化的图像数量
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        axes[i].imshow(images[i], cmap='viridis')
        axes[i].axis('off')
    plt.suptitle(title)

    # 保存图像
    save_file = os.path.join(save_path, f"{title}.png")
    plt.savefig(save_file, bbox_inches='tight')
    plt.close(fig)  # 关闭图像，节省内存
    print(f"Saved {title} visualization to {save_file}")


def read_prepared_data(args):
    data = []
    label_name = "/" + args.data_document_path + "/" + args.ConType + "/" + args.name + "/csv/" + args.name + args.ConType + ".csv"
    label = pd.read_csv(label_name)
    target = []
    for k in range(args.trail_number):
        filename = args.data_document_path + "/" + args.ConType + "/" + args.name + "/" + args.name +"Tra" + str(k + 1) + ".csv"
        data_pf = pd.read_csv(filename, header=None)
        # if args.dataset == "KULDataset":
        #     eeg_data = data_pf.iloc[:49920, :64]
        # else:
        eeg_data = data_pf.iloc[:, :64]
        data.append(eeg_data)
        target.append(label.iloc[k, args.label_col])
    return data, target


def sliding_window(eeg_datas, labels, args, out_channels):
    window_size = args.window_length
    stride = int(window_size * (1 - args.overlap))

    train_eeg = []
    test_eeg = []
    train_label = []
    test_label = []

    # eeg_datas:(60,6400,64)
    for m in range(len(labels)): # m => trail 数
        eeg = eeg_datas[m] # (80176,64)
        label = labels[m]
        windows = []
        new_label = []
        for i in range(0, eeg.shape[0] - window_size + 1, stride): # 计算窗口数量
            window = eeg[i:i+window_size, :]
            windows.append(window)
            new_label.append(label)

        # 去掉训练集最后一个窗口，避免与测试集重叠
        train_eeg.append(np.array(windows)[:int(len(windows)*0.9)])
        test_eeg.append(np.array(windows)[int(len(windows)*0.9):])
        train_label.append(np.array(new_label)[:int(len(windows)*0.9)])
        test_label.append(np.array(new_label)[int(len(windows)*0.9):])

    train_eeg = np.stack(train_eeg, axis=0).reshape(-1, window_size, out_channels)
    test_eeg = np.stack(test_eeg, axis=0).reshape(-1, window_size, out_channels)
    train_label = np.stack(train_label, axis=0).reshape(-1, 1)
    test_label = np.stack(test_label, axis=0).reshape(-1, 1)

    return train_eeg, test_eeg, train_label, test_label


def sliding_window_cross_trials(sub, eeg_datas, labels, args, out_channels):
    window_size = args.window_length
    stride = int(window_size * (1 - args.overlap))

    all_train_eeg = []
    all_test_eeg = []
    all_train_label = []
    all_test_label = []

    if "AVGCDataset" in args.dataset:
        # Define the trials for each subject
        select_trials = {
            "S1": [1, 4], "S3": [1, 4], "S4": [2, 6], "S7": [3, 6],
            "S8": [2, 6], "S9": [2, 6], "S10": [2, 6], "S11": [3, 6],
            "S12": [3, 6], "S13": [2, 6], "S14": [2, 6], "S15": [2, 6], "S16": [3, 6]}

        # Get the specific trials to select for this subject
        selected_trials = select_trials.get(sub)

        if selected_trials is None:
            raise ValueError(f"Subject {sub} not found in the select_trials dictionary.")

        # Create train and test sets based on the selected trials
        train_eeg = []
        test_eeg = []
        train_label = []
        test_label = []

        for i in range(len(labels)):
            eeg = eeg_datas[i]
            label = labels[i]
            windows = []
            new_label = []
            for j in range(0, eeg.shape[0] - window_size + 1, stride):
                window = eeg[j:j + window_size, :]
                windows.append(window)
                new_label.append(label)

            # Check if current trial is in the selected trials for test set
            if (i + 1) in select_trials:  # Trial indices are 1-based
                test_eeg.append(np.array(windows))
                test_label.append(np.array(new_label))
            else:
                train_eeg.append(np.array(windows))
                train_label.append(np.array(new_label))

        all_train_eeg.append(np.stack(train_eeg, axis=0).reshape(-1, window_size, out_channels))
        all_test_eeg.append(np.stack(test_eeg, axis=0).reshape(-1, window_size, out_channels))
        all_train_label.append(np.stack(train_label, axis=0).reshape(-1, 1))
        all_test_label.append(np.stack(test_label, axis=0).reshape(-1, 1))

    else:
        # KUL
        num_trials = len(labels)
        if num_trials < 10:
            # If fewer than 10 trials, just use the last trial as test
            train_eeg = []
            test_eeg = []
            train_label = []
            test_label = []

            for i in range(num_trials):
                eeg = eeg_datas[i]
                label = labels[i]
                windows = []
                new_label = []
                for j in range(0, eeg.shape[0] - window_size + 1, stride):
                    window = eeg[j:j + window_size, :]
                    windows.append(window)
                    new_label.append(label)

                # If it's the last trial, it's the test set
                if i == num_trials - 1 or i == num_trials - 2:
                    test_eeg.append(np.array(windows))
                    test_label.append(np.array(new_label))
                else:
                    train_eeg.append(np.array(windows))
                    train_label.append(np.array(new_label))

            all_train_eeg.append(np.stack(train_eeg, axis=0).reshape(-1, window_size, out_channels))
            all_test_eeg.append(np.stack(test_eeg, axis=0).reshape(-1, window_size, out_channels))
            all_train_label.append(np.stack(train_label, axis=0).reshape(-1, 1))
            all_test_label.append(np.stack(test_label, axis=0).reshape(-1, 1))

        # DTU
        else:
            # Split into 90% training and 10% testing
            num_train_trials = int(0.9 * num_trials)
            train_trials = range(num_train_trials)
            test_trials = range(num_train_trials, num_trials)

            train_eeg = []
            test_eeg = []
            train_label = []
            test_label = []

            for i in range(num_trials):
                eeg = eeg_datas[i]
                label = labels[i]
                windows = []
                new_label = []
                for j in range(0, eeg.shape[0] - window_size + 1, stride):
                    window = eeg[j:j + window_size, :]
                    windows.append(window)
                    new_label.append(label)

                # Add to the appropriate set based on trial index
                if i in test_trials:
                    test_eeg.append(np.array(windows))
                    test_label.append(np.array(new_label))
                else:
                    train_eeg.append(np.array(windows))
                    train_label.append(np.array(new_label))

            all_train_eeg.append(np.stack(train_eeg, axis=0).reshape(-1, window_size, out_channels))
            all_test_eeg.append(np.stack(test_eeg, axis=0).reshape(-1, window_size, out_channels))
            all_train_label.append(np.stack(train_label, axis=0).reshape(-1, 1))
            all_test_label.append(np.stack(test_label, axis=0).reshape(-1, 1))

    # Convert all collected data into numpy arrays
    all_train_eeg = np.concatenate(all_train_eeg, axis=0)
    all_test_eeg = np.concatenate(all_test_eeg, axis=0)
    all_train_label = np.concatenate(all_train_label, axis=0)
    all_test_label = np.concatenate(all_test_label, axis=0)

    return all_train_eeg, all_test_eeg, all_train_label, all_test_label


def sliding_window_cross_trials_random(sub, eeg_datas, labels, args, out_channels):
    window_size = args.window_length
    stride = int(window_size * (1 - args.overlap))

    all_train_eeg = []
    all_test_eeg = []
    all_train_label = []
    all_test_label = []

    if "AVGCDataset" in args.dataset:
        # Same as before for AVGCDataset
        # #  Moving Videos
        # select_trials = {
        #     "S1": [1, 4], "S3": [1, 4], "S4": [1, 5], "S7": [1, 6],
        #     "S8": [1, 5], "S9": [1, 6], "S10": [1, 6], "S11": [1, 6],
        #     "S12": [1, 6], "S13": [1, 5], "S14": [1, 6], "S15": [1, 6], "S16": [1, 6]}
        # selected_trials = select_trials.get(sub)
        #
        # # Moving Target Noise
        select_trials = {
            "S1": [], "S3": [], "S4": [2, 6], "S7": [2, 3],
            "S8": [2, 6], "S9": [2, 5], "S10": [2, 4], "S11": [2, 3],
            "S12": [2, 3], "S13": [2, 6], "S14": [2, 5], "S15": [2, 4], "S16": [2, 3]}
        selected_trials = select_trials.get(sub)

        ## No Visuals
        # select_trials = {
        #     "S1": [2, 5], "S3": [2, 5], "S4": [3, 7], "S7": [4, 7],
        #     "S8": [3, 7], "S9": [3, 7], "S10": [3, 7], "S11": [4, 7],
        #     "S12": [4, 7], "S13": [3, 7], "S14": [3, 7], "S15": [3, 7], "S16": [4, 7]}
        # selected_trials = select_trials.get(sub)

        # # Fixed Video
        # select_trials = {
        #     "S1": [3, 6], "S3": [3, 6], "S4": [4, 8], "S7": [5, 8],
        #     "S8": [4, 8], "S9": [4, 8], "S10": [5, 8], "S11": [5, 8],
        #     "S12": [5, 8], "S13": [4, 8], "S14": [4], "S15": [5, 8], "S16": [5, 8]}
        # selected_trials = select_trials.get(sub)

        if selected_trials is None:
            raise ValueError(f"Subject {sub} not found in the select_trials dictionary.")
        # if selected_trials is None:
        #     raise ValueError(f"Subject {sub} not found in the unselect_trials dictionary.")

        # Create train and test sets based on the selected trials
        train_eeg = []
        test_eeg = []
        train_label = []
        test_label = []

        for i in range(len(labels)):
            eeg = eeg_datas[i]
            label = labels[i]
            windows = []
            new_label = []
            for j in range(0, eeg.shape[0] - window_size + 1, stride):
                window = eeg[j:j + window_size, :]
                windows.append(window)
                new_label.append(label)

            if (i + 1) in selected_trials:  # Trial indices are 1-based
                test_eeg.append(np.array(windows))
                test_label.append(np.array(new_label))
            # elif (i+1) in unselected_trials:
            #     continue
            else:
                train_eeg.append(np.array(windows))
                train_label.append(np.array(new_label))

        all_train_eeg.append(np.stack(train_eeg, axis=0).reshape(-1, window_size, out_channels))
        all_test_eeg.append(np.stack(test_eeg, axis=0).reshape(-1, window_size, out_channels))
        all_train_label.append(np.stack(train_label, axis=0).reshape(-1, 1))
        all_test_label.append(np.stack(test_label, axis=0).reshape(-1, 1))

    # KUL\DTU
    else:
        num_trials = len(labels)
        # KUL
        if num_trials < 10:

            # 找到标签为0和标签为1的试验索引
            class_0_indices = [i for i in range(num_trials) if labels[i] == 0]
            class_1_indices = [i for i in range(num_trials) if labels[i] == 1]

            # 随机选择一个标签为0的试验和一个标签为1的试验作为测试集
            test_trial_0_index = np.random.choice(class_0_indices, 1)[0]
            test_trial_1_index = np.random.choice(class_1_indices, 1)[0]

            # 剩余的作为训练集
            train_trials = [i for i in range(num_trials) if i != test_trial_0_index and i != test_trial_1_index]

            train_eeg = []
            test_eeg = []
            train_label = []
            test_label = []

            for i in range(num_trials):
                eeg = eeg_datas[i]
                label = labels[i]
                windows = []
                new_label = []
                for j in range(0, eeg.shape[0] - window_size + 1, stride):
                    window = eeg[j:j + window_size, :]
                    windows.append(window)
                    new_label.append(label)

                # 如果是测试集
                if i == test_trial_0_index or i == test_trial_1_index:
                    test_eeg.append(np.array(windows))
                    test_label.append(np.array(new_label))
                else:
                    train_eeg.append(np.array(windows))
                    train_label.append(np.array(new_label))

            # 更新训练集和测试集数据
            all_train_eeg.append(np.stack(train_eeg, axis=0).reshape(-1, window_size, out_channels))
            all_test_eeg.append(np.stack(test_eeg, axis=0).reshape(-1, window_size, out_channels))
            all_train_label.append(np.stack(train_label, axis=0).reshape(-1, 1))
            all_test_label.append(np.stack(test_label, axis=0).reshape(-1, 1))
        # DTU
        else:
            # Randomly select 10% of trials as the test set and the rest as the training set
            num_test_trials = int(0.1 * num_trials)
            test_trial_indices = np.random.choice(num_trials, num_test_trials, replace=False)
            train_trial_indices = [i for i in range(num_trials) if i not in test_trial_indices]

            train_eeg = []
            test_eeg = []
            train_label = []
            test_label = []

            for i in range(num_trials):
                eeg = eeg_datas[i]
                label = labels[i]
                windows = []
                new_label = []
                for j in range(0, eeg.shape[0] - window_size + 1, stride):
                    window = eeg[j:j + window_size, :]
                    windows.append(window)
                    new_label.append(label)

                if i in test_trial_indices:
                    test_eeg.append(np.array(windows))
                    test_label.append(np.array(new_label))
                else:
                    train_eeg.append(np.array(windows))
                    train_label.append(np.array(new_label))

            all_train_eeg.append(np.stack(train_eeg, axis=0).reshape(-1, window_size, out_channels))
            all_test_eeg.append(np.stack(test_eeg, axis=0).reshape(-1, window_size, out_channels))
            all_train_label.append(np.stack(train_label, axis=0).reshape(-1, 1))
            all_test_label.append(np.stack(test_label, axis=0).reshape(-1, 1))

    # Convert all collected data into numpy arrays
    all_train_eeg = np.concatenate(all_train_eeg, axis=0)
    all_test_eeg = np.concatenate(all_test_eeg, axis=0)
    all_train_label = np.concatenate(all_train_label, axis=0)
    all_test_label = np.concatenate(all_test_label, axis=0)

    return all_train_eeg, all_test_eeg, all_train_label, all_test_label



def sliding_window_random(eeg_datas, labels, args, out_channels):
    window_size = args.window_length
    stride = int(window_size * (1 - args.overlap))  # Calculate stride

    train_eeg = []
    valid_eeg = []
    test_eeg = []
    train_label = []
    valid_label = []
    test_label = []

    total_samples = eeg_datas[0].shape[0]  # 6400
    segment_size = int(total_samples * 0.1)  # Size of each segment (for validation and testing) 640
    train_size = int(total_samples - 2 * segment_size)  # Remaining samples for training 5120

    # 随机选择验证集和测试集的起始位置
    start_val = np.random.randint(0, total_samples - 2 * segment_size)  # 随机选择验证集的起始位置
    start_test = np.random.randint(0, total_samples - segment_size)  # 随机选择测试集的起始位置

    # 确保验证集和测试集不重叠
    while abs(start_val - start_test) < segment_size:  # 如果重叠，重新选择
        start_test = np.random.randint(0, total_samples - segment_size)

    # eeg_datas:(60,6400,64)
    for m in range(len(labels)):  # m => trial number
        eeg = eeg_datas[m]  # (6400, 64)
        label = labels[m]

        # 划分验证集、测试集和训练集
        val_data = eeg[start_val:start_val + segment_size]  # 640个采样点作为验证集
        test_data = eeg[start_test:start_test + segment_size]  # 640个采样点作为测试集
        # 修复训练集的划分
        if start_val < start_test:
            # 如果验证集的起始位置在测试集之前
            train_data = np.concatenate([eeg[:start_val], eeg[start_val + segment_size:start_test], eeg[start_test + segment_size:]], axis=0)
        else:
            # 如果验证集的起始位置在测试集之后
            train_data = np.concatenate([eeg[:start_test], eeg[start_test + segment_size:start_val], eeg[start_val + segment_size:]], axis=0)

        # 训练集样本数量是否符合预期
        assert train_data.shape[0] == train_size, f"训练集的样本数不正确: {train_data.shape[0]}"

        # 创建滑动窗口函数
        def create_windows(data):
            windows = []
            for i in range(0, data.shape[0] - window_size + 1, stride):
                window = data[i:i + window_size, :]
                windows.append(window)
            return np.array(windows)

        # 对每个数据集应用滑动窗口
        train_windows = create_windows(train_data)
        val_windows = create_windows(val_data)
        test_windows = create_windows(test_data)

        # 添加到相应的列表中
        train_eeg.append(train_windows)
        valid_eeg.append(val_windows)
        test_eeg.append(test_windows)

        # 为每个窗口分配标签（将标签重复到每个窗口）
        train_label.append([label] * len(train_windows))
        valid_label.append([label] * len(val_windows))
        test_label.append([label] * len(test_windows))

    # for i, trial in enumerate(train_eeg):
    #     print(f"Trial {i} shape: {trial.shape}")
    # 将数据栈起来并重塑形状
    train_eeg = np.stack(train_eeg, axis=0).reshape(-1, window_size, out_channels)

    val_eeg = np.stack(valid_eeg, axis=0).reshape(-1, window_size, out_channels)
    test_eeg = np.stack(test_eeg, axis=0).reshape(-1, window_size, out_channels)
    train_label = np.stack(train_label, axis=0).reshape(-1, 1)
    val_label = np.stack(valid_label, axis=0).reshape(-1, 1)
    test_label = np.stack(test_label, axis=0).reshape(-1, 1)

    return train_eeg, val_eeg, test_eeg, train_label, val_label, test_label


def sliding_window_random_v2(eeg_datas, labels, args, out_channels):
    window_size = args.window_length
    stride = int(window_size * (1 - args.overlap))  # Calculate stride

    train_eeg = []
    valid_eeg = []
    test_eeg = []
    train_label = []
    valid_label = []
    test_label = []

    total_samples = eeg_datas[0].shape[0]  # 6400
    segment_size = int(total_samples * 0.1)  # Size of each segment (for validation and testing)
    train_size = int(total_samples - 2 * segment_size)  # Remaining samples for training

    # 随机选择验证集和测试集的起始位置
    start_val = np.random.randint(0, total_samples - segment_size)  # 随机选择验证集的起始位置
    start_test = np.random.randint(0, total_samples - segment_size)  # 随机选择测试集的起始位置

    # 确保验证集和测试集不重叠
    while (start_test >= start_val and start_test < start_val + segment_size) or \
          (start_val >= start_test and start_val < start_test + segment_size):
        start_test = np.random.randint(0, total_samples - segment_size)

    # eeg_datas:(60,6400,64)
    for m in range(len(labels)):  # m => trial number
        eeg = eeg_datas[m]  # (6400, 64)
        label = labels[m]

        # 划分验证集、测试集和训练集
        val_data = eeg[start_val:start_val + segment_size]  # 640个采样点作为验证集
        test_data = eeg[start_test:start_test + segment_size]  # 640个采样点作为测试集
        # 修复训练集的划分
        if start_val < start_test:
            # 如果验证集的起始位置在测试集之前
            train_data = np.concatenate([eeg[:start_val], eeg[start_val + segment_size:start_test], eeg[start_test + segment_size:]], axis=0)
        else:
            # 如果验证集的起始位置在测试集之后
            train_data = np.concatenate([eeg[:start_test], eeg[start_test + segment_size:start_val], eeg[start_val + segment_size:]], axis=0)

        # 训练集样本数量是否符合预期
        assert train_data.shape[0] == train_size, f"训练集的样本数不正确: {train_data.shape[0]}"

        # 创建滑动窗口函数
        def create_windows(data):
            windows = []
            for i in range(0, data.shape[0] - window_size + 1, stride):
                window = data[i:i + window_size, :]
                windows.append(window)
            return np.array(windows)

        # 对每个数据集应用滑动窗口
        train_windows = create_windows(train_data)
        val_windows = create_windows(val_data)
        test_windows = create_windows(test_data)

        # 添加到相应的列表中
        train_eeg.append(train_windows)
        valid_eeg.append(val_windows)
        test_eeg.append(test_windows)

        # 为每个窗口分配标签（将标签重复到每个窗口）
        train_label.append([label] * len(train_windows))
        valid_label.append([label] * len(val_windows))
        test_label.append([label] * len(test_windows))

    # 将数据栈起来并重塑形状
    train_eeg = np.stack(train_eeg, axis=0).reshape(-1, window_size, out_channels)
    val_eeg = np.stack(valid_eeg, axis=0).reshape(-1, window_size, out_channels)
    test_eeg = np.stack(test_eeg, axis=0).reshape(-1, window_size, out_channels)
    train_label = np.stack(train_label, axis=0).reshape(-1, 1)
    val_label = np.stack(valid_label, axis=0).reshape(-1, 1)
    test_label = np.stack(test_label, axis=0).reshape(-1, 1)

    return train_eeg, val_eeg, test_eeg, train_label, val_label, test_label




def sliding_window_K_fold(eeg_datas, labels, args, out_channels, n_folds=5):
    window_size = args.window_length
    stride = int(window_size * (1 - args.overlap))

    # Preprocess sliding windows for each trial
    eeg_windows = []
    label_windows = []
    for m in range(len(labels)):  # Loop through each trial
        eeg = eeg_datas[m]  # (80176, 64)
        label = labels[m]
        windows = []
        new_labels = []

        # Create sliding windows
        for i in range(0, eeg.shape[0] - window_size + 1, stride):
            window = eeg[i:i + window_size, :]
            windows.append(window)
            new_labels.append(label)

        eeg_windows.append(np.array(windows))  # Shape: (num_windows, window_size, out_channels)
        label_windows.append(np.array(new_labels))  # Shape: (num_windows, )

    # Stack windows across trials
    eeg_windows = np.concatenate(eeg_windows, axis=0)
    label_windows = np.concatenate(label_windows, axis=0)

    # Prepare 5-fold cross-validation split
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
    fold_data = []

    for train_index, test_index in kf.split(eeg_windows):
        train_eeg, test_eeg = eeg_windows[train_index], eeg_windows[test_index]
        train_label, test_label = label_windows[train_index], label_windows[test_index]

        # Reshape if necessary
        train_eeg = train_eeg.reshape(-1, window_size, out_channels)
        test_eeg = test_eeg.reshape(-1, window_size, out_channels)
        train_label = train_label.reshape(-1, 1)
        test_label = test_label.reshape(-1, 1)

        # Store each fold’s train and test data
        fold_data.append((train_eeg, test_eeg, train_label, test_label))

    return fold_data


def read_prepared_data_all_subjects(args):
    data = []
    target = []

    # 遍历所有受试者
    for subject in tqdm(range(1, args.subject_number + 1), desc="Reading data for all subjects"):
        subject_data = []
        subject_target = []

        name = f"S{subject}"
        label = pd.read_csv(
            args.data_document_path + "/" + args.ConType + "/" + name + "/csv/" + name + args.ConType + ".csv"
        )

        for k in range(args.trail_number):
            filename = args.data_document_path + "/" + args.ConType + "/" + name + "/" + name + "Tra" + str(
                k + 1) + ".csv"
            data_pf = pd.read_csv(filename, header=None)
            eeg_data = data_pf.iloc[:, :]  # KUL, DTU
            subject_data.append(eeg_data)
            subject_target.append(label.iloc[k, args.label_col])

        data.append(subject_data)
        target.append(subject_target)

    return data, target

def sliding_window_for_all_subject(eeg_datas, labels, args, out_channels, save_path='/data/wjq/AAD/OpenAAD/processed_data_cross_subjects'):
    # 定义保存文件路径
    save_file = os.path.join(save_path, f'processed_{args.dataset}_data_{args.subject_number}_subjects_{args.length}s.pkl')

    # 检查是否已经存在处理好的数据
    if os.path.exists(save_file):
        print(f"Loading pre-processed data from {save_file}")
        # 直接加载处理好的数据
        with open(save_file, 'rb') as f:
            data = pickle.load(f)
        return data['train_eeg'], data['valid_eeg'], data['test_eeg'], data['train_label'], data['valid_label'], data['test_label']

    # 如果数据没有保存过，则进行数据处理
    print("Processing data...")

    window_size = args.window_length
    stride = int(window_size * (1 - args.overlap))

    train_eeg_list = []
    valid_eeg_list = []
    test_eeg_list = []
    train_label_list = []
    valid_label_list = []
    test_label_list = []

    # 遍历每个受试者的数据
    for subject_idx in tqdm(range(len(eeg_datas)), desc="Processing sliding windows for all subjects"):
        subject_eeg_data = eeg_datas[subject_idx]
        subject_labels = labels[subject_idx]

        train_eeg = []
        test_eeg = []
        train_label = []
        test_label = []

        # 对每个 trial 进行滑动窗口操作
        for trial_idx in tqdm(range(len(subject_labels)), desc=f"Sliding window for subject {subject_idx + 1}", leave=False):
            eeg = subject_eeg_data[trial_idx]
            eeg = eeg.to_numpy()  # 将 DataFrame 转换为 NumPy 数组
            label = subject_labels[trial_idx]

            windows = []
            new_labels = []

            for i in range(0, eeg.shape[0] - window_size + 1, stride):
                window = eeg[i:i + window_size, :]
                windows.append(window)
                new_labels.append(label)

            # 将每个 trial 的片段分为训练集和测试集
            train_eeg.append(np.array(windows)[:int(len(windows) * 0.9)])
            test_eeg.append(np.array(windows)[int(len(windows) * 0.9):])
            train_label.append(np.array(new_labels)[:int(len(windows) * 0.9)])
            test_label.append(np.array(new_labels)[int(len(windows) * 0.9):])

        # 将该受试者的训练和测试数据合并
        train_eeg = np.stack(train_eeg, axis=0).reshape(-1, window_size, out_channels)
        test_eeg = np.stack(test_eeg, axis=0).reshape(-1, window_size, out_channels)
        train_label = np.stack(train_label, axis=0).reshape(-1, 1)
        test_label = np.stack(test_label, axis=0).reshape(-1, 1)

        indices = np.arange(len(train_label))
        np.random.seed(args.seed)
        np.random.shuffle(indices)

        train_eeg = train_eeg[indices]
        train_label = train_label[indices]

        # 进一步从训练集中划分验证集
        train_eeg, valid_eeg, train_label, valid_label = train_test_split(
            train_eeg, train_label, test_size=0.1, random_state=args.seed
        )

        if args.is_CSP:
            # 对训练集进行CSP拟合
            csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space',
                      norm_trace=True)
            train_eeg = train_eeg.transpose(0, 2, 1)  # 变换为 (样本数, 通道数, 窗口大小)
            train_eeg = csp.fit_transform(train_eeg, train_label.squeeze())  # 拟合并转换训练数据
            train_eeg = train_eeg.transpose(0, 2, 1)  # 变换为 (样本数, 通道数, 窗口大小)

            valid_eeg = valid_eeg.transpose(0, 2, 1)
            valid_eeg = csp.transform(valid_eeg)  # 只进行特征转换，不进行拟合
            valid_eeg = valid_eeg.transpose(0, 2, 1)  # 变换为 (样本数, 通道数, 窗口大小)

            test_eeg = test_eeg.transpose(0, 2, 1)
            test_eeg = csp.transform(test_eeg)  # 只进行特征转换，不进行拟合
            test_eeg = test_eeg.transpose(0, 2, 1)

        # 将该受试者的数据加入全体列表
        train_eeg_list.append(train_eeg)
        valid_eeg_list.append(valid_eeg)
        test_eeg_list.append(test_eeg)
        train_label_list.append(train_label)
        valid_label_list.append(valid_label)
        test_label_list.append(test_label)

    # 合并所有受试者的数据
    all_train_eeg = np.concatenate(train_eeg_list, axis=0)
    all_valid_eeg = np.concatenate(valid_eeg_list, axis=0)
    all_test_eeg = np.concatenate(test_eeg_list, axis=0)
    all_train_label = np.concatenate(train_label_list, axis=0)
    all_valid_label = np.concatenate(valid_label_list, axis=0)
    all_test_label = np.concatenate(test_label_list, axis=0)

    # 保存处理好的数据
    os.makedirs(save_path, exist_ok=True)
    with open(save_file, 'wb') as f:
        pickle.dump({
            'train_eeg': all_train_eeg,
            'valid_eeg': all_valid_eeg,
            'test_eeg': all_test_eeg,
            'train_label': all_train_label,
            'valid_label': all_valid_label,
            'test_label': all_test_label
        }, f)

    print(f"Data processed and saved to {save_file}")

    return all_train_eeg, all_valid_eeg, all_test_eeg, all_train_label, all_valid_label, all_test_label


def sliding_window_LOSO(eeg_datas, labels, args, out_channels, save_path='/data/wjq/AAD/OpenAAD/processed_data_LOSO'):
    """
    使用留一被试法（LOSO）对 EEG 数据进行滑动窗口处理，并对每个受试者的数据单独应用 CSP 特征提取。处理后的每个折叠的数据将被保存或返回。

    参数：
        eeg_datas (list): 所有受试者的 EEG 数据列表，每个元素对应一个受试者的数据。
        labels (list): 所有受试者的标签列表，每个元素对应一个受试者的标签。
        args (Namespace): 包含各种参数的命名空间（如窗口长度、重叠比例、随机种子等）。
        out_channels (int): 输出的通道数量，用于重塑数据形状。
        save_path (str): 保存处理后数据的根目录路径。

    返回：
        folds_data (list): 每个折叠的数据字典列表，包含 train_eeg, test_eeg, train_label, test_label。
        csp_models (list): 每个折叠的 CSP 模型字典，包含每个受试者的 CSP 模型。
    """

    num_subjects = len(eeg_datas)
    folds_data = []

    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    for test_subject_idx in tqdm(range(num_subjects), desc="LOSO Folds"):
        # 定义当前折叠的保存文件路径
        save_file = os.path.join(
            save_path,
            f'fold_{test_subject_idx + 1}_processed_{args.dataset}_data_{args.subject_number}_subjects_{args.length}s.pkl'
        )

        # 检查是否已经存在当前折叠的处理数据
        if os.path.exists(save_file):
            print(f"Loading pre-processed data for fold {test_subject_idx + 1} from {save_file}")
            with open(save_file, 'rb') as f:
                fold_data = pickle.load(f)
            folds_data.append(fold_data)
            continue  # 跳过当前折叠的处理，继续下一个折叠

        print(f"Processing fold {test_subject_idx + 1} where subject {test_subject_idx + 1} is the test set...")

        window_size = args.window_length
        stride = int(window_size * (1 - args.overlap))

        train_eeg_list = []
        # valid_eeg_list = []
        test_eeg_list = []
        train_label_list = []
        # valid_label_list = []
        test_label_list = []

        # 定义训练集和测试集的受试者索引
        test_subject = test_subject_idx
        train_subjects = [i for i in range(num_subjects) if i != test_subject]

        # 处理训练集受试者的数据
        for subject_idx in tqdm(train_subjects, desc=f"Processing training subjects for fold {test_subject + 1}",leave=False):
            subject_eeg_data = eeg_datas[subject_idx]
            subject_labels = labels[subject_idx]

            for trial_idx in range(len(subject_labels)):
                eeg = subject_eeg_data[trial_idx]
                eeg = eeg.to_numpy()  # 将 DataFrame 转换为 NumPy 数组
                label = subject_labels[trial_idx]

                for i in range(0, eeg.shape[0] - window_size + 1, stride):
                    window = eeg[i:i + window_size, :]
                    # # 划分训练集和验证集
                    # if np.random.rand() < 0.1:
                    #     valid_eeg_list.append(window)
                    #     valid_label_list.append(label)
                    # else:
                    train_eeg_list.append(window)
                    train_label_list.append(label)

        # 处理测试集受试者的数据
        subject_eeg_data = eeg_datas[test_subject]
        subject_labels = labels[test_subject]

        for trial_idx in range(len(subject_labels)):
            eeg = subject_eeg_data[trial_idx]
            eeg = eeg.to_numpy()  # 将 DataFrame 转换为 NumPy 数组
            label = subject_labels[trial_idx]

            for i in range(0, eeg.shape[0] - window_size + 1, stride):
                window = eeg[i:i + window_size, :]
                test_eeg_list.append(window)
                test_label_list.append(label)

        # 转换为 NumPy 数组
        train_eeg = np.array(train_eeg_list).reshape(-1, window_size, out_channels)
        # valid_eeg = np.array(valid_eeg_list).reshape(-1, window_size, out_channels)
        test_eeg = np.array(test_eeg_list).reshape(-1, window_size, out_channels)
        train_label = np.array(train_label_list).reshape(-1, 1)
        # valid_label = np.array(valid_label_list).reshape(-1, 1)
        test_label = np.array(test_label_list).reshape(-1, 1)

        # 随机打乱训练集
        indices = np.arange(len(train_label))
        np.random.seed(args.seed)
        np.random.shuffle(indices)
        train_eeg = train_eeg[indices]
        train_label = train_label[indices]


        # CSP 处理
        if args.is_CSP:
            csp = CSP(
                n_components=args.csp_comp,
                reg=None,
                log=None,
                cov_est='concat',
                transform_into='csp_space',
                norm_trace=True
            )
            # 训练 CSP
            train_eeg_csp = train_eeg.transpose(0, 2, 1)  # (样本数, 通道数, 窗口大小)
            csp.fit(train_eeg_csp, train_label.squeeze())
            train_eeg = csp.transform(train_eeg_csp).transpose(0, 2, 1)

            # 转换验证集和测试集
            valid_eeg_csp = valid_eeg.transpose(0, 2, 1)
            valid_eeg = csp.transform(valid_eeg_csp).transpose(0, 2, 1)

            test_eeg_csp = test_eeg.transpose(0, 2, 1)
            test_eeg = csp.transform(test_eeg_csp).transpose(0, 2, 1)

        # 构建当前折叠的数据字典
        fold_data = {
            'train_eeg': train_eeg,
            'test_eeg': test_eeg,
            'train_label': train_label,
            'test_label': test_label
        }

        # 保存当前折叠的数据
        with open(save_file, 'wb') as f:
            pickle.dump(fold_data, f)
        print(f"Fold {test_subject_idx + 1} data processed and saved to {save_file}")

        # 将当前折叠的数据添加到列表中
        folds_data.append(fold_data)

    print("All folds processed.")
    return folds_data


def sliding_window_LOSO_CSP_per_subject(eeg_datas, labels, args, out_channels,
                                        save_path='/data/wjq/AAD/OpenAAD/cross_subjects_datasets/processed_data_LOSO_0115'):
    """
    使用留一被试法（LOSO）对 EEG 数据进行滑动窗口处理，并对每个受试者的数据单独应用 CSP 特征提取。处理后的每个折叠的数据将被保存或返回。

    参数：
        eeg_datas (list): 所有受试者的 EEG 数据列表，每个元素对应一个受试者的数据。
        labels (list): 所有受试者的标签列表，每个元素对应一个受试者的标签。
        args (Namespace): 包含各种参数的命名空间（如窗口长度、重叠比例、随机种子等）。
        out_channels (int): 输出的通道数量，用于重塑数据形状。
        save_path (str): 保存处理后数据的根目录路径。

    返回：
        folds_data (list): 每个折叠的数据字典列表，包含 train_eeg, test_eeg, train_label, test_label。
        csp_models (list): 每个折叠的 CSP 模型字典，包含每个受试者的 CSP 模型。
    """

    num_subjects = len(eeg_datas)
    folds_data = []
    csp_models = []

    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    for test_subject_idx in tqdm(range(num_subjects), desc="LOSO Folds"):
        # 定义当前折叠的保存文件路径
        save_file = os.path.join(
            save_path,
            f'fold_{test_subject_idx + 1}_processed_{args.dataset}_data_{args.subject_number}_subjects_{args.length}s.pkl'
        )

        # 检查是否已经存在当前折叠的处理数据
        if os.path.exists(save_file):
            print(f"Loading pre-processed data for fold {test_subject_idx + 1} from {save_file}")
            with open(save_file, 'rb') as f:
                fold_data = pickle.load(f)
            folds_data.append(fold_data)
            continue  # 跳过当前折叠的处理，继续下一个折叠

        print(f"Processing fold {test_subject_idx + 1} where subject {test_subject_idx + 1} is the test set...")

        window_size = args.window_length
        stride = int(window_size * (1 - args.overlap))

        train_eeg_list = []
        test_eeg_list = []
        train_label_list = []
        test_label_list = []

        # CSP models for this fold
        # fold_csp_models = {}

        # 定义训练集和测试集的受试者索引
        test_subject = test_subject_idx
        train_subjects = [i for i in range(num_subjects) if i != test_subject]

        # 处理训练集受试者的数据
        for subject_idx in tqdm(train_subjects, desc=f"Processing training subjects for fold {test_subject + 1}",
                                leave=False):
            subject_eeg_data = eeg_datas[subject_idx]
            subject_labels = labels[subject_idx]

            subject_windows = []
            subject_train_label_list = []  # 使用单独的变量来避免混淆

            for trial_idx in range(len(subject_labels)):
                eeg = subject_eeg_data[trial_idx]
                eeg = eeg.to_numpy()  # 将 DataFrame 转换为 NumPy 数组
                label = subject_labels[trial_idx]

                for i in range(0, eeg.shape[0] - window_size + 1, stride):
                    window = eeg[i:i + window_size, :]
                    subject_windows.append(window)
                    subject_train_label_list.append(label)

            # 转换为 NumPy 数组
            subject_eeg = np.array(subject_windows).reshape(-1, window_size, out_channels)
            subject_label_array = np.array(subject_train_label_list).reshape(-1, 1)

            # 随机打乱训练集数据
            indices = np.arange(len(subject_label_array))
            np.random.seed(args.seed)
            np.random.shuffle(indices)
            subject_eeg = subject_eeg[indices]
            subject_label_array = subject_label_array[indices]

            # 训练 CSP 并转换数据
            if args.is_CSP:
                csp = CSP(
                    n_components=args.csp_comp,
                    reg=None,
                    log=None,
                    cov_est='concat',
                    transform_into='csp_space',
                    norm_trace=True
                )
                subject_eeg_csp = subject_eeg.transpose(0, 2, 1)  # (样本数, 通道数, 窗口大小)
                csp.fit(subject_eeg_csp, subject_label_array.squeeze())
                subject_eeg_transformed = csp.transform(subject_eeg_csp).transpose(0, 2, 1)
                # 存储该受试者的 CSP 模型
                # fold_csp_models[f'subject_{subject_idx + 1}'] = csp
            else:
                subject_eeg_transformed = subject_eeg

            # 将转换后的数据加入训练集列表
            train_eeg_list.append(subject_eeg_transformed)
            train_label_list.append(subject_label_array)

        # 将所有训练集数据拼接
        train_eeg = np.concatenate(train_eeg_list, axis=0)
        train_labels = np.concatenate(train_label_list, axis=0)

        # 处理测试集受试者的数据
        # print(f"Processing test subject {test_subject + 1}...")
        test_subject_eeg_data = eeg_datas[test_subject]
        test_subject_labels = labels[test_subject]

        # test_windows = []
        for trial_idx in range(len(test_subject_labels)):
            eeg = test_subject_eeg_data[trial_idx]
            eeg = eeg.to_numpy()
            label = test_subject_labels[trial_idx]

            for i in range(0, eeg.shape[0] - window_size + 1, stride):
                window = eeg[i:i + window_size, :]
                test_eeg_list.append(window)
                test_label_list.append(label)

        test_eeg = np.array(test_eeg_list).reshape(-1, window_size, out_channels)
        test_labels = np.array(test_label_list).reshape(-1, 1)

        # 如果需要对测试集应用 CSP（通常不需要，但根据需求调整）
        if args.is_CSP:
            # csp_test = CSP(
            #     n_components=args.csp_comp,
            #     reg=None,
            #     log=None,
            #     cov_est='concat',
            #     transform_into='csp_space',
            #     norm_trace=True
            # )
            test_eeg_csp = test_eeg.transpose(0, 2, 1)
            test_eeg_transformed = csp.transform(test_eeg_csp).transpose(0, 2, 1)

        else:
            test_eeg_transformed = test_eeg

        # 赋值转换后的测试集数据
        test_eeg = test_eeg_transformed if args.is_CSP else test_eeg

        # 构建当前折叠的数据字典
        fold_data = {
            'train_eeg': train_eeg,
            'test_eeg': test_eeg,
            'train_label': train_labels,
            'test_label': test_labels,
        }

        # 保存当前折叠的数据
        with open(save_file, 'wb') as f:
            pickle.dump(fold_data, f)
        print(f"Fold {test_subject_idx + 1} data processed and saved to {save_file}")

        # 将当前折叠的数据和 CSP 模型添加到列表中
        folds_data.append(fold_data)
        # csp_models.append(fold_csp_models)

    print("All folds processed.")
    return folds_data


def sliding_window_LOSO_CSP_all_subject(eeg_datas, labels, args, out_channels, save_path):
    """
    使用留一被试法（LOSO）对 EEG 数据进行滑动窗口处理，并统一拟合一个 CSP 模型。
    处理后的每个折叠的数据将被保存或返回。

    参数：
        eeg_datas (list): 所有受试者的 EEG 数据列表，每个元素对应一个受试者的数据。
        labels (list): 所有受试者的标签列表，每个元素对应一个受试者的标签。
        args (Namespace): 包含各种参数的命名空间（如窗口长度、重叠比例、随机种子等）。
        out_channels (int): 输出的通道数量，用于重塑数据形状。
        save_path (str): 保存处理后数据的根目录路径。

    返回：
        folds_data (list): 每个折叠的数据字典列表，包含
            'train_eeg', 'test_eeg',
            'train_label', 'test_label'。
        csp_models (list): 每个折叠的 CSP 模型。
    """

    num_subjects = len(eeg_datas)
    folds_data = []


    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    for test_subject_idx in tqdm(range(num_subjects), desc="LOSO Folds"):
        # 定义当前折叠的保存文件路径
        save_file = os.path.join(
            save_path,
            f'fold_{test_subject_idx + 1}_processed_{args.dataset}_data_{args.subject_number}_subjects_{args.length}s.pkl'
        )

        # 如果已经存在当前折叠的处理数据，则直接加载
        if os.path.exists(save_file):
            print(f"Loading pre-processed data for fold {test_subject_idx + 1} from {save_file}")
            with open(save_file, 'rb') as f:
                fold_data = pickle.load(f)
            folds_data.append(fold_data)
            continue  # 跳过当前折叠的处理，继续下一个折叠

        print(f"\nProcessing fold {test_subject_idx + 1} where subject {test_subject_idx + 1} is the test set...")

        window_size = args.window_length
        stride = int(window_size * (1 - args.overlap))

        # 收集所有训练和测试数据
        train_eeg_list = []
        train_label_list = []
        test_eeg_list = []
        test_label_list = []

        # 定义训练集和测试集的受试者索引
        test_subject = test_subject_idx
        train_subjects = [i for i in range(num_subjects) if i != test_subject]

        # ========== 处理训练集受试者的数据 ==========
        for subject_idx in tqdm(train_subjects, desc=f"[Fold {test_subject_idx+1}] Processing training subjects",
                                leave=False):
            subject_eeg_data = eeg_datas[subject_idx]
            subject_labels = labels[subject_idx]

            subject_windows = []
            subject_label_array = []

            # --- 滑窗 ---
            for trial_idx in range(len(subject_labels)):
                eeg = subject_eeg_data[trial_idx].to_numpy()  # 转换为 NumPy 数组
                label = subject_labels[trial_idx]

                # 将每个 trial 按指定滑窗和步长切片
                for i in range(0, eeg.shape[0] - window_size + 1, stride):
                    window = eeg[i:i + window_size, :]
                    subject_windows.append(window)
                    subject_label_array.append(label)

            # 转换为 NumPy 数组
            subject_eeg_arr = np.array(subject_windows).reshape(-1, window_size, out_channels)
            subject_label_arr = np.array(subject_label_array).reshape(-1, 1)

            # --- 打乱 ---
            np.random.seed(args.seed)
            indices = np.arange(len(subject_label_arr))
            np.random.shuffle(indices)
            subject_eeg_arr = subject_eeg_arr[indices]
            subject_label_arr = subject_label_arr[indices]

            # --- 将该受试者的训练集加入全局列表 ---
            train_eeg_list.append(subject_eeg_arr)
            train_label_list.append(subject_label_arr)

        # 将所有训练集数据拼接
        train_eeg = np.concatenate(train_eeg_list, axis=0) if len(train_eeg_list) > 0 else np.array([])
        train_labels = np.concatenate(train_label_list, axis=0) if len(train_label_list) > 0 else np.array([])

        # ========== 拟合 CSP 模型 ==========
        if args.is_CSP:
            # 拟合 CSP 在所有训练数据上
            csp = CSP(
                n_components=args.csp_comp,
                reg=None,
                log=None,
                cov_est='concat',
                transform_into='csp_space',
                norm_trace=True
            )
            # CSP 需要的数据格式为 (n_samples, n_channels, n_times)
            train_eeg_csp = train_eeg.transpose(0, 2, 1)
            csp.fit(train_eeg_csp, train_labels.squeeze())

            # 转换训练集
            train_eeg = csp.transform(train_eeg_csp).transpose(0, 2, 1)
        else:
            train_eeg = train_eeg

        # ========== 处理测试集受试者的数据 ==========
        test_subject_eeg_data = eeg_datas[test_subject]
        test_subject_labels = labels[test_subject]

        for trial_idx in range(len(test_subject_labels)):
            eeg = test_subject_eeg_data[trial_idx].to_numpy()
            label = test_subject_labels[trial_idx]

            for i in range(0, eeg.shape[0] - window_size + 1, stride):
                window = eeg[i:i + window_size, :]
                test_eeg_list.append(window)
                test_label_list.append(label)

        test_eeg = np.array(test_eeg_list).reshape(-1, window_size, out_channels)
        test_labels = np.array(test_label_list).reshape(-1, 1)

        # 转换测试集
        if args.is_CSP:
            # 使用拟合好的 CSP 模型转换测试集
            test_eeg_csp = test_eeg.transpose(0, 2, 1)
            test_eeg = csp.transform(test_eeg_csp).transpose(0, 2, 1)
        else:
            test_eeg = test_eeg

        # 构建当前折叠的数据字典
        fold_data = {
            'train_eeg': train_eeg,
            'train_label': train_labels,
            'test_eeg': test_eeg,
            'test_label': test_labels,
        }

        # 保存当前折叠的数据
        with open(save_file, 'wb') as f:
            pickle.dump(fold_data, f)
        print(f"Fold {test_subject_idx + 1} data processed and saved to {save_file}")

        # 将当前折叠的数据添加到列表中
        # folds_data.append(fold_data)
    #
    # print("All folds processed.")
    # return folds_data

def sliding_window_LOSO_CSP_add_valid(eeg_datas, labels, args, out_channels, save_path):
    """
    使用留一被试法（LOSO）对 EEG 数据进行滑动窗口处理，并在每个受试者上拆分出训练集（80%）与验证集（20%），
    之后将测试受试者的全部数据作为测试集。可选地对训练数据及验证、测试数据应用 CSP 特征提取。

    参数：
        eeg_datas (list): 所有受试者的 EEG 数据列表，每个元素对应一个受试者的数据。
        labels (list): 所有受试者的标签列表，每个元素对应一个受试者的标签。
        args (Namespace): 包含各种参数的命名空间（如窗口长度、重叠比例、随机种子等）。
        out_channels (int): 输出的通道数量，用于重塑数据形状。
        save_path (str): 保存处理后数据的根目录路径。

    返回：
        folds_data (list): 每个折叠的数据字典列表，包含
            'train_eeg', 'val_eeg', 'test_eeg',
            'train_label', 'val_label', 'test_label'。
    """

    num_subjects = len(eeg_datas)
    folds_data = []


    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    for test_subject_idx in tqdm(range(num_subjects), desc="LOSO Folds"):
        # 定义当前折叠的保存文件路径
        save_file = os.path.join(
            save_path,
            f'fold_{test_subject_idx + 1}_processed_{args.dataset}_data_{args.subject_number}_subjects_{args.length}s.pkl'
        )

        # 如果已经存在当前折叠的处理数据，则直接加载
        if os.path.exists(save_file):
            print(f"Loading pre-processed data for fold {test_subject_idx + 1} from {save_file}")
            with open(save_file, 'rb') as f:
                fold_data = pickle.load(f)
            folds_data.append(fold_data)
            continue  # 跳过当前折叠的处理，继续下一个折叠

        print(f"\nProcessing fold {test_subject_idx + 1} where subject {test_subject_idx + 1} is the test set...")

        window_size = args.window_length
        stride = int(window_size * (1 - args.overlap))

        # 收集所有训练/验证/测试数据
        train_eeg_list = []
        train_label_list = []
        val_eeg_list = []
        val_label_list = []
        test_eeg_list = []
        test_label_list = []

        # 定义训练集和测试集的受试者索引
        test_subject = test_subject_idx
        train_subjects = [i for i in range(num_subjects) if i != test_subject]

        # ========== 处理训练集（并拆分验证集）受试者的数据 ==========
        for subject_idx in tqdm(train_subjects, desc=f"[Fold {test_subject_idx+1}] Processing training subjects",
                                leave=False):
            subject_eeg_data = eeg_datas[subject_idx]
            subject_labels = labels[subject_idx]

            subject_windows = []
            subject_label_array = []

            # --- 滑窗 ---
            for trial_idx in range(len(subject_labels)):
                eeg = subject_eeg_data[trial_idx].to_numpy()  # 转换为 NumPy 数组
                label = subject_labels[trial_idx]

                # 将每个 trial 按指定滑窗和步长切片
                for i in range(0, eeg.shape[0] - window_size + 1, stride):
                    window = eeg[i:i + window_size, :]
                    subject_windows.append(window)
                    subject_label_array.append(label)

            # 转换为 NumPy 数组
            subject_eeg_arr = np.array(subject_windows).reshape(-1, window_size, out_channels)
            subject_label_arr = np.array(subject_label_array).reshape(-1, 1)

            # --- 打乱 ---
            np.random.seed(args.seed)
            indices = np.arange(len(subject_label_arr))
            np.random.shuffle(indices)
            subject_eeg_arr = subject_eeg_arr[indices]
            subject_label_arr = subject_label_arr[indices]

            # --- 拆分训练集(80%)和验证集(20%) ---
            split_idx = int(0.8 * len(subject_label_arr))
            subject_train_eeg = subject_eeg_arr[:split_idx]
            subject_train_label = subject_label_arr[:split_idx]
            subject_val_eeg = subject_eeg_arr[split_idx:]
            subject_val_label = subject_label_arr[split_idx:]

            # --- 如果需要使用 CSP, 则只在本受试者的训练部分上 fit ---
            if args.is_CSP:
                # 拟合 CSP
                csp = CSP(
                    n_components=args.csp_comp,
                    reg=None,
                    log=None,
                    cov_est='concat',
                    transform_into='csp_space',
                    norm_trace=True
                )

                # (n_samples, n_channels, n_times) 格式来 fit
                subject_train_eeg_csp = subject_train_eeg.transpose(0, 2, 1)
                csp.fit(subject_train_eeg_csp, subject_train_label.squeeze())

                # 对训练部分进行变换
                subject_train_eeg_transformed = csp.transform(subject_train_eeg_csp)
                # 再转回 (n_samples, n_times, n_channels) 形式
                subject_train_eeg_transformed = subject_train_eeg_transformed.transpose(0, 2, 1)

                # 对验证部分进行变换
                subject_val_eeg_csp = subject_val_eeg.transpose(0, 2, 1)
                subject_val_eeg_transformed = csp.transform(subject_val_eeg_csp)
                subject_val_eeg_transformed = subject_val_eeg_transformed.transpose(0, 2, 1)

            else:
                subject_train_eeg_transformed = subject_train_eeg
                subject_val_eeg_transformed = subject_val_eeg

            # --- 将该受试者处理后的训练集与验证集加入全局列表 ---
            train_eeg_list.append(subject_train_eeg_transformed)
            train_label_list.append(subject_train_label)
            val_eeg_list.append(subject_val_eeg_transformed)
            val_label_list.append(subject_val_label)

        # 将所有训练集数据、验证集数据分别拼接
        train_eeg = np.concatenate(train_eeg_list, axis=0) if len(train_eeg_list) > 0 else np.array([])
        train_labels = np.concatenate(train_label_list, axis=0) if len(train_label_list) > 0 else np.array([])
        val_eeg = np.concatenate(val_eeg_list, axis=0) if len(val_eeg_list) > 0 else np.array([])
        val_labels = np.concatenate(val_label_list, axis=0) if len(val_label_list) > 0 else np.array([])

        # ========== 处理测试集受试者的数据 ==========
        test_subject_eeg_data = eeg_datas[test_subject]
        test_subject_labels = labels[test_subject]

        for trial_idx in range(len(test_subject_labels)):
            eeg = test_subject_eeg_data[trial_idx].to_numpy()
            label = test_subject_labels[trial_idx]

            for i in range(0, eeg.shape[0] - window_size + 1, stride):
                window = eeg[i:i + window_size, :]
                test_eeg_list.append(window)
                test_label_list.append(label)

        test_eeg = np.array(test_eeg_list).reshape(-1, window_size, out_channels)
        test_labels = np.array(test_label_list).reshape(-1, 1)

        # 如果需要对测试集应用 CSP（保持与原逻辑一致，这里对测试集重新 fit）
        if args.is_CSP:
            csp_test = CSP(
                n_components=args.csp_comp,
                reg=None,
                log=None,
                cov_est='concat',
                transform_into='csp_space',
                norm_trace=True
            )
            test_eeg_csp = test_eeg.transpose(0, 2, 1)
            test_eeg_transformed = csp_test.transform(test_eeg_csp).transpose(0, 2, 1)
        else:
            test_eeg_transformed = test_eeg

        # 构建当前折叠的数据字典
        fold_data = {
            'train_eeg': train_eeg,
            'train_label': train_labels,
            'val_eeg': val_eeg,
            'val_label': val_labels,
            'test_eeg': test_eeg_transformed if args.is_CSP else test_eeg,
            'test_label': test_labels
        }

        # 保存当前折叠的数据
        with open(save_file, 'wb') as f:
            pickle.dump(fold_data, f)
        print(f"Fold {test_subject_idx + 1} data processed and saved to {save_file}")

        # 将当前折叠的数据添加到列表中
        # folds_data.append(fold_data)
    #
    # print("All folds processed.")
    # return folds_data

def sliding_window_LOSO_DE_add_valid(eeg_datas, labels, args, out_channels, save_path):
    """
    使用留一被试法（LOSO）对 EEG 数据进行滑动窗口处理，并在每个受试者上拆分出训练集（80%）与验证集（20%），
    之后将测试受试者的全部数据作为测试集。可选地对训练数据及验证、测试数据应用 CSP 特征提取。

    参数：
        eeg_datas (list): 所有受试者的 EEG 数据列表，每个元素对应一个受试者的数据。
        labels (list): 所有受试者的标签列表，每个元素对应一个受试者的标签。
        args (Namespace): 包含各种参数的命名空间（如窗口长度、重叠比例、随机种子等）。
        out_channels (int): 输出的通道数量，用于重塑数据形状。
        save_path (str): 保存处理后数据的根目录路径。

    返回：
        folds_data (list): 每个折叠的数据字典列表，包含
            'train_eeg', 'val_eeg', 'test_eeg',
            'train_label', 'val_label', 'test_label'。
    """

    num_subjects = len(eeg_datas)
    folds_data = []


    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    # ============ 定义一个简化的 DE 计算函数示例 ============ #
    def compute_de_features(eeg_data, args):
        """
        计算 DE 特征。
        参数:
            eeg_data (numpy.ndarray): EEG 数据，形状为 (n_samples, window_size, n_channels)
            args (Namespace): 包含参数的命名空间
        返回:
            list: 各频段的 DE 特征列表（或可直接堆叠为一个 ndarray）
        """
        # 计算五个频段的 DE 特征
        de_features = [
            to_alpha0(eeg_data, args),
            to_alpha1(eeg_data, args),
            to_alpha2(eeg_data, args),
            to_alpha3(eeg_data, args),
            to_alpha4(eeg_data, args)
        ]
        return de_features

    # ============ 主循环：遍历所有被试者，LOS0 ============ #
    for test_subject_idx in tqdm(range(num_subjects), desc="LOSO Folds"):
        # 定义当前折叠的保存文件路径
        save_file = os.path.join(
            save_path,
            f'fold_{test_subject_idx + 1}_processed_{args.dataset}_data_{args.subject_number}_subjects_{args.length}s.pkl'
        )

        # 如果已经存在当前折叠的处理数据，则直接加载
        if os.path.exists(save_file):
            print(f"Loading pre-processed data for fold {test_subject_idx + 1} from {save_file}")
            with open(save_file, 'rb') as f:
                fold_data = pickle.load(f)
            folds_data.append(fold_data)
            continue  # 跳过当前折叠的处理，继续下一个折叠

        print(f"\nProcessing fold {test_subject_idx + 1} where subject {test_subject_idx + 1} is the test set...")

        window_size = args.window_length
        stride = int(window_size * (1 - args.overlap))

        # 收集所有训练/验证/测试数据
        train_eeg_list = []
        train_label_list = []
        val_eeg_list = []
        val_label_list = []
        test_eeg_list = []
        test_label_list = []

        # 定义训练集和测试集的受试者索引
        test_subject = test_subject_idx
        train_subjects = [i for i in range(num_subjects) if i != test_subject]

        # ========== 处理训练集（并拆分验证集）受试者的数据 ==========
        for subject_idx in tqdm(train_subjects, desc=f"[Fold {test_subject_idx+1}] Processing training subjects",
                                leave=False):
            subject_eeg_data = eeg_datas[subject_idx]
            subject_labels = labels[subject_idx]

            subject_windows = []
            subject_label_array = []

            # --- 滑窗 ---
            for trial_idx in range(len(subject_labels)):
                eeg = subject_eeg_data[trial_idx].to_numpy()  # 转换为 NumPy 数组
                label = subject_labels[trial_idx]

                # 将每个 trial 按指定滑窗和步长切片
                for i in range(0, eeg.shape[0] - window_size + 1, stride):
                    window = eeg[i:i + window_size, :]
                    subject_windows.append(window)
                    subject_label_array.append(label)

            # 转换为 NumPy 数组
            subject_eeg_arr = np.array(subject_windows).reshape(-1, window_size, out_channels)
            subject_label_arr = np.array(subject_label_array).reshape(-1, 1)

            # --- 打乱 ---
            np.random.seed(args.seed)
            indices = np.arange(len(subject_label_arr))
            np.random.shuffle(indices)
            subject_eeg_arr = subject_eeg_arr[indices]
            subject_label_arr = subject_label_arr[indices]

            # --- 拆分训练集(80%)和验证集(20%) ---
            split_idx = int(0.8 * len(subject_label_arr))
            subject_train_eeg = subject_eeg_arr[:split_idx]
            subject_train_label = subject_label_arr[:split_idx]
            subject_val_eeg = subject_eeg_arr[split_idx:]
            subject_val_label = subject_label_arr[split_idx:]

            subject_train_eeg_transformed = subject_train_eeg
            subject_val_eeg_transformed = subject_val_eeg

            # --- 将该受试者处理后的训练集与验证集加入全局列表 ---
            train_eeg_list.append(subject_train_eeg_transformed)
            train_label_list.append(subject_train_label)
            val_eeg_list.append(subject_val_eeg_transformed)
            val_label_list.append(subject_val_label)

        # 将所有训练集数据、验证集数据分别拼接
        train_eeg = np.concatenate(train_eeg_list, axis=0) if len(train_eeg_list) > 0 else np.array([])
        train_labels = np.concatenate(train_label_list, axis=0) if len(train_label_list) > 0 else np.array([])
        val_eeg = np.concatenate(val_eeg_list, axis=0) if len(val_eeg_list) > 0 else np.array([])
        val_labels = np.concatenate(val_label_list, axis=0) if len(val_label_list) > 0 else np.array([])

        # ========== 处理测试集受试者的数据 ==========
        test_subject_eeg_data = eeg_datas[test_subject]
        test_subject_labels = labels[test_subject]

        for trial_idx in range(len(test_subject_labels)):
            eeg = test_subject_eeg_data[trial_idx].to_numpy()
            label = test_subject_labels[trial_idx]

            for i in range(0, eeg.shape[0] - window_size + 1, stride):
                window = eeg[i:i + window_size, :]
                test_eeg_list.append(window)
                test_label_list.append(label)

        test_eeg = np.array(test_eeg_list).reshape(-1, window_size, out_channels)
        test_labels = np.array(test_label_list).reshape(-1, 1)

        # ========== 计算 DE 特征（基于原始滑窗数据） ========== #
        train_de_features = compute_de_features(train_eeg, args)
        val_de_features = compute_de_features(val_eeg, args) if val_eeg.size else []
        test_de_features = compute_de_features(test_eeg, args) if test_eeg.size else []

        # 如果需要将 DE 特征转换为图像
        train_de_features = [gen_images(feat, args) for feat in train_de_features]
        val_de_features = [gen_images(feat, args) for feat in val_de_features] if val_de_features else []
        test_de_features = [gen_images(feat, args) for feat in test_de_features] if test_de_features else []

        # 将五个频段的 DE 特征堆叠
        # 注：请根据自己的 DE 特征形状来决定如何 stack 或者 concat
        de_train_eeg = np.stack(train_de_features, axis=1)  # 形如 (N, 5, ...)
        de_val_eeg = np.stack(val_de_features, axis=1) if len(val_de_features) else np.array([])
        de_test_eeg = np.stack(test_de_features, axis=1) if len(test_de_features) else np.array([])

        de_train_eeg = np.expand_dims(de_train_eeg, axis=-1)
        de_val_eeg = np.expand_dims(de_val_eeg, axis=-1)
        de_test_eeg = np.expand_dims(de_test_eeg, axis=-1)

        # 构建当前折叠的数据字典
        fold_data = {
            'train_eeg': de_train_eeg,
            'train_label': train_labels,
            'val_eeg': de_val_eeg,
            'val_label': val_labels,
            'test_eeg': de_test_eeg,
            'test_label': test_labels
        }

        # 保存当前折叠的数据
        with open(save_file, 'wb') as f:
            pickle.dump(fold_data, f)
        print(f"Fold {test_subject_idx + 1} data processed and saved to {save_file}")

def sliding_window_LOSO_CSP_DE_add_valid(eeg_datas, labels, args, out_channels, save_path):
    num_subjects = len(eeg_datas)
    folds_data = []
    os.makedirs(save_path, exist_ok=True)

    # 简化的DE计算函数
    def compute_de_features(eeg_data, args):
        de_features = [
            to_alpha0(eeg_data, args),
            to_alpha1(eeg_data, args),
            to_alpha2(eeg_data, args),
            to_alpha3(eeg_data, args),
            to_alpha4(eeg_data, args)
        ]
        return de_features

    for test_subject_idx in tqdm(range(num_subjects), desc="LOSO Folds"):
        save_file = os.path.join(
            save_path,
            f'fold_{test_subject_idx + 1}_processed_{args.dataset}_data_{args.subject_number}_subjects_{args.length}s.pkl'
        )
        if os.path.exists(save_file):
            print(f"Loading pre-processed data for fold {test_subject_idx + 1} from {save_file}")
            with open(save_file, 'rb') as f:
                fold_data = pickle.load(f)
            folds_data.append(fold_data)
            continue

        print(f"\nProcessing fold {test_subject_idx + 1} where subject {test_subject_idx + 1} is the test set...")

        window_size = args.window_length
        stride = int(window_size * (1 - args.overlap))

        # 初始化数据列表
        csp_train_eeg_list, raw_train_eeg_list = [], []
        csp_val_eeg_list, raw_val_eeg_list = [], []
        train_label_list, val_label_list = [], []
        # test_eeg_list, test_label_list = [], []

        train_subjects = [i for i in range(num_subjects) if i != test_subject_idx]

        # ========== 对训练 + 验证的受试者做处理 ========== #
        for subject_idx in tqdm(train_subjects, desc=f"[Fold {test_subject_idx + 1}] Processing training subjects", leave=False):
            subject_eeg_data = eeg_datas[subject_idx]
            subject_labels = labels[subject_idx]

            subject_windows = []
            subject_label_array = []

            # --- 滑窗 ---
            for trial_idx in range(len(subject_labels)):
                eeg = subject_eeg_data[trial_idx].to_numpy()  # dataframe -> numpy
                label = subject_labels[trial_idx]

                # 按指定滑窗 + 步长切片
                for i in range(0, eeg.shape[0] - window_size + 1, stride):
                    window = eeg[i:i + window_size, :]
                    subject_windows.append(window)
                    subject_label_array.append(label)

            # 转为 numpy 数组
            subject_eeg_arr = np.array(subject_windows).reshape(-1, window_size, out_channels)
            subject_label_arr = np.array(subject_label_array).reshape(-1, 1)

            # 打乱
            np.random.seed(args.seed)
            indices = np.arange(len(subject_label_arr))
            np.random.shuffle(indices)
            subject_eeg_arr = subject_eeg_arr[indices]
            subject_label_arr = subject_label_arr[indices]

            # 拆分训练集 (80%) 和验证集 (20%)
            split_idx = int(0.8 * len(subject_label_arr))
            subject_train_eeg = subject_eeg_arr[:split_idx]
            subject_train_label = subject_label_arr[:split_idx]
            subject_val_eeg = subject_eeg_arr[split_idx:]
            subject_val_label = subject_label_arr[split_idx:]

            # --- 如果需要使用 CSP, 则只在本受试者的训练部分上 fit ---
            if args.is_CSP:
                # 拟合 CSP
                csp = CSP(
                    n_components=args.csp_comp,
                    reg=None,
                    log=None,
                    cov_est='concat',
                    transform_into='csp_space',
                    norm_trace=True
                )
                subject_train_eeg_csp = subject_train_eeg.transpose(0, 2, 1)
                csp.fit(subject_train_eeg_csp, subject_train_label.squeeze())
                subject_train_eeg_transformed = csp.transform(subject_train_eeg_csp).transpose(0, 2, 1)
                subject_val_eeg_csp = subject_val_eeg.transpose(0, 2, 1)
                subject_val_eeg_transformed = csp.transform(subject_val_eeg_csp).transpose(0, 2, 1)
            else:
                subject_train_eeg_transformed = subject_train_eeg
                subject_val_eeg_transformed = subject_val_eeg

            # 保存转换后的数据和原始数据
            csp_train_eeg_list.append(subject_train_eeg_transformed)
            raw_train_eeg_list.append(subject_train_eeg)
            csp_val_eeg_list.append(subject_val_eeg_transformed)
            raw_val_eeg_list.append(subject_val_eeg)
            train_label_list.append(subject_train_label)
            val_label_list.append(subject_val_label)

        # 拼接数据
        csp_train_eeg = np.concatenate(csp_train_eeg_list, axis=0) if csp_train_eeg_list else np.array([])
        raw_train_eeg = np.concatenate(raw_train_eeg_list, axis=0) if raw_train_eeg_list else np.array([])
        train_labels = np.concatenate(train_label_list, axis=0) if train_label_list else np.array([])
        csp_val_eeg = np.concatenate(csp_val_eeg_list, axis=0) if csp_val_eeg_list else np.array([])
        raw_val_eeg = np.concatenate(raw_val_eeg_list, axis=0) if raw_val_eeg_list else np.array([])
        val_labels = np.concatenate(val_label_list, axis=0) if val_label_list else np.array([])

        # 处理测试集
        test_subject_eeg_data = eeg_datas[test_subject_idx]
        test_subject_labels = labels[test_subject_idx]
        test_eeg, test_labels = [], []

        for trial_idx in range(len(test_subject_labels)):
            eeg = test_subject_eeg_data[trial_idx].to_numpy()
            label = test_subject_labels[trial_idx]
            for i in range(0, eeg.shape[0] - window_size + 1, stride):
                test_eeg.append(eeg[i:i + window_size, :])
                test_labels.append(label)

        test_eeg = np.array(test_eeg).reshape(-1, window_size, out_channels)
        test_labels = np.array(test_labels).reshape(-1, 1)

        if args.is_CSP:
            test_eeg_csp = test_eeg.transpose(0, 2, 1)
            csp_test_eeg = csp.transform(test_eeg_csp).transpose(0, 2, 1)
        else:
            csp_test_eeg = test_eeg

        # 计算DE特征（使用原始数据）
        train_de_features = compute_de_features(raw_train_eeg, args) if raw_train_eeg.size else []
        val_de_features = compute_de_features(raw_val_eeg, args) if raw_val_eeg.size else []
        test_de_features = compute_de_features(test_eeg, args) if test_eeg.size else []

        # 转换为图像并处理维度
        if args.use_image:
            train_de_features = [gen_images(feat, args) for feat in train_de_features]
            val_de_features = [gen_images(feat, args) for feat in val_de_features] if val_de_features else []
            test_de_features = [gen_images(feat, args) for feat in test_de_features] if test_de_features else []

        de_train_eeg = np.stack(train_de_features, axis=1) if train_de_features else np.array([])
        de_val_eeg = np.stack(val_de_features, axis=1) if val_de_features else np.array([])
        de_test_eeg = np.stack(test_de_features, axis=1) if test_de_features else np.array([])

        if args.dbpnet:
            # 调整维度以适应DBPNet
            de_train_eeg = np.expand_dims(de_train_eeg, axis=-1).transpose(0, 4, 1, 2, 3)
            de_val_eeg = np.expand_dims(de_val_eeg, axis=-1).transpose(0, 4, 1, 2, 3) if de_val_eeg.size else np.array([])
            de_test_eeg = np.expand_dims(de_test_eeg, axis=-1).transpose(0, 4, 1, 2, 3) if de_test_eeg.size else np.array([])

        fold_data = {
            'csp_train_eeg': csp_train_eeg,
            'de_train_eeg': de_train_eeg,
            'train_label': train_labels,
            'csp_val_eeg': csp_val_eeg,
            'de_val_eeg': de_val_eeg,
            'val_label': val_labels,
            'csp_test_eeg': csp_test_eeg,
            'de_test_eeg': de_test_eeg,
            'test_label': test_labels
        }

        with open(save_file, 'wb') as f:
            pickle.dump(fold_data, f)
        print(f"Fold {test_subject_idx + 1} data processed and saved to {save_file}")

def sliding_window_LOSO_CSP_DE_all_subject(eeg_datas, labels, args, out_channels, save_path):
    """
    使用留一被试法（LOSO）对 EEG 数据进行滑动窗口处理，并统一拟合一个 CSP 模型。
    处理后的每个折叠的数据将被保存或返回。

    参数：
        eeg_datas (list): 所有受试者的 EEG 数据列表，每个元素对应一个受试者的数据。
        labels (list): 所有受试者的标签列表，每个元素对应一个受试者的标签。
        args (Namespace): 包含各种参数的命名空间（如窗口长度、重叠比例、随机种子等）。
        out_channels (int): 输出的通道数量，用于重塑数据形状。
        save_path (str): 保存处理后数据的根目录路径。

    返回：
        folds_data (list): 每个折叠的数据字典列表，包含
            'train_eeg', 'test_eeg',
            'train_label', 'test_label'。
        csp_models (list): 每个折叠的 CSP 模型。
    """

    # num_subjects = len(eeg_datas)
    num_subjects = len(eeg_datas)
    folds_data = []
    os.makedirs(save_path, exist_ok=True)


    for test_subject_idx in tqdm(range(num_subjects), desc="LOSO Folds"):
        # 定义当前折叠的保存文件路径
        save_file = os.path.join(
            save_path,
            f'fold_{test_subject_idx + 1}_processed_{args.dataset}_data_{args.subject_number}_subjects_{args.length}s.pkl'
        )

        # 如果已经存在当前折叠的处理数据，则直接加载
        if os.path.exists(save_file):
            print(f"Loading pre-processed data for fold {test_subject_idx + 1} from {save_file}")
            with open(save_file, 'rb') as f:
                fold_data = pickle.load(f)
            folds_data.append(fold_data)
            continue  # 跳过当前折叠的处理，继续下一个折叠

        print(f"\nProcessing fold {test_subject_idx + 1} where subject {test_subject_idx + 1} is the test set...")

        window_size = args.window_length
        stride = int(window_size * (1 - args.overlap))

        # 收集所有训练和测试数据
        train_eeg_list = []
        train_label_list = []
        test_eeg_list = []
        test_label_list = []

        # 定义训练集和测试集的受试者索引
        test_subject = test_subject_idx
        train_subjects = [i for i in range(num_subjects) if i != test_subject]

        # ========== 处理训练集受试者的数据 ==========
        for subject_idx in tqdm(train_subjects, desc=f"[Fold {test_subject_idx+1}] Processing training subjects",
                                leave=False):
            subject_eeg_data = eeg_datas[subject_idx]
            subject_labels = labels[subject_idx]

            subject_windows = []
            subject_label_array = []

            # --- 滑窗 ---
            for trial_idx in range(len(subject_labels)):
                eeg = subject_eeg_data[trial_idx].to_numpy()  # 转换为 NumPy 数组
                label = subject_labels[trial_idx]

                # 将每个 trial 按指定滑窗和步长切片
                for i in range(0, eeg.shape[0] - window_size + 1, stride):
                    window = eeg[i:i + window_size, :]
                    subject_windows.append(window)
                    subject_label_array.append(label)

            # 转换为 NumPy 数组
            subject_eeg_arr = np.array(subject_windows).reshape(-1, window_size, out_channels)
            subject_label_arr = np.array(subject_label_array).reshape(-1, 1)

            # --- 打乱 ---
            np.random.seed(args.seed)
            indices = np.arange(len(subject_label_arr))
            np.random.shuffle(indices)
            subject_eeg_arr = subject_eeg_arr[indices]
            subject_label_arr = subject_label_arr[indices]

            # --- 将该受试者的训练集加入全局列表 ---
            train_eeg_list.append(subject_eeg_arr)
            train_label_list.append(subject_label_arr)

        # 将所有训练集数据拼接
        train_eeg = np.concatenate(train_eeg_list, axis=0) if len(train_eeg_list) > 0 else np.array([])
        train_labels = np.concatenate(train_label_list, axis=0) if len(train_label_list) > 0 else np.array([])

        # ========== 拟合 CSP 模型 ==========
        # if args.is_CSP:
        # 拟合 CSP 在所有训练数据上
        csp = CSP(
            n_components=args.csp_comp,
            reg=None,
            log=None,
            cov_est='concat',
            transform_into='csp_space',
            norm_trace=True
        )
        # CSP 需要的数据格式为 (n_samples, n_channels, n_times)
        train_eeg_csp = train_eeg.transpose(0, 2, 1)
        csp.fit(train_eeg_csp, train_labels.squeeze())


        # 转换训练集
        csp_train_eeg = csp.transform(train_eeg_csp)
        # else:
        #     train_eeg_transformed = train_eeg

        # ========== 处理测试集受试者的数据 ==========
        test_subject_eeg_data = eeg_datas[test_subject]
        test_subject_labels = labels[test_subject]

        for trial_idx in range(len(test_subject_labels)):
            eeg = test_subject_eeg_data[trial_idx].to_numpy()
            label = test_subject_labels[trial_idx]

            for i in range(0, eeg.shape[0] - window_size + 1, stride):
                window = eeg[i:i + window_size, :]
                test_eeg_list.append(window)
                test_label_list.append(label)

        test_eeg = np.array(test_eeg_list).reshape(-1, window_size, out_channels)
        test_labels = np.array(test_label_list).reshape(-1, 1)

        # 转换测试集
        # if args.is_CSP:
        # 使用拟合好的 CSP 模型转换测试集
        test_eeg_csp = test_eeg.transpose(0, 2, 1)
        csp_test_eeg = csp.transform(test_eeg_csp)
        # else:
        #     test_eeg_transformed = test_eeg

        def compute_de_features(eeg_data, args):
            """
            计算 DE 特征。
            参数:
                eeg_data (numpy.ndarray): EEG 数据，形状为 (n_samples, window_size, n_channels)
                args (Namespace): 包含参数的命名空间
            返回:
                list: 各频段的 DE 特征列表
            """
            # 计算五个频段的 DE 特征
            de_features = [
                to_alpha0(eeg_data, args),
                to_alpha1(eeg_data, args),
                to_alpha2(eeg_data, args),
                to_alpha3(eeg_data, args),
                to_alpha4(eeg_data, args)
            ]
            return de_features

            # 计算训练集 DE 特征

        train_de_features = compute_de_features(train_eeg, args)
        # 计算测试集 DE 特征
        test_de_features = compute_de_features(test_eeg, args)

        # if args.use_image:
        # 将 DE 特征转换为图像
        train_de_features = [gen_images(feat, args) for feat in train_de_features]
        test_de_features = [gen_images(feat, args) for feat in test_de_features]

        # 堆叠五个频段的 DE 特征
        # 假设每个 de_feature 是一个 numpy 数组，形状为 (n_samples, feature_dim) 或图像
        # 如果是图像，堆叠时需要确保形状兼容
        # 这里假设堆叠在新的维度上
        # 例如，对于特征向量，堆叠后形状为 (n_samples, 5, feature_dim)
        # 对于图像，堆叠后形状为 (n_samples, 5, height, width, channels)

        # 使用 np.stack 堆叠五个频段的特征
        train_de_eeg = np.stack(train_de_features, axis=1)  # 新维度为频段
        test_de_eeg = np.stack(test_de_features, axis=1)

        if args.dbpnet == True:
            train_de_eeg = np.expand_dims(train_de_eeg, axis=-1)  # (N, 5, 32 , 32 , 1)
            test_de_eeg = np.expand_dims(test_de_eeg, axis=-1)
            de_train_eeg = train_de_eeg.transpose(0, 4, 1, 2, 3) # (N,1, 5, 32 , 32 )
            de_test_eeg = test_de_eeg.transpose(0, 4, 1, 2, 3)   # (N,1, 5, 32 , 32 )
        else:
            de_train_eeg = train_de_eeg
            de_test_eeg = test_de_eeg

        # 构建当前折叠的数据字典
        fold_data = {
            'csp_train_eeg': csp_train_eeg,
            'de_train_eeg':de_train_eeg,
            'train_label': train_labels,
            'csp_test_eeg': csp_test_eeg,
            'de_test_eeg':de_test_eeg,
            'test_label': test_labels,
        }

        # 保存当前折叠的数据
        with open(save_file, 'wb') as f:
            pickle.dump(fold_data, f)
        print(f"Fold {test_subject_idx + 1} data processed and saved to {save_file}")


def sliding_window_cv(eeg_datas, labels, args, out_channels):
    window_size = args.window_length
    stride = int(window_size * (1 - args.overlap))

    train_eeg = []
    test_eeg = []
    train_label = []
    test_label = []

    # eeg_datas:(8,80176,64)
    for m in range(len(labels)):
        eeg = eeg_datas[m]
        label = labels[m]
        windows = []
        new_label = []
        for i in range(0, eeg.shape[0] - window_size + 1, stride): # 计算窗口数量
            window = eeg[i:i+window_size, :]
            windows.append(window)
            new_label.append(label)
        train_eeg.append(np.array(windows)[:int(len(windows)*0.9)])
        train_label.append(np.array(new_label)[:int(len(windows)*0.9)])

    train_eeg = np.stack(train_eeg, axis=0).reshape(-1, window_size, out_channels)
    train_label = np.stack(train_label, axis=0).reshape(-1, 1)

    return train_eeg, train_label

def to_alpha0(data, args):
    alpha_data = []
    for window in data:
        window_data0 = np.fft.fft(window, n=args.window_length, axis=0)
        window_data0 = np.abs(window_data0)
        window_data0 = np.sum(np.power(window_data0[args.point0_low:args.point0_high, :], 2), axis=0)
        window_data0 = np.log2(window_data0 / args.window_length)
        alpha_data.append(window_data0)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data

def to_alpha1(data, args):
    alpha_data = []
    for window in data:
        window_data1 = np.fft.fft(window, n=args.window_length, axis=0)
        window_data1 = np.abs(window_data1)
        window_data1 = np.sum(np.power(window_data1[args.point1_low:args.point1_high, :], 2), axis=0)
        window_data1 = np.log2(window_data1 / args.window_length)
        alpha_data.append(window_data1)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data

def to_alpha2(data, args):
    alpha_data = []
    for window in data:
        window_data2 = np.fft.fft(window, n=args.window_length, axis=0)
        window_data2 = np.abs(window_data2)
        window_data2 = np.sum(np.power(window_data2[args.point2_low:args.point2_high, :], 2), axis=0)
        window_data2 = np.log2(window_data2 / args.window_length)
        alpha_data.append(window_data2)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data

def to_alpha3(data, args):
    alpha_data = []
    for window in data:
        window_data3 = np.fft.fft(window, n=args.window_length, axis=0)
        window_data3 = np.abs(window_data3)
        window_data3 = np.sum(np.power(window_data3[args.point3_low:args.point3_high, :], 2), axis=0)
        window_data3 = np.log2(window_data3 / args.window_length)
        alpha_data.append(window_data3)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data

def to_alpha4(data, args):
    alpha_data = []
    for window in data:
        window_data4= np.fft.fft(window, n=args.window_length, axis=0)
        window_data4 = np.abs(window_data4)
        window_data4 = np.sum(np.power(window_data4[args.point4_low:args.point4_high, :], 2), axis=0)
        window_data4 = np.log2(window_data4 / args.window_length)
        alpha_data.append(window_data4)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data


def to_alpha_one_band(data, args):
    alpha_data = []
    for window in data:
        window_data0 = np.fft.fft(window, n=args.window_length, axis=0)
        window_data0 = np.abs(window_data0)
        window_data0 = np.sum(np.power(window_data0[args.point0_low:args.point4_high, :], 2), axis=0)
        window_data0 = np.log2(window_data0 / args.window_length)
        alpha_data.append(window_data0)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data
