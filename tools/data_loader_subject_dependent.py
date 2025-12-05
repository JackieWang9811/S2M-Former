import sys
sys.path.append('/data/wjq/AAD/OpenAAD/')
from dotmap import DotMap
from .function import *
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from datetime import datetime
np.set_printoptions(suppress=True)


# Set SEEDS
def set_random_seeds(seed):
    ''' Set random seeds for reproducibility. '''
    seed_val = seed
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Single Branch
class CustomDatasets(Dataset):
    # initialization: data and label
    def __init__(self, seq_data, event_data):
        self.data = seq_data
        self.label = event_data

    # get the size of data
    def __len__(self):
        return len(self.label)

    # get the data and label
    def __getitem__(self, index):
        data = torch.Tensor(self.data[index])
        label = torch.Tensor(self.label[index])

        return data, label

# Dual Branch
class DualCustomDatasets(Dataset):
    # initialization: data and label
    def __init__(self, seq_data, fre_data, event_data):
        self.seq_data = seq_data
        self.fre_data = fre_data
        self.label = event_data

    # get the size of data
    def __len__(self):
        return len(self.label)

    # get the data and label
    def __getitem__(self, index):
        seq_data = torch.Tensor(self.seq_data[index])
        fre_data = torch.Tensor(self.fre_data[index])
        # label = torch.LongTensor(self.label[index])
        label = torch.Tensor(self.label[index])

        return seq_data, fre_data, label


class TripleCustomDatasets(Dataset):
    # initialization: data and label
    def __init__(self, seq_data, fre_data, eeg_data, event_data):
        self.seq_data = seq_data
        self.fre_data = fre_data
        self.eeg_data = eeg_data
        self.label = event_data

    # get the size of data
    def __len__(self):
        return len(self.label)

    # get the data and label
    def __getitem__(self, index):
        seq_data = torch.Tensor(self.seq_data[index])
        fre_data = torch.Tensor(self.fre_data[index])
        eeg_data = torch.Tensor(self.eeg_data[index])
        # label = torch.LongTensor(self.label[index])
        label = torch.Tensor(self.label[index])

        return seq_data, fre_data, eeg_data, label

# Add these lines just after splitting the data in the `main` function
def count_labels(labels, label_name, logger=None):
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique, counts))
    print(f"Counts in {label_name}: {label_counts}")
    logger.info(f"Counts in {label_name}: {label_counts}")

def log_selected_args(args, logger):
    selected_keys = ['lr', 'weight_decay', 'T_max']
    for key in selected_keys:
        if key in args.keys():
            logger.info(f'{key}: {args[key]}')


# No need to modify, put this into the main.py
def getData(name="S1", data_document_path="/data/wjq/AAD/DTUDataset", is_cross_trials= False, ConType="No_vanilla", length = 1, seed= 312, branch = 1, is_CSP = True, is_DE= False, use_image = True, is_3D = False, use_multi_band=True, logger=None, args=None):

    return get_datasets_data(name, data_document_path=data_document_path, is_cross_trials=is_cross_trials, length = length, seed= seed, branch = branch, is_CSP = is_CSP, is_DE= is_DE, use_image=use_image, is_3D = is_3D, use_multi_band= use_multi_band, logger_path=logger, args=args)


def get_datasets_data(name="S1", data_document_path="/data/wjq/AAD/DTUDataset", is_cross_trials=False, length = 1, seed= 312, branch = 1, is_CSP = True, is_DE= False, is_3D = False, use_multi_band= True, use_image= True, logger_path=None, args=None):

    # set_random_seeds(seed)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.current_time = current_time
    if args is None:
        args = DotMap()

    args.model_name = args.get("model_name")

    if data_document_path == "/data/wjq/AAD/DTUDataset":
        args.trail_number = 60
        args.dataset = "DTUDataset"
    elif data_document_path == "/data/wjq/AAD/KULDataset":
        args.trail_number = 8 # 8: KUL 60:DTU 20 trials = 4 + 4 + 12  - 第一次实验 4 次，第二次实验 4 次，第三次实验 12 次（实验 1 的 4 次演讲的前 2 分钟，重复 3 次）
        args.dataset = "KULDataset"

    elif data_document_path == "/data/wjq/AAD/AVGCDataset_no_visuals":
        args.trail_number = 2
        args.dataset = "AVGCDataset_no_visuals"

    elif data_document_path == "/data/wjq/AAD/AVGCDataset_fixed_video":
        args.trail_number = 2
        args.dataset = "AVGCDataset_fixed_video"

    elif data_document_path == "/data/wjq/AAD/AVGCDataset_moving_video":
        args.trail_number = 2
        args.dataset = "AVGCDataset_fixed_video"

    elif data_document_path == "/data/wjq/AAD/AVGCDataset_moving_target_noise":
        args.trail_number = 2
        args.dataset = "AVGCDataset_fixed_video"

    elif data_document_path == "/data/wjq/AAD/AVGCDataset_across_conditions":
        args.dataset = "AVGCDataset_across_conditions"
        if args.name == "S1" or args.name == "S3":
            args.trail_number = 6
        elif args.name == "S14":
            args.trail_number = 7
        else:
            args.trail_number = 8

    elif data_document_path == "/data/wjq/AAD/AVGCDataset_across_conditions_zscore":
        args.dataset = "AVGCDataset_across_conditions_zscore"
        if args.name == "S1" or args.name == "S3":
            args.trail_number = 6
        elif args.name == "S14":
            args.trail_number = 7
        else:
            args.trail_number = 8


    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.label_col = 0

    """
    #########################
    ##  Training Parameter ##
    #########################
    """

    args.seed = seed

    dataset_name = args.dataset

    if logger_path == None:
        if is_CSP == True:
            args.log_path = f"./result/{dataset_name}/{args.ConType}/{args.model_name}/{length}s/seed_{seed}_csp_comp_{args.csp_comp}_{current_time}"
        else:
            args.log_path = f"./result/{dataset_name}/{args.ConType}/{args.model_name}/{length}s/seed_{seed}_{current_time}"
        # 新增SNN模型判断逻辑
        if hasattr(args, 'snn_model') and isinstance(args.snn_model, str):
            args.log_path += f"_{args.snn_model}"  # 追加模型名称
        logger = get_logger(args.name, args.log_path, length)
        log_selected_args(args, logger)
    else:
        logger = get_logger(args.name, logger_path, length)
        log_selected_args(args, logger)

    print(args.ConType)


    eeg_data, event_data = read_prepared_data(args)
    data = np.vstack(eeg_data)
    eeg_data = data.reshape([args.trail_number, -1, args.eeg_channel]) # (8,46080,64)  (8,25200,64)
    event_data = np.vstack(event_data)
    if args.dataset == "DTUDataset":
        if eeg_data.shape[1] == 6400:
            args.fs = 128
            logger.info(f"{args.fs}")
            print(f"{args.fs}")
            args.window_length = math.floor(args.fs * length) # eg: 1s : window_length = 128*1;  2s: window_length = 128*2 = 256
            args.delta_low = 1
            args.delta_high = 3
            args.theta_low = 4
            args.theta_high = 7
            args.alpha_low = 8
            args.alpha_high = 13
            args.beta_low = 14
            args.beta_high = 30
            args.gamma_low = 31
            args.gamma_high = 50
            args.frequency_resolution = args.fs / args.window_length
            args.point0_low = math.ceil(args.delta_low / args.frequency_resolution)
            args.point0_high = math.ceil(args.delta_high / args.frequency_resolution) + 1
            args.point1_low = math.ceil(args.theta_low / args.frequency_resolution)
            args.point1_high = math.ceil(args.theta_high / args.frequency_resolution) + 1
            args.point2_low = math.ceil(args.alpha_low / args.frequency_resolution)
            args.point2_high = math.ceil(args.alpha_high / args.frequency_resolution) + 1
            args.point3_low = math.ceil(args.beta_low / args.frequency_resolution)
            args.point3_high = math.ceil(args.beta_high / args.frequency_resolution) + 1
            args.point4_low = math.ceil(args.gamma_low / args.frequency_resolution)
            args.point4_high = math.ceil(args.gamma_high / args.frequency_resolution) + 1
        else:
            args.fs = 64
            logger.info(f"{args.fs}")
            print(f"{args.fs}")
            args.window_length = math.floor(args.fs * length) # eg: 0.1s window_length = 64*0.1 = 7(向上取整) 1s : window_length = 64*1;  2s: window_length = 64*2 = 128,
            args.delta_low = 1
            args.delta_high = 3
            args.theta_low = 4
            args.theta_high = 7
            args.alpha_low = 8
            args.alpha_high = 13
            args.beta_low = 14
            args.beta_high = 30
            args.gamma_low = 31
            args.gamma_high = 32
            args.frequency_resolution = args.fs / args.window_length
            args.point0_low = math.ceil(args.delta_low / args.frequency_resolution)
            args.point0_high = math.ceil(args.delta_high / args.frequency_resolution) + 1
            args.point1_low = math.ceil(args.theta_low / args.frequency_resolution)
            args.point1_high = math.ceil(args.theta_high / args.frequency_resolution) + 1
            args.point2_low = math.ceil(args.alpha_low / args.frequency_resolution)
            args.point2_high = math.ceil(args.alpha_high / args.frequency_resolution) + 1
            args.point3_low = math.ceil(args.beta_low / args.frequency_resolution)
            args.point3_high = math.ceil(args.beta_high / args.frequency_resolution) + 1
            args.point4_low = math.ceil(args.gamma_low / args.frequency_resolution)
            args.point4_high = math.ceil(args.gamma_high / args.frequency_resolution) + 1

    elif args.dataset == "KULDataset":
        if eeg_data.shape[1] >= 46080:
            args.fs = 128
            logger.info(f"{args.fs}")
            print(f"{args.fs}")
            args.window_length = math.floor(args.fs * length) # eg: 1s : window_length = 128*1;  2s: window_length = 128*2 = 256
            args.delta_low = 1
            args.delta_high = 3
            args.theta_low = 4
            args.theta_high = 7
            args.alpha_low = 8
            args.alpha_high = 13
            args.beta_low = 14
            args.beta_high = 30
            args.gamma_low = 31
            args.gamma_high = 50
            args.frequency_resolution = args.fs / args.window_length
            args.point0_low = math.ceil(args.delta_low / args.frequency_resolution)
            args.point0_high = math.ceil(args.delta_high / args.frequency_resolution) + 1
            args.point1_low = math.ceil(args.theta_low / args.frequency_resolution)
            args.point1_high = math.ceil(args.theta_high / args.frequency_resolution) + 1
            args.point2_low = math.ceil(args.alpha_low / args.frequency_resolution)
            args.point2_high = math.ceil(args.alpha_high / args.frequency_resolution) + 1
            args.point3_low = math.ceil(args.beta_low / args.frequency_resolution)
            args.point3_high = math.ceil(args.beta_high / args.frequency_resolution) + 1
            args.point4_low = math.ceil(args.gamma_low / args.frequency_resolution)
            args.point4_high = math.ceil(args.gamma_high / args.frequency_resolution) + 1

        else:
            args.fs = 64
            logger.info(f"{args.fs}")
            print(f"{args.fs}")
            args.window_length = math.floor(args.fs * length) # eg: 0.1s window_length = 64*0.1 = 7(向上取整) 1s : window_length = 64*1;  2s: window_length = 64*2 = 128,
            args.delta_low = 1
            args.delta_high = 3
            args.theta_low = 4
            args.theta_high = 7
            args.alpha_low = 8
            args.alpha_high = 13
            args.beta_low = 14
            args.beta_high = 30
            args.gamma_low = 31
            args.gamma_high = 32
            args.frequency_resolution = args.fs / args.window_length
            args.point0_low = math.ceil(args.delta_low / args.frequency_resolution)
            args.point0_high = math.ceil(args.delta_high / args.frequency_resolution) + 1
            args.point1_low = math.ceil(args.theta_low / args.frequency_resolution)
            args.point1_high = math.ceil(args.theta_high / args.frequency_resolution) + 1
            args.point2_low = math.ceil(args.alpha_low / args.frequency_resolution)
            args.point2_high = math.ceil(args.alpha_high / args.frequency_resolution) + 1
            args.point3_low = math.ceil(args.beta_low / args.frequency_resolution)
            args.point3_high = math.ceil(args.beta_high / args.frequency_resolution) + 1
            args.point4_low = math.ceil(args.gamma_low / args.frequency_resolution)
            args.point4_high = math.ceil(args.gamma_high / args.frequency_resolution) + 1

    elif "AVGCDataset" in args.dataset:
        args.fs = 128
        logger.info(f"{args.fs}")
        print(f"{args.fs}")
        args.window_length = math.floor(args.fs * length) # eg: 1s : window_length = 128*1;  2s: window_length = 128*2 = 256
        args.delta_low = 1
        args.delta_high = 3
        args.theta_low = 4
        args.theta_high = 7
        args.alpha_low = 8
        args.alpha_high = 13
        args.beta_low = 14
        args.beta_high = 30
        args.gamma_low = 31
        args.gamma_high = 40
        args.frequency_resolution = args.fs / args.window_length
        args.point0_low = math.ceil(args.delta_low / args.frequency_resolution)
        args.point0_high = math.ceil(args.delta_high / args.frequency_resolution) + 1
        args.point1_low = math.ceil(args.theta_low / args.frequency_resolution)
        args.point1_high = math.ceil(args.theta_high / args.frequency_resolution) + 1
        args.point2_low = math.ceil(args.alpha_low / args.frequency_resolution)
        args.point2_high = math.ceil(args.alpha_high / args.frequency_resolution) + 1
        args.point3_low = math.ceil(args.beta_low / args.frequency_resolution)
        args.point3_high = math.ceil(args.beta_high / args.frequency_resolution) + 1
        args.point4_low = math.ceil(args.gamma_low / args.frequency_resolution)
        args.point4_high = math.ceil(args.gamma_high / args.frequency_resolution) + 1



    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)

    label = np.array(event_data).squeeze()

    # Choose between 5-fold cross-validation and 8:1:1 split
    if args.is_cross_validation:
        # Perform 5-fold cross-validation
        fold_data = sliding_window_K_fold(eeg_data, label, args, args.eeg_channel, n_folds=5)
        fold_loaders = []

        for fold_index, (train_eeg, test_eeg, train_label, test_label) in enumerate(fold_data):
            # Preprocessing for each fold
            train_loader, valid_loader, test_loader = prepare_data_loaders(train_eeg, train_label, test_eeg, test_label, args, branch, is_CSP, is_DE, use_multi_band, use_image, seed, logger)
            fold_loaders.append((train_loader, valid_loader, test_loader))

        return args, logger, fold_loaders  # Return loaders for each fold

    elif is_cross_trials:
        # Perform 8:1:1 split, 最后的10%永远作为测试集
        # train_eeg, test_eeg, train_label, test_label = sliding_window_cross_trials(name, eeg_data, label, args, args.eeg_channel)
        # train_loader, valid_loader, test_loader = prepare_data_loaders(train_eeg, train_label, test_eeg, test_label, args, branch, is_CSP, is_DE, is_3D, use_multi_band, use_image, seed, logger)

        # 随机10%
        train_eeg, test_eeg, train_label, test_label = sliding_window_cross_trials_random(name, eeg_data, label, args, args.eeg_channel)
        train_loader, valid_loader, test_loader = prepare_data_loaders(train_eeg, train_label, test_eeg, test_label, args, branch, is_CSP, is_DE, is_3D, use_multi_band, use_image, seed, logger)

        return args, logger, train_loader, valid_loader, test_loader  # Return single split loaders

    else:
        # Perform 8:1:1 split, 最后的10%永远作为测试集
        train_eeg, test_eeg, train_label, test_label = sliding_window(eeg_data, label, args, args.eeg_channel)
        train_loader, valid_loader, test_loader = prepare_data_loaders(train_eeg, train_label, test_eeg, test_label, args, branch, is_CSP, is_DE, is_3D, use_multi_band, use_image, seed, logger)

        # train_eeg, valid_eeg, test_eeg, train_label, valid_label, test_label = sliding_window_random_v2(eeg_data, label, args, args.eeg_channel)
        # train_loader, valid_loader, test_loader = prepare_data_loaders_random(train_eeg, train_label, valid_eeg, valid_label, test_eeg, test_label, args, branch, is_CSP, is_DE, is_3D, use_multi_band, use_image, seed, logger)

        return args, logger, train_loader, valid_loader, test_loader  # Return single split loaders



def prepare_data_loaders(train_eeg, train_label, test_eeg, test_label, args, branch, is_CSP, is_DE, is_3D, use_multi_band, use_image, seed, logger):

    if branch ==1:

        ######### USE CSP Feature ##########
        if is_CSP ==True:

            indices = np.arange(len(train_label))
            np.random.seed(seed)
            np.random.shuffle(indices)

            train_eeg = train_eeg[indices]
            train_label = train_label[indices]

        ######### USE DE Feature ##########
        elif is_DE ==True:

            ######### MULTI BANDS ##########
            if use_multi_band == True:

                # fft=> DE feature (five frequency)
                train_data0 = to_alpha0(train_eeg, args)
                test_data0 = to_alpha0(test_eeg, args)
                train_data1 = to_alpha1(train_eeg, args)
                test_data1 = to_alpha1(test_eeg, args)
                train_data2 = to_alpha2(train_eeg, args)
                test_data2 = to_alpha2(test_eeg, args)
                train_data3 = to_alpha3(train_eeg, args)
                test_data3 = to_alpha3(test_eeg, args)
                train_data4 = to_alpha4(train_eeg, args)
                test_data4 = to_alpha4(test_eeg, args)

                if use_image:

                    train_data0 = gen_images(train_data0, args)
                    test_data0 = gen_images(test_data0, args)
                    train_data1 = gen_images(train_data1, args)
                    test_data1 = gen_images(test_data1, args)
                    train_data2 = gen_images(train_data2, args)
                    test_data2 = gen_images(test_data2, args)
                    train_data3 = gen_images(train_data3, args)
                    test_data3 = gen_images(test_data3, args)
                    train_data4 = gen_images(train_data4, args)
                    test_data4 = gen_images(test_data4, args)

                    input_train_data = np.stack([train_data0, train_data1, train_data2, train_data3, train_data4],
                                                axis=1)
                    test_data = np.stack([test_data0, test_data1, test_data2, test_data3, test_data4], axis=1)

                    input_train_data = np.expand_dims(input_train_data, axis=-1)
                    test_data = np.expand_dims(test_data, axis=-1)

                else:
                    input_train_data = np.stack([train_data0, train_data1, train_data2, train_data3, train_data4],
                                                axis=1)  # (N, 5, 64)
                    test_data = np.stack([test_data0, test_data1, test_data2, test_data3, test_data4],
                                         axis=1)  # # (N, 5, 64)

                train_eeg = input_train_data
                test_eeg = test_data

            ######### SINGLE BAND ##########
            else:
                # FFT
                train_eeg = to_alpha_one_band(train_eeg, args)
                test_eeg = to_alpha_one_band(test_eeg, args)

                train_eeg = gen_images(train_eeg, args)
                test_eeg = gen_images(test_eeg, args)

                train_eeg = train_eeg.reshape(train_eeg.shape[0], 1, 32, 32)
                test_eeg = test_eeg.reshape(test_eeg.shape[0], 1, 32, 32)

            indices = np.arange(len(train_label))
            np.random.seed(seed)
            np.random.shuffle(indices)

            train_eeg = train_eeg[indices]
            train_label = train_label[indices]


        ######### USE RAW EEG ##########
        else:

            train_eeg = train_eeg.transpose(0, 2, 1)
            test_eeg = test_eeg.transpose(0, 2, 1)


            indices = np.arange(len(train_label))
            np.random.seed(seed)
            np.random.shuffle(indices)

            train_eeg = train_eeg[indices]
            train_label = train_label[indices]


        train_data, valid_data, train_label, valid_label = train_test_split(train_eeg, train_label, test_size=0.1,  random_state=seed)

        if is_CSP:

            csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space', norm_trace=True)

            train_data = train_data.transpose(0, 2, 1)
            train_data = np.nan_to_num(train_data)  # 替换所有 NaN 和 Inf
            train_data = csp.fit_transform(train_data, train_label.squeeze())  #Fit and transform the training set

            # Only convert the valid set data without fitting CSP
            valid_data = valid_data.transpose(0, 2, 1)
            valid_data = np.nan_to_num(valid_data)  # 替换所有 NaN 和 Inf
            valid_data = csp.transform(valid_data)  # Only perform feature transformation without fitting

            # Only convert the test set data without fitting CSP
            test_eeg = test_eeg.transpose(0, 2, 1)
            test_eeg = np.nan_to_num(test_eeg)  # 替换所有 NaN 和 Inf
            test_eeg = csp.transform(test_eeg)  # Only perform feature transformation without fitting


        test_data = test_eeg

        train_loader = DataLoader(dataset=CustomDatasets(train_data, train_label),
                                      batch_size= args.batch_size, drop_last=False, num_workers=8)
        valid_loader = DataLoader(dataset=CustomDatasets(valid_data, valid_label),
                                      batch_size= args.batch_size, drop_last=False, num_workers=8)
        test_loader = DataLoader(dataset=CustomDatasets(test_data, test_label),
                                     batch_size= args.batch_size, drop_last=False, num_workers=8)


    elif branch == 2:

        ######### fft=> DE feature (five frequency) #######
        train_data0 = to_alpha0(train_eeg, args)  # (5632, 32, 32)
        test_data0 = to_alpha0(test_eeg, args)
        train_data1 = to_alpha1(train_eeg, args)
        test_data1 = to_alpha1(test_eeg, args)
        train_data2 = to_alpha2(train_eeg, args)
        test_data2 = to_alpha2(test_eeg, args)
        train_data3 = to_alpha3(train_eeg, args)
        test_data3 = to_alpha3(test_eeg, args)
        train_data4 = to_alpha4(train_eeg, args)
        test_data4 = to_alpha4(test_eeg, args)

        if use_image:

            train_data0 = gen_images(train_data0, args)  # (5632, 32, 32)
            test_data0 = gen_images(test_data0, args)
            train_data1 = gen_images(train_data1, args)
            test_data1 = gen_images(test_data1, args)
            train_data2 = gen_images(train_data2, args)
            test_data2 = gen_images(test_data2, args)
            train_data3 = gen_images(train_data3, args)
            test_data3 = gen_images(test_data3, args)
            train_data4 = gen_images(train_data4, args)
            test_data4 = gen_images(test_data4, args)

            input_train_data = np.stack([train_data0, train_data1, train_data2, train_data3, train_data4],
                                        axis=1)  # (N, 5, 32, 32)
            test_data = np.stack([test_data0, test_data1, test_data2, test_data3, test_data4], axis=1)

            if is_3D:
                fre_train_data = np.expand_dims(input_train_data, axis=-1)  # (N, 5, 32 , 32 , 1)
                fre_test_data = np.expand_dims(test_data, axis=-1)
            else:
                fre_train_data = input_train_data
                fre_test_data  = test_data

        else:
            input_train_data = np.stack([train_data0, train_data1, train_data2, train_data3, train_data4],
                                        axis=1)  # (N, 5, 64)
            test_data = np.stack([test_data0, test_data1, test_data2, test_data3, test_data4], axis=1) # # (N, 5, 64)


            fre_train_data = input_train_data
            fre_test_data  = test_data

        ############ CSP #############


        seq_train_data = train_eeg
        seq_test_data = test_eeg
        # del data


        indices = np.arange(len(train_label))
        np.random.seed(seed)
        np.random.shuffle(indices)

        fre_train_data = fre_train_data[indices]
        seq_train_data = seq_train_data[indices]
        train_label = train_label[indices]


        seq_train_data, seq_valid_data, fre_train_data, fre_valid_data, train_label, valid_label = train_test_split(
            seq_train_data, fre_train_data, train_label, test_size=0.1, random_state=seed)


        csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space',
                  norm_trace=True)

        seq_train_data = seq_train_data.transpose(0, 2, 1)
        seq_train_data = csp.fit_transform(seq_train_data, train_label.squeeze())  # 对训练集进行拟合和转换

        seq_valid_data = seq_valid_data.transpose(0, 2, 1)
        seq_valid_data = csp.transform(seq_valid_data)  # 只进行特征转换，不进行拟合


        seq_test_data = seq_test_data.transpose(0, 2, 1)
        seq_test_data = csp.transform(seq_test_data)  # 只进行特征转换，不进行拟合

        if is_3D:
            fre_train_data = fre_train_data.transpose(0, 4, 1, 2, 3)
            fre_valid_data = fre_valid_data.transpose(0, 4, 1, 2, 3)
            fre_test_data = fre_test_data.transpose(0, 4, 1, 2, 3)

        train_loader = DataLoader(dataset=DualCustomDatasets(seq_train_data, fre_train_data, train_label),
                                  batch_size=args.batch_size, drop_last=False, num_workers=8)
        valid_loader = DataLoader(dataset=DualCustomDatasets(seq_valid_data, fre_valid_data, valid_label),
                                  batch_size=args.batch_size, drop_last=False, num_workers=8)
        test_loader = DataLoader(dataset=DualCustomDatasets(seq_test_data, fre_test_data, test_label),
                                 batch_size=args.batch_size, drop_last=False, num_workers=8)


    args.n_train = np.size(train_label)
    args.n_valid = np.size(valid_label)
    args.n_test = np.size(test_label)

    if branch == 1 :
        args.data_shape = train_loader.dataset.data.shape

    # Count occurrences in each label dataset
    count_labels(train_label, "train_label", logger)
    count_labels(valid_label, "valid_label", logger)
    count_labels(test_label, "test_label", logger)

    return train_loader, valid_loader, test_loader



def prepare_data_loaders_random(train_eeg, train_label, valid_eeg, valid_label, test_eeg, test_label, args, branch, is_CSP, is_DE, is_3D, use_multi_band, use_image, seed, logger):

    if branch ==1:

        ######### USE CSP Feature ##########
        if is_CSP ==True:

            indices = np.arange(len(train_label))
            np.random.seed(seed)
            np.random.shuffle(indices)

            train_eeg = train_eeg[indices]
            train_label = train_label[indices]

        ######### USE DE Feature ##########
        elif is_DE ==True:

            ######### MULTI BANDS ##########
            if use_multi_band == True:

                # fft=> DE feature (five frequency)
                train_data0 = to_alpha0(train_eeg, args)
                valid_data0 = to_alpha0(valid_eeg, args)
                test_data0 = to_alpha0(test_eeg, args)

                train_data1 = to_alpha1(train_eeg, args)
                valid_data1 = to_alpha1(valid_eeg, args)
                test_data1 = to_alpha1(test_eeg, args)

                train_data2 = to_alpha2(train_eeg, args)
                valid_data2 = to_alpha2(valid_eeg, args)
                test_data2 = to_alpha2(test_eeg, args)

                train_data3 = to_alpha3(train_eeg, args)
                valid_data3 = to_alpha3(valid_eeg, args)
                test_data3 = to_alpha3(test_eeg, args)

                train_data4 = to_alpha4(train_eeg, args)
                valid_data4 = to_alpha4(valid_eeg, args)
                test_data4 = to_alpha4(test_eeg, args)

                if use_image:

                    train_data0 = gen_images(train_data0, args)
                    valid_data0 = gen_images(valid_data0, args)
                    test_data0 = gen_images(test_data0, args)

                    train_data1 = gen_images(train_data1, args)
                    valid_data1 = gen_images(valid_data1, args)
                    test_data1 = gen_images(test_data1, args)

                    train_data2 = gen_images(train_data2, args)
                    valid_data2 = gen_images(valid_data2, args)
                    test_data2 = gen_images(test_data2, args)

                    train_data3 = gen_images(train_data3, args)
                    valid_data3 = gen_images(valid_data3, args)
                    test_data3 = gen_images(test_data3, args)

                    train_data4 = gen_images(train_data4, args)
                    valid_data4 = gen_images(valid_data4, args)
                    test_data4 = gen_images(test_data4, args)

                    input_train_data = np.stack([train_data0, train_data1, train_data2, train_data3, train_data4], axis=1)
                    valid_data = np.stack([valid_data0, valid_data1, valid_data2, valid_data3, valid_data4], axis=1)
                    test_data = np.stack([test_data0, test_data1, test_data2, test_data3, test_data4], axis=1)



                else:
                    input_train_data = np.stack([train_data0, train_data1, train_data2, train_data3, train_data4], axis=1)  # (N, 5, 64)
                    valid_data = np.stack([valid_data0, valid_data1, valid_data2, valid_data3, valid_data4], axis=1)
                    test_data = np.stack([test_data0, test_data1, test_data2, test_data3, test_data4],
                                         axis=1)  # # (N, 5, 64)

                train_eeg = input_train_data
                valid_eeg = valid_data
                test_eeg = test_data

            ######### SINGLE BAND ##########
            else:

                # FFT
                train_eeg = to_alpha0(train_eeg, args)
                test_eeg = to_alpha0(test_eeg, args)

                train_eeg = gen_images(train_eeg, args)
                test_eeg = gen_images(test_eeg, args)



            indices = np.arange(len(train_label))
            np.random.seed(seed)
            np.random.shuffle(indices)

            train_eeg = train_eeg[indices]
            train_label = train_label[indices]


        ######### USE RAW EEG ##########
        else:

            train_eeg = train_eeg.transpose(0, 2, 1)
            valid_eeg = valid_eeg.transpose(0, 2, 1)
            test_eeg = test_eeg.transpose(0, 2, 1)


            indices = np.arange(len(train_label))
            np.random.seed(seed)
            np.random.shuffle(indices)

            train_eeg = train_eeg[indices]
            train_label = train_label[indices]


        # train_data, valid_data, train_label, valid_label = train_test_split(train_eeg, train_label, test_size=0.1,  random_state=seed)
        train_data = train_eeg
        valid_data = valid_eeg

        if is_CSP:

            csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space', norm_trace=True)

            train_data = train_data.transpose(0, 2, 1)
            train_data = np.nan_to_num(train_data)  # 替换所有 NaN 和 Inf
            train_data = csp.fit_transform(train_data, train_label.squeeze())  #Fit and transform the training set

            # Only convert the valid set data without fitting CSP
            valid_data = valid_data.transpose(0, 2, 1)
            valid_data = np.nan_to_num(valid_data)  # 替换所有 NaN 和 Inf
            valid_data = csp.transform(valid_data)  # Only perform feature transformation without fitting

            # Only convert the test set data without fitting CSP
            test_eeg = test_eeg.transpose(0, 2, 1)
            test_eeg = np.nan_to_num(test_eeg)  # 替换所有 NaN 和 Inf
            test_eeg = csp.transform(test_eeg)  # Only perform feature transformation without fitting

        # elif is_DE:
        #     train_data = train_data.transpose(0, 4, 1, 2, 3)
        #     valid_data = valid_data.transpose(0, 4, 1, 2, 3)
        #     test_eeg = test_eeg.transpose(0, 4, 1, 2, 3)

        test_data = test_eeg

        train_loader = DataLoader(dataset=CustomDatasets(train_data, train_label),
                                      batch_size= args.batch_size, drop_last=False, num_workers=8)
        valid_loader = DataLoader(dataset=CustomDatasets(valid_data, valid_label),
                                      batch_size= args.batch_size, drop_last=False, num_workers=8)
        test_loader = DataLoader(dataset=CustomDatasets(test_data, test_label),
                                     batch_size= args.batch_size, drop_last=False, num_workers=8)


    elif branch == 2:

        ######### fft=> DE feature (five frequency) #######
        train_data0 = to_alpha0(train_eeg, args)
        valid_data0 = to_alpha0(valid_eeg, args)
        test_data0 = to_alpha0(test_eeg, args)

        train_data1 = to_alpha1(train_eeg, args)
        valid_data1 = to_alpha1(valid_eeg, args)
        test_data1 = to_alpha1(test_eeg, args)

        train_data2 = to_alpha2(train_eeg, args)
        valid_data2 = to_alpha2(valid_eeg, args)
        test_data2 = to_alpha2(test_eeg, args)

        train_data3 = to_alpha3(train_eeg, args)
        valid_data3 = to_alpha3(valid_eeg, args)
        test_data3 = to_alpha3(test_eeg, args)

        train_data4 = to_alpha4(train_eeg, args)
        valid_data4 = to_alpha4(valid_eeg, args)
        test_data4 = to_alpha4(test_eeg, args)

        if use_image:

            train_data0 = gen_images(train_data0, args)
            valid_data0 = gen_images(valid_data0, args)
            test_data0 = gen_images(test_data0, args)

            train_data1 = gen_images(train_data1, args)
            valid_data1 = gen_images(valid_data1, args)
            test_data1 = gen_images(test_data1, args)

            train_data2 = gen_images(train_data2, args)
            valid_data2 = gen_images(valid_data2, args)
            test_data2 = gen_images(test_data2, args)

            train_data3 = gen_images(train_data3, args)
            valid_data3 = gen_images(valid_data3, args)
            test_data3 = gen_images(test_data3, args)

            train_data4 = gen_images(train_data4, args)
            valid_data4 = gen_images(valid_data4, args)
            test_data4 = gen_images(test_data4, args)

            input_train_data = np.stack([train_data0, train_data1, train_data2, train_data3, train_data4], axis=1)
            valid_data = np.stack([valid_data0, valid_data1, valid_data2, valid_data3, valid_data4], axis=1)
            test_data = np.stack([test_data0, test_data1, test_data2, test_data3, test_data4], axis=1)

            if is_3D:

                fre_train_data = np.expand_dims(input_train_data, axis=-1)  # (N, 5, 32 , 32 , 1)
                fre_valid_data = np.expand_dims(valid_data, axis=-1)
                fre_test_data = np.expand_dims(test_data, axis=-1)

            else:
                fre_train_data = input_train_data
                fre_valid_data = valid_data
                fre_test_data  = test_data


            # DBPNet
            # input_train_data = np.stack([train_data0, train_data1, train_data2, train_data3, train_data4], axis=1)  # (N, 5, 32, 32)
            # valid_data = np.stack([valid_data0, valid_data1, valid_data2, valid_data3, valid_data4], axis=1)
            # test_data = np.stack([test_data0, test_data1, test_data2, test_data3, test_data4], axis=1)
            #
            # fre_train_data = np.expand_dims(input_train_data, axis=-1)  # (N, 5, 32 , 32 , 1)
            # fre_valid_data = np.expand_dims(valid_data, axis=-1)
            # fre_test_data = np.expand_dims(test_data, axis=-1)


        else:
            input_train_data = np.stack([train_data0, train_data1, train_data2, train_data3, train_data4],
                                        axis=1)  # (N, 5, 64)
            valid_data = np.stack([valid_data0, valid_data1, valid_data2, valid_data3, valid_data4], axis=1)
            test_data = np.stack([test_data0, test_data1, test_data2, test_data3, test_data4], axis=1) # # (N, 5, 64)


            fre_train_data = input_train_data
            fre_valid_data = valid_data
            fre_test_data  = test_data

        ############ CSP #############

        # eeg_data = eeg_data[:, :args.eeg_channel, :]
        # label = np.array(event_data).squeeze()
        # train_eeg, test_eeg, train_label, test_label = sliding_window(eeg_data, label, args, args.eeg_channel)


        seq_train_data = train_eeg
        seq_valid_data = valid_eeg
        seq_test_data = test_eeg
        # del data


        indices = np.arange(len(train_label))
        np.random.seed(seed)
        np.random.shuffle(indices)

        fre_train_data = fre_train_data[indices]
        seq_train_data = seq_train_data[indices]
        train_label = train_label[indices]


        # seq_train_data, seq_valid_data, fre_train_data, fre_valid_data, train_label, valid_label = train_test_split(
        #     seq_train_data, fre_train_data, train_label, test_size=0.1, random_state=seed)


        csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space',
                  norm_trace=True)

        seq_train_data = seq_train_data.transpose(0, 2, 1)
        seq_train_data = csp.fit_transform(seq_train_data, train_label.squeeze())  # 对训练集进行拟合和转换

        seq_valid_data = seq_valid_data.transpose(0, 2, 1)
        seq_valid_data = csp.transform(seq_valid_data)  # 只进行特征转换，不进行拟合

        seq_test_data = seq_test_data.transpose(0, 2, 1)
        seq_test_data = csp.transform(seq_test_data)  # 只进行特征转换，不进行拟合
        if is_3D:
            # DBPNet
            fre_train_data = fre_train_data.transpose(0, 4, 1, 2, 3)
            fre_valid_data = fre_valid_data.transpose(0, 4, 1, 2, 3)
            fre_test_data = fre_test_data.transpose(0, 4, 1, 2, 3)

        train_loader = DataLoader(dataset=DualCustomDatasets(seq_train_data, fre_train_data, train_label),
                                  batch_size=args.batch_size, drop_last=False, num_workers=8)
        valid_loader = DataLoader(dataset=DualCustomDatasets(seq_valid_data, fre_valid_data, valid_label),
                                  batch_size=args.batch_size, drop_last=False, num_workers=8)
        test_loader = DataLoader(dataset=DualCustomDatasets(seq_test_data, fre_test_data, test_label),
                                 batch_size=args.batch_size, drop_last=False, num_workers=8)

    args.n_train = np.size(train_label)
    args.n_valid = np.size(valid_label)
    args.n_test = np.size(test_label)

    if branch == 1 :
        args.data_shape = train_loader.dataset.data.shape

    # Count occurrences in each label dataset
    count_labels(train_label, "train_label", logger)
    count_labels(valid_label, "valid_label", logger)
    count_labels(test_label, "test_label", logger)

    return train_loader, valid_loader, test_loader
