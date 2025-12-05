import sys
sys.path.append('/data/wjq/AAD/OpenAAD/')
from dotmap import DotMap
from mne.decoding import CSP
from tools.utils import *
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
def getData(name="all", data_document_path="/data/wjq/AAD/DTUDataset", ConType="No_vanilla", length = 1, seed= 312, branch = 1, is_CSP = True, is_DE= False, split=None, logger=None, args=None):
    return get_data(name, data_document_path=data_document_path, ConType=ConType, length = length, seed= seed, branch = branch, is_CSP = is_CSP, is_DE= is_DE, split=split, logger=logger, args=args)

def get_data(name="all", data_document_path="/data/wjq/AAD/DTUDataset", ConType="No_vanilla", length = 1, seed= 312, branch = 1, is_CSP = True, is_DE= False,split= None, logger=None, args=None):

    set_random_seeds(seed)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.current_time = current_time

    if args is None:
        args = DotMap()
    args.data_document_path = data_document_path
    args.model_name = args.get("model_name")
    args.length = length
    args.name = name
    args.ConType = ConType


    if data_document_path == "/data/wjq/AAD/DTUDataset":
        args.trail_number = 60
        args.dataset = "DTUDataset"
        args.subject_number = 18
    elif data_document_path == "/data/wjq/AAD/KULDataset":
        args.trail_number = 8 # 8: KUL 60:DTU 20 trials = 4 + 4 + 12  - 第一次实验 4 次，第二次实验 4 次，第三次实验 12 次（实验 1 的 4 次演讲的前 2 分钟，重复 3 次）
        args.dataset = "KULDataset"
        args.subject_number = 16
    elif data_document_path == "/data/wjq/AAD/AVGCDataset_no_visuals":
        args.trail_number = 2
        args.dataset = "AVGCDataset_no_visuals"

    elif data_document_path == "/data/wjq/AAD/AVGCDataset_across_conditions":
        args.subject_number = 11
        args.dataset = "AVGCDataset_across_conditions"
        if args.name == "S1" or args.name == "S3":
            args.trail_number = 6
        elif args.name == "S14":
            args.trail_number = 7
        else:
            args.trail_number = 8

    elif data_document_path == "/data/wjq/AAD/AVGCDataset_across_conditions_zscore":
        args.subject_number = 11
        args.dataset = "AVGCDataset_across_conditions_zscore"
        if args.name == "S1" or args.name == "S3":
            args.trail_number = 6
        elif args.name == "S14":
            args.trail_number = 7
        else:
            args.trail_number = 8



    args.label_col = 0

    """
    #########################
    ##  Training Parameter ##
    #########################
    """
    args.seed = seed

    dataset_name = os.path.basename(args.data_document_path)

    if logger == None:
        if is_CSP == True:
            args.log_path = f"./result/{dataset_name}/{args.ConType}/{args.model_name}/{length}s/seed_{seed}_csp_comp_{args.csp_comp}_{current_time}"
            logger = get_logger(args.name, args.log_path, length)
        else:
            args.log_path = f"./result/{dataset_name}/{args.ConType}/{args.model_name}/{length}s/seed_{seed}_{current_time}"
            logger = get_logger(args.name, args.log_path, length)
        log_selected_args(args, logger)


    print(args.ConType)
    args.fs = 128
    logger.info(f"{args.fs}")
    print(f"{args.fs}")
    args.window_length = math.ceil(
        args.fs * length)  # eg: 1s : window_length = 128*1;  2s: window_length = 128*2 = 256


    set_random_seeds(seed)
    if branch ==1 and is_CSP and is_DE == False:
        save_dir = f'/data/wjq/AAD/OpenAAD/cross_subjects_datasets/{args.ConType}/processed_data_LOSO_CSP_{length}s'
    elif branch ==1 and is_CSP == False and is_DE ==True:
        save_dir = f'/data/wjq/AAD/OpenAAD/cross_subjects_datasets/{args.ConType}/processed_data_LOSO_DE_{length}s'
    elif branch ==2:
        if args.dbpnet == True:
            save_dir = f'/data/wjq/AAD/OpenAAD/cross_subjects_datasets/{args.ConType}/processed_data_LOSO_CSP_DE_DBPNet_{args.length}s'
        else:
            save_dir = f'/data/wjq/AAD/OpenAAD/cross_subjects_datasets/{args.ConType}/processed_data_LOSO_CSP_DE_{args.length}s'

    os.makedirs(save_dir, exist_ok=True)
    all_fold_files = []
    for fold_idx in range(1, args.subject_number + 1):
        fold_file = os.path.join(
            save_dir,
            f'fold_{fold_idx}_processed_{args.dataset}_data_{args.subject_number}_subjects_{args.length}s.pkl'
        )
        all_fold_files.append(fold_file)
    # 2) 检查这些文件是否全部存在
    all_exist = all(os.path.exists(f) for f in all_fold_files)
    if all_exist:
        print("所有 Fold 文件都已经存在，跳过 `read_prepared_data_all_subjects` 和滑窗预处理。")
        return args, logger, []

    else:
        print("至少有部分 fold 文件不存在，需要先读取原始数据并做预处理……")
        # 这时再读取
        eeg_data, event_data = read_prepared_data_all_subjects(args)

        if args.dataset == "DTUDataset":
            # if eeg_data[0][0].shape[1] == 6400:
            args.fs = 128
            logger.info(f"{args.fs}")
            print(f"{args.fs}")
            args.window_length = math.ceil(
                args.fs * length)  # eg: 1s : window_length = 128*1;  2s: window_length = 128*2 = 256
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
            # else:
            #     args.fs = 64
            #     logger.info(f"{args.fs}")
            #     print(f"{args.fs}")
            #     args.window_length = math.ceil(
            #         args.fs * length)  # eg: 0.1s window_length = 64*0.1 = 7(向上取整) 1s : window_length = 64*1;  2s: window_length = 64*2 = 128,
            #     args.delta_low = 1
            #     args.delta_high = 3
            #     args.theta_low = 4
            #     args.theta_high = 7
            #     args.alpha_low = 8
            #     args.alpha_high = 13
            #     args.beta_low = 14
            #     args.beta_high = 30
            #     args.gamma_low = 31
            #     args.gamma_high = 32
            #     args.frequency_resolution = args.fs / args.window_length
            #     args.point0_low = math.ceil(args.delta_low / args.frequency_resolution)
            #     args.point0_high = math.ceil(args.delta_high / args.frequency_resolution) + 1
            #     args.point1_low = math.ceil(args.theta_low / args.frequency_resolution)
            #     args.point1_high = math.ceil(args.theta_high / args.frequency_resolution) + 1
            #     args.point2_low = math.ceil(args.alpha_low / args.frequency_resolution)
            #     args.point2_high = math.ceil(args.alpha_high / args.frequency_resolution) + 1
            #     args.point3_low = math.ceil(args.beta_low / args.frequency_resolution)
            #     args.point3_high = math.ceil(args.beta_high / args.frequency_resolution) + 1
            #     args.point4_low = math.ceil(args.gamma_low / args.frequency_resolution)
            #     args.point4_high = math.ceil(args.gamma_high / args.frequency_resolution) + 1

        elif args.dataset == "KULDataset":
            # if eeg_data[0][0].shape[1] >= 46080:
            args.fs = 128
            logger.info(f"{args.fs}")
            print(f"{args.fs}")
            args.window_length = math.ceil(
                args.fs * length)  # eg: 1s : window_length = 128*1;  2s: window_length = 128*2 = 256
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

            # else:
            #     args.fs = 64
            #     logger.info(f"{args.fs}")
            #     print(f"{args.fs}")
            #     args.window_length = math.ceil(
            #         args.fs * length)  # eg: 0.1s window_length = 64*0.1 = 7(向上取整) 1s : window_length = 64*1;  2s: window_length = 64*2 = 128,
            #     args.delta_low = 1
            #     args.delta_high = 3
            #     args.theta_low = 4
            #     args.theta_high = 7
            #     args.alpha_low = 8
            #     args.alpha_high = 13
            #     args.beta_low = 14
            #     args.beta_high = 30
            #     args.gamma_low = 31
            #     args.gamma_high = 32
            #     args.frequency_resolution = args.fs / args.window_length
            #     args.point0_low = math.ceil(args.delta_low / args.frequency_resolution)
            #     args.point0_high = math.ceil(args.delta_high / args.frequency_resolution) + 1
            #     args.point1_low = math.ceil(args.theta_low / args.frequency_resolution)
            #     args.point1_high = math.ceil(args.theta_high / args.frequency_resolution) + 1
            #     args.point2_low = math.ceil(args.alpha_low / args.frequency_resolution)
            #     args.point2_high = math.ceil(args.alpha_high / args.frequency_resolution) + 1
            #     args.point3_low = math.ceil(args.beta_low / args.frequency_resolution)
            #     args.point3_high = math.ceil(args.beta_high / args.frequency_resolution) + 1
            #     args.point4_low = math.ceil(args.gamma_low / args.frequency_resolution)
            #     args.point4_high = math.ceil(args.gamma_high / args.frequency_resolution) + 1

        elif "AVGCDataset" in args.dataset:
            args.fs = 128
            logger.info(f"{args.fs}")
            print(f"{args.fs}")
            args.window_length = math.ceil(
                args.fs * length)  # eg: 1s : window_length = 128*1;  2s: window_length = 128*2 = 256
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

        args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)

        if split== "loso":
            if branch == 1:
                if is_CSP:
                    sliding_window_LOSO_CSP_all_subject(eeg_data, event_data, args, args.eeg_channel, save_dir)
                    return args, logger, []
                elif is_DE:
                    sliding_window_LOSO_DE_add_valid(eeg_data, event_data, args, args.eeg_channel, save_dir)
                    return args, logger, []
            elif branch ==2:
                sliding_window_LOSO_CSP_DE_all_subject(eeg_data, event_data, args, args.eeg_channel, save_dir)
                return args, logger, []


def get_KUL_data(name="S1", data_document_path="/data/wjq/AAD/DTUDataset", ConType="No_vanilla", length = 1, seed= 312, branch = 1, is_CSP = True, is_DE= False, logger=None, args=None):
    pass