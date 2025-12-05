import sys
sys.path.append('/data/wjq/AAD/OpenAAD/')

from dotmap import DotMap
from tools.utils import *
from tools.data_loader_subject_independent import getData
from model_zoo.S2MFormer import SpikingBranchformer # ****
import numpy as np
import torch
from spikingjelly.activation_based import surrogate,functional

import torch.nn as nn
import torch.optim as optim

np.set_printoptions(suppress=True)
from tqdm import tqdm
from torch.utils.data import Dataset
import pickle


def log_selected_args(args, logger):
    selected_keys = ['lr', 'weight_decay', 'T_max']
    for key in selected_keys:
        if key in args.keys():
            logger.info(f'{key}: {args[key]}')

def count_labels(labels, label_name, logger=None):
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique, counts))
    print(f"Counts in {label_name}: {label_counts}")
    logger.info(f"Counts in {label_name}: {label_counts}")

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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 训练前初始化配置
def initiate(args, fold_data, logger, subject_id):

    # 从 fold_data 中获取训练集和测试集数据
    csp_train_eeg = fold_data['csp_train_eeg']
    de_train_eeg = fold_data['de_train_eeg']
    train_label = fold_data['train_label']

    csp_test_eeg = fold_data['csp_test_eeg']
    de_test_eeg = fold_data['de_test_eeg']
    test_label = fold_data['test_label']

    # 创建自定义数据集和数据加载器
    train_dataset = DualCustomDatasets(csp_train_eeg, de_train_eeg, train_label)
    test_dataset = DualCustomDatasets(csp_test_eeg, de_test_eeg, test_label)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,  shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = SpikingBranchformer(args)

    count_labels(train_label, "train_label", logger)
    count_labels(test_label, "test_label", logger)
    # 打印模型参数量
    print(model)
    print(f"The model has {count_parameters(model):,} trainable parameters.")
    logger.info(f"The model has {count_parameters(model):,} trainable parameters.")

    # 获取日志文件的目录
    log_dir = os.path.dirname(logger.handlers[0].baseFilename)
    # 创建模型结构文件的路径
    model_file_path = os.path.join(log_dir, f"model_structure.txt")

    # 保存模型结构到单独的文件中
    with open(model_file_path, 'w') as f:
        f.write(str(model))
        f.write(f"\nThe model has {count_parameters(model):,} trainable parameters.\n")

    # Save args parameters
    args_file_path = os.path.join(log_dir, f"args_parameters.txt")
    with open(args_file_path, 'w') as f:
        for key, value in args.items():
            f.write(f"{key}: {value}\n")

    # 获取损失函数
    criterion = nn.CrossEntropyLoss()

    if args.dataset == "DTUDataset":

        #########################
        ##     DTU Training    ##
        #########################

        args.lr = 2e-3
        args.weight_decay = 1e-2
        log_selected_args(args, logger)
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1, eta_min= (args.lr*3)/ 20)
    elif args.dataset == "KULDataset":
        #########################
        ##     KUL Training    ##
        #########################
        args.lr = 2e-3
        # args.lr = 2e-4
        args.weight_decay = 1e-2
        log_selected_args(args, logger)
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min= (args.lr*3)/ 20)

    elif "AVGCDataset_across_conditions" in args.dataset:
        #########################
        ##     AVGC Training    ##
        #########################
        args.lr = 5e-4
        args.weight_decay = 1e-2
        log_selected_args(args, logger)
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min= (args.lr*3)/ 20)


    model = model.cuda()
    criterion = criterion.cuda()

    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}

    return train_model(settings, args, train_loader, test_loader, subject_id, logger)

def train_model(settings, args, train_loader, test_loader, subject_id, logger):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']

    def train(model, optimizer, criterion, scheduler, epoch):
        model.train()
        proc_loss, proc_size = 0, 0
        train_acc_sum = 0
        train_loss_sum = 0
        num_batches = len(train_loader)
        for i_batch, batch_data in enumerate(train_loader):
            seq_data, fre_data, train_label = batch_data
            train_label = train_label.squeeze(-1)
            seq_data, fre_data, train_label = seq_data.cuda(), fre_data.cuda(), train_label.cuda()
            # seq_data = seq_data.permute(0,2,1)
            batch_size = train_label.size(0)

            # Forward pass
            preds = model(seq_data, fre_data)
            loss = criterion(preds, train_label.long())
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 如果使用多层的清况下，是否会有效果？
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            functional.reset_net(model)
            proc_loss += loss.item() * batch_size
            proc_size += batch_size
            train_loss_sum += loss.item() * batch_size
            predicted = preds.data.max(1)[1]
            train_acc_sum += predicted.eq(train_label).cpu().sum()

            if args.verbose:
                if i_batch % args.log_interval == 0 and i_batch > 0 and i_batch < num_batches:
                    avg_loss = proc_loss / proc_size
                    avg_acc = train_acc_sum / proc_size
                    print('Epoch {:2d} | Batch {:3d}/{:3d} | Train Loss {:5.4f} | Train Acc {:5.4f}'.
                          format(epoch, i_batch, num_batches, avg_loss, avg_acc))
                    logger.info('Epoch {:2d} | Batch {:3d}/{:3d} | Train Loss {:5.4f} | Train Acc {:5.4f}'.
                                format(epoch, i_batch, num_batches, avg_loss, avg_acc))
                    proc_loss, proc_size, train_acc_sum = 0, 0, 0

        # 更新学习率
        scheduler.step()

        return train_loss_sum / len(train_loader.dataset)

    def evaluate(model, criterion, loader, num_samples):
        model.eval()
        total_loss = 0.0
        test_acc_sum = 0
        with torch.no_grad():
            for i_batch, batch_data in enumerate(loader):
                seq_data, fre_data, test_label = batch_data
                test_label = test_label.squeeze(-1)
                seq_data, fre_data, test_label = seq_data.cuda(), fre_data.cuda(), test_label.cuda()
                # seq_data = seq_data.permute(0, 2, 1)
                preds = model(seq_data, fre_data)
                loss = criterion(preds, test_label.long())
                total_loss += loss.item() * test_label.size(0)

                predicted = preds.data.max(1)[1]
                test_acc_sum += predicted.eq(test_label).cpu().sum()
                functional.reset_net(model)

        avg_loss = total_loss / num_samples
        avg_acc = test_acc_sum / num_samples
        return avg_loss, avg_acc

    best_epoch = 1
    best_test_acc = 0.0
    best_epoch_acc = 0
    stale_acc = 0

    for epoch in tqdm(range(1, args.max_epoch + 1), desc='Training Epoch', leave=False):
        train_loss = train(model, optimizer, criterion, scheduler, epoch)
        # 直接在测试集上评估
        test_loss, test_acc = evaluate(model, criterion, test_loader, len(test_loader.dataset))

        print("-" * 50)
        print(
            'Epoch {:2d} Finish | Subject {} | Train Loss {:5.4f} | Test Loss {:5.4f} | Test Acc {:5.4f}'.format(
                epoch,
                subject_id,
                train_loss,
                test_loss,
                test_acc))
        logger.info(
            'Epoch {:2d} Finish | Subject {} | Train Loss {:5.4f} | Test Loss {:5.4f} | Test Acc {:5.4f}'.format(
                epoch,
                subject_id,
                train_loss,
                test_loss,
                test_acc))
        print("-" * 50)

        # 检查测试集准确率是否上升
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"保存最佳准确率模型 S{subject_id}_best_acc.pt!")
            logger.info(f"保存最佳准确率模型 S{subject_id}_best_acc.pt!")
            save_model_loso(args, model, name=f"{subject_id}_best_acc")
            stale_acc = 0
        else:
            stale_acc += 1

        # 实现基于准确率的早停
        if stale_acc > args.patience:
            logger.info(f"在第 {epoch} 轮由于准确率未提升，提前停止训练！")
            print(f"在第 {epoch} 轮由于准确率未提升，提前停止训练！")
            break

    logger.info(f"Subject: {subject_id}, Acc: {best_test_acc:.2f}")
    print(f"Subject: {subject_id}, Acc: {best_test_acc:.2f}")

    return best_test_acc  # 返回测试准确率以便在主函数中汇总





def main(name="S1", data_document_path="/data/wjq/AAD/DTUDataset", ConType="No_vanilla", batch_size= 128, model_name = None, length = 2, seed= 200, branch = 1, is_CSP = True, is_DE= False, split=None, logger=None): # ../DTUDataset   ../KULDataset

    """
    ####################################
    ##  Initialize Training Parameter ##
    ####################################
    """

    args = DotMap()
    args.model_name = model_name
    args.batch_size = batch_size
    args.csp_comp = 32
    args.T_max = 25
    args.verbose = True
    args.length = length
    args.name = name
    args.subject_number = 18
    args.data_document_path = data_document_path
    args.ConType = ConType
    args.overlap = 0.5
    args.max_epoch = 500
    args.patience = 25
    args.log_interval = 100
    args.image_size = 32
    args.eeg_channel = 64
    args.is_CSP = is_CSP
    args.is_DE = is_DE
    args.dbpnet = False
    args.use_image=True
    args.csp_comp = 64
    args.scale = True

    args.v_reset = 0.5
    args.chunk_size = 2

    ### SNN ###
    args.ts = 4

    args.init_tau = 2.0
    args.v_threshold = 1.0
    args.surrogate_function = surrogate.ATan(alpha=5.0)
    args.detach_reset = True
    args.backend = 'cupy'
    args.spike_mode = "lif"
    args.decay_input = True

    ### SpikingSCR  ###
    # args.dim = 8
    args.embed_dim = 8 # 模态投影维度  16
    args.hidden_dims = 16  # 单模态MLP隐藏层维度  # 48

    args.num_heads = 4  # 注意力头数量
    args.depths = 1  # Transformer 深度
    args.n_outputs = 2  # 输出维度 (假设任务是 10 分类)
    args.dropout_l = 0.0  # dropout 概率
    args.use_norm = True  # 是否使用正则化
    args.use_dp = True  # 是否使用 dropout
    args.use_dw_bias = True  # 深度卷积是否使用 bias
    args.use_rope = False
    args.bias = True
    args.patch_size = 2
    args.in_channels = 5
    args.image_size = 32
    args.total_tokens = math.ceil(128 * length) + (args.image_size // args.patch_size) **2
    args.tem_tokens = math.ceil(128 * length)
    args.fre_tokens = (args.image_size // args.patch_size) **2

    args, logger, _ = getData(name=name, data_document_path=data_document_path, ConType=ConType, length = length, seed= seed, branch = branch, is_CSP = is_CSP, is_DE= is_DE, split = split, logger=logger, args=args)

    all_test_accuracies = []
    for fold_idx in range(0, args.subject_number):
        print(f"正在训练第 {fold_idx + 1} 个折叠")
        logger.info(f"正在训练第 {fold_idx + 1} 个折叠")

        fold_file = os.path.join(
            f'/data/wjq/AAD/OpenAAD/cross_subjects_datasets/{ConType}/processed_data_LOSO_CSP_DE_{args.length}s',
            f'fold_{fold_idx + 1}_processed_{args.dataset}_data_{args.subject_number}_subjects_{args.length}s.pkl'
        )

        print(f"加载并训练第 {fold_idx + 1} 个fold: {fold_file}")
        # 仅加载当前 fold
        with open(fold_file, 'rb') as f:
            fold_data = pickle.load(f)

        # 训练并评估模型
        test_acc = initiate(args, fold_data, logger, subject_id=(fold_idx + 1))
        all_test_accuracies.append(test_acc)

    logger.info(f"所有折叠的准确率: {all_test_accuracies}")
    # 汇总并报告所有折叠的结果
    mean_acc = np.mean(all_test_accuracies) *100
    std_acc = np.std(all_test_accuracies) *100
    print(f"LOSO 交叉验证结果: 平均准确率 = {mean_acc:.2f}%, 标准差 = {std_acc:.2f}%")
    logger.info(f"LOSO 交叉验证结果: 平均准确率 = {mean_acc:.2f}%, 标准差 = {std_acc:.2f}%")


if __name__ == "__main__":
    torch.set_default_dtype(torch.double)
    main()
