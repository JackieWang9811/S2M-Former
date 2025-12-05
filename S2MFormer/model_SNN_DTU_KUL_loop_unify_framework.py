import sys
sys.path.append('/data/wjq/AAD/OpenAAD/')

from dotmap import DotMap
from tools.data_loader_subject_dependent import getData
from tools.utils import *
from tools.function import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from spikingjelly.activation_based import surrogate,functional

from model_zoo.S2MFormer import SpikingBranchformer # ****


np.set_printoptions(suppress=True)
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def log_selected_args(args, logger):
    selected_keys = ['lr', 'weight_decay', 'T_max']
    for key in selected_keys:
        if key in args.keys():
            logger.info(f'{key}: {args[key]}')



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 训练前初始化配置
def initiate(args, train_loader, valid_loader, test_loader, subject, logger):
    model = SpikingBranchformer(args)

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

    criterion = nn.CrossEntropyLoss()


    if args.dataset == "DTUDataset":

        #########################
        ##     DTU Training    ##
        #########################

        if args.is_cross_trials:
            args.lr =2e-4
        else:
            args.lr = 5e-4
        args.weight_decay = 1e-2 #
        log_selected_args(args, logger)
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1, eta_min= (args.lr*3)/ 20)
    elif args.dataset == "KULDataset":
        #########################
        ##     KUL Training    ##
        #########################
        if args.is_cross_trials:
            args.lr =2e-4
        else:
            args.lr = 1e-3 # 1e-3
        args.weight_decay = 1e-2
        log_selected_args(args, logger)
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=3, eta_min= (args.lr*3)/ 10)

    elif "AVGCDataset_across_conditions" in args.dataset:
        #########################
        ##     AVGC Training    ##
        #########################、
        if args.is_cross_trials:
            args.lr =2e-4
        else:
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

    return train_model(settings, args, train_loader, valid_loader, test_loader, subject,logger)

def train_model(settings, args, train_loader, valid_loader, test_loader, subject, logger):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']

    def train(model, optimizer, criterion, scheduler):
        model.train()
        proc_loss, proc_size = 0, 0
        num_batches = int(args.n_train // args.batch_size)
        train_acc_sum = 0
        train_loss_sum = 0
        for i_batch, batch_data in enumerate(train_loader):
            seq_data, fre_data, train_label = batch_data
            train_label = train_label.squeeze(-1)
            seq_data, fre_data, train_label = seq_data.cuda(), fre_data.cuda(), train_label.cuda()
            # seq_data = seq_data.permute(0, 2, 1)
            batch_size = train_label.size(0)

            # Forward pass
            preds = model(seq_data, fre_data)
            loss = criterion(preds, train_label.long())
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            functional.reset_net(model)
            proc_loss += loss.item() * batch_size
            proc_size += batch_size
            train_loss_sum += loss.item() * batch_size
            predicted = preds.data.max(1)[1]
            train_acc_sum += predicted.eq(train_label).cpu().sum()

            if i_batch % args.log_interval == 0 and i_batch > 0 and i_batch < num_batches:
                avg_loss = proc_loss / proc_size
                avg_acc = train_acc_sum / proc_size
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Train Loss {:5.4f} | Train Acc {:5.4f}'.
                      format(epoch, i_batch, num_batches, avg_loss, avg_acc))
                proc_loss, proc_size, train_acc_sum = 0, 0, 0

        scheduler.step()

        return train_loss_sum / args.n_train

    def evaluate(model, criterion, test=False):
        model.eval()
        if test:
            loader = test_loader
            num_batches = args.n_test
        else:
            loader = valid_loader
            num_batches = args.n_valid
        total_loss = 0.0
        test_acc_sum = 0
        proc_size = 0

        with torch.no_grad():
            for i_batch, batch_data in enumerate(loader):
                seq_data, fre_data, test_label = batch_data
                test_label = test_label.squeeze(-1)
                seq_data, fre_data, test_label = seq_data.cuda(), fre_data.cuda(), test_label.cuda()
                # seq_data = seq_data.permute(0, 2, 1)
                proc_size += args.batch_size
                preds = model(seq_data, fre_data)

                # Backward and optimize
                optimizer.zero_grad()

                total_loss += criterion(preds, test_label.long()).item() * args.batch_size

                predicted = preds.data.max(1)[1]  # 32
                test_acc_sum += predicted.eq(test_label).cpu().sum()
                functional.reset_net(model)

        avg_loss = total_loss / num_batches

        avg_acc = test_acc_sum /num_batches

        return avg_loss, avg_acc

    best_epoch = 1
    best_valid_loss = float('inf')
    best_valid_acc = 0.0
    for epoch in tqdm(range(1, args.max_epoch + 1)):
        print()
        train_loss = train(model, optimizer, criterion, scheduler)
        val_loss, val_acc = evaluate(model, criterion, test=False)

        print("-" * 50)
        print(
            'Epoch {:2d} Finsh | Subject {} | Train Loss {:5.4f} | Valid Loss {:5.4f} | Valid Acc {:5.4f}'.format(epoch,
                                                                                                                  args.subject_number,
                                                                                                                  train_loss,
                                                                                                                  val_loss,
                                                                                                                  val_acc))
        logger.info(
            'Epoch {:2d} Finsh | Subject {} | Train Loss {:5.4f} | Valid Loss {:5.4f} | Valid Acc {:5.4f}'.format(epoch,
                                                                                                                  args.subject_number,
                                                                                                                  train_loss,
                                                                                                                  val_loss,
                                                                                                                  val_acc))
        print("-" * 50)

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best_epoch_loss = epoch
            print(f"Saved best loss model at pre_trained_models/{save_load_name(args, name=args.name)}_loss.pt!")
            save_model(args, model, name=f"{args.name}_loss")
            stale_loss = 0
        else:
            stale_loss += 1

        # Check if current validation accuracy is higher than the best recorded one
        if val_acc > best_valid_acc:
            best_valid_acc = val_acc
            best_epoch_acc = epoch
            print(f"Saved best accuracy model at pre_trained_models/{save_load_name(args, name=args.name)}_acc.pt!")
            save_model(args, model, name=f"{args.name}_acc")
            stale_acc = 0
        else:
            stale_acc += 1

        # Implementing early stopping based on both loss and accuracy
        if stale_loss > args.patience and stale_acc > args.patience:
            logger.info(f"Early stopping at epoch {epoch} due to no improvement in both loss and accuracy!")
            break

    # 加载基于最佳损失的模型并评估
    model_best_loss = load_model_new(args, model_type='loss')
    test_loss, test_acc = evaluate(model_best_loss, criterion, test=True)
    logger.info(f'Best epoch loss: {best_epoch_loss}')
    print(f'Best epoch loss: {best_epoch_loss}')

    logger.info(f"Test results for best loss model: Loss = {test_loss}, Accuracy = {test_acc}")
    print(f"Test results for best loss model: Loss = {test_loss}, Accuracy = {test_acc}")

    logger.info(f"Subject: {subject}, Acc: {test_acc:.2f}")
    print(f"Subject: {subject}, Acc: {test_acc:.2f}")


    # 加载基于最佳准确率的模型并评估
    model_best_acc = load_model_new(args, model_type='acc')
    test_loss_acc, test_acc_acc = evaluate(model_best_acc, criterion, test=True)
    logger.info(f'Best epoch acc: {best_epoch_acc}')
    print(f'Best epoch acc: {best_epoch_acc}')

    logger.info(f"Test results for best accuracy model: Loss = {test_loss_acc}, Accuracy = {test_acc_acc}")
    print(f"Test results for best accuracy model: Loss = {test_loss_acc}, Accuracy = {test_acc_acc}")

    logger.info(f"Subject: {subject}, Acc: {test_acc_acc:.2f}")
    print(f"Subject: {subject}, Acc: {test_acc_acc:.2f}")


    # 比较两个模型的准确率，并打印出最高的准确率
    if test_acc > test_acc_acc:

        logger.info(f"The highest accuracy is obtained by the best loss {test_acc}")
        print(f"The highest accuracy is obtained by the best loss {test_acc}")
    else:
        logger.info(
            f"The highest accuracy is obtained by the best acc {test_acc_acc}")
        print(f"The highest accuracy is obtained by the best acc {test_acc_acc}")

    # model = load_model_new(args, name=args.name)
    # test_loss, test_acc = evaluate(model, criterion, test=True)

    return test_loss, test_acc

def set_random_seeds(seed):
    ''' Set random seeds for reproducibility. '''
    seed_val = seed
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_selected_args(args, logger):
    selected_keys = ['lr', 'weight_decay', 'T_max']
    for key in selected_keys:
        if key in args.keys():
            logger.info(f'{key}: {args[key]}')


# Add these lines just after splitting the data in the `main` function
def count_labels(labels, label_name, logger=None):
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique, counts))
    print(f"Counts in {label_name}: {label_counts}")
    logger.info(f"Counts in {label_name}: {label_counts}")


def main(name="S1", data_document_path="/data/wjq/AAD/DTUDataset", ConType="No_vanilla", is_cross_trials=None, model_name = None, length = 2, seed= 200, branch = 1, is_CSP = True, is_DE= False, use_image = True, logger=None): # ../DTUDataset


    set_random_seeds(seed)

    args = DotMap()
    args.model_name = model_name
    args.length = length
    args.name = name
    args.subject_number = int(args.name[1:])
    args.data_document_path = data_document_path
    args.ConType = ConType
    args.overlap = 0.5
    args.batch_size = 32
    args.max_epoch = 500
    args.patience = 25
    args.log_interval = 100 if args.length == 0.1 else 20
    args.image_size = 32
    args.eeg_channel = 64
    args.csp_comp = 64
    args.scale = True
    args.is_cross_trials = is_cross_trials
    args.v_reset = 0.5
    args.chunk_size = 2

    ### SNN ###
    if args.is_cross_trials:
        args.ts = 4  # KUL AVGC 4 time steps DTU 4 time steps
    else:
        args.ts = 6  # KUL DTU 4 time steps AVGC 6 time steps

    args.init_tau = 2.0
    args.v_threshold = 1.0
    args.surrogate_function = surrogate.ATan(alpha=5.0)
    args.detach_reset = True
    args.backend = 'torch' # 'cupy' 'torch'
    args.spike_mode = "lif"
    args.decay_input = True

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


    args,  logger, train_loader, valid_loader, test_loader = getData(name=name, data_document_path=data_document_path, is_cross_trials=is_cross_trials,
                                                                     ConType=ConType, length = length, seed= seed, branch = branch,
                                                                     is_CSP = is_CSP, is_DE= is_DE,use_image=use_image, logger=logger, args=args)

    # 训练
    initiate(args, train_loader, valid_loader, test_loader, args.name, logger)


if __name__ == "__main__":
    # torch.set_default_dtype(torch.double)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print("Start")
    path = "/data/wjq/AAD/DTUDataset"  # ./DTUDataset  ./KULDataset
    # path = "/data/wjq/AAD/DTUDataset" # ./DTUDataset  ./KULDataset
    if path == "/data/wjq/AAD/KULDataset":
        dataset = "KULDataset"
        sub_names = ['S' + str(i+1) for i in range(0, 16)]
    elif path == "/data/wjq/AAD/DTUDataset":
        dataset = "DTUDataset"
        sub_names = ['S' + str(i + 1) for i in range(0, 1)]

    """
    Dataset
    """

    ConType = "No_vanilla_128" # No_vanilla 注意，DARNet在论文中明确说了是滤波至64Hz进行的，当然也可以设置128Hz的滤波测试下结果
    seed = 200 # 312 200 64 0 1111 32 10086 2222
    model_name = "S2MFormer"
    window_lengths = 2 # 这里应该就是传入的 0.1/ 1 / 2

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_path = f"./result/{dataset}/{ConType}/{model_name}/{window_lengths}s/seed_{seed}_{current_time}"

    for sub in sub_names:
        logger = get_logger(sub, log_path, window_lengths)
        main(sub, path, ConType, window_lengths, seed, logger)

