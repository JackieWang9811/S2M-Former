import math
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class SSFCNN(nn.Module):
    def __init__(self):
        super(SSFCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 16 * 16, 512)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 32)
        self.batch_norm3 = nn.BatchNorm1d(32)

        self.fc3 = nn.Linear(32, 2)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.pool1(nn.ReLU()(self.batch_norm1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.flatten(x)
        x = nn.ReLU()(self.batch_norm2(self.fc1(x)))
        x = self.dropout2(x)
        x = nn.ReLU()(self.batch_norm3(self.fc2(x)))
        # x = nn.Softmax(dim=1)(self.fc3(x))
        x = self.fc3(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    import os
    from dotmap import DotMap
    # set_random_seeds(200)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # 设置参数
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = DotMap()
    length = 2

    # args.in_channels = 5
    """
    #########################
    ##  Training Parameter ##
    #########################
    """
    args.lr = 3e-3
    args.weight_decay = 1e-2
    args.T_max = 100
    args.scale = True
    args.tem_tokens = math.ceil(128 * length)
    args.csp_dim = 32
    args.fre_tokens = args.eeg_channel
    args.dataset = "DTUDataset"


    # 初始化模型
    model = SSFCNN().to(device)

    print(model)
    fre_tensor = torch.rand([32, 1, 32, 32]).to(device)  # (batch_size, csp_dim, sample_points)
    # 前向传播
    output = model(fre_tensor)
    # print(output)
    print("Output shape:", output.shape)
    param_m = count_parameters(model) / 1e6  #
    print("Model size: {:.2f}M".format(param_m))  # Model size: 4.21M