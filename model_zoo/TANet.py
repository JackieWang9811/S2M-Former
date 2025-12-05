import math
import torch
import torch.nn as nn

class MySpaceAttention(nn.Module):
    def __init__(self, channels_num, sample_len, sa_block_num, se_cnn_num=10, dropout=0.5):
        super(MySpaceAttention, self).__init__()
        self.sa_block_num = sa_block_num
        self.sample_len = sample_len
        self.channels_num = channels_num

        # Sequential model to match the Keras implementation
        self.conv2d = nn.Conv2d(in_channels=channels_num, out_channels=se_cnn_num, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.bn2d = nn.BatchNorm2d(num_features=se_cnn_num)
        self.elu1 = nn.ReLU() # Activation ELU to match Keras
        self.maxpool2d = nn.MaxPool2d((se_cnn_num, 1),stride=(1, 1)) #
        # self.avgpool2d = nn.AvgPool2d((se_cnn_num, 1), stride=(1, 1))  #
        # self.avgpool2d = nn.AdaptiveAvgPool2d(1)
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.sample_len, self.sample_len//4)
        self.elu2 = nn.ReLU() # Activation ELU
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.sample_len//4, self.sample_len)
        # self.elu3 = nn.ReLU() # Activation ELU

    def forward(self, x):
        input = x
        batch, sequence_len, channels_num = x.size()

        # Reshape 输入以匹配 Conv2d 的需求 (batch, 1, sequence_len, channels_num)
        # x = x.view(batch, channels_num, sequence_len, 1)
        x = x.reshape(batch, channels_num, sequence_len, 1)
        # 卷积和池化操作
        x = self.conv2d(x)  # (16,10, sequence_len, 1)
        x = self.bn2d(x)
        x = self.elu1(x)
        x = x.permute(0, 2, 1, 3) # (16,10, sequence_len, 1) => (16, sequence_len, 10, 1)
        x = self.maxpool2d(x)  # (16,sequence_len, channels_num, 1)
        # x = self.avgpool2d(x) # (16,sequence_len, channels_num, 1)
        x = x.view(batch, sequence_len)
        x = self.drop1(x)

        # 展平操作，进入全连接层
        # x = x.view(batch, -1)  # 展平为 (batch, se_cnn_num)
        x = self.elu2(self.fc1(x))
        x = self.drop2(x)
        # attention_weights = self.elu3(self.fc2(x))  # 输出大小 (batch, channels_num)
        # x = sself.fc1(x)
        attention_weights = self.fc2(x)
        # 生成注意力权重，使用 Sigmoid 来限制在 [0, 1] 范围内
        # attention_weights = torch.sigmoid(attention_weights)
        # attention_weights = torch.softmax(attention_weights, dim=-2)

        attention_weights = attention_weights.view(batch, sequence_len, 1)  # (batch, 1, channels_num)
        # attention_weights = attention_weights.repeat(1, 1, channels_num)  # 扩展至 (batch, sequence_len, channels_num)

        # attention_weights = torch.sigmoid(attention_weights)

        # 应用注意力权重
        output = input * attention_weights  # 对原始输入应用注意力权重

        return output


class MyTemporalAttention(nn.Module):
    def __init__(self, sample_len, cnn_block_len, dim,  kq_dim, dropout_rate=0.5):
        super(MyTemporalAttention, self).__init__()
        self.fc_q = nn.Linear(dim, kq_dim)
        self.q_drop = nn.Dropout(dropout_rate)
        self.fc_k = nn.Linear(dim, kq_dim)
        self.k_drop = nn.Dropout(dropout_rate)
        self.fc_v = nn.Linear(dim, int(sample_len//cnn_block_len))
        self.v_drop = nn.Dropout(dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)
        self.sample_len = sample_len

    def forward(self, x):
        residual = x
        # q = self.fc_q(x)  # (batch, seq_len, kq_dim)
        q = self.q_drop(self.fc_q(x))  # (batch, seq_len, kq_dim)

        # k = self.fc_k(x)  # (batch, seq_len, kq_dim)
        k = self.k_drop(self.fc_k(x))  # (batch, seq_len, kq_dim)

        # v = self.fc_v(x)  # (batch, seq_len, input_dim)
        v = self.v_drop(self.fc_v(x))  # (batch, seq_len, input_dim)

        # 缩放点积注意力
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.sample_len)  # (batch, seq_len, seq_len)
        attn_weights = self.softmax(attn_weights)
        attn_weights = self.dropout(attn_weights)

        # 计算输出
        out = torch.matmul(attn_weights, v)  # (batch, seq_len, input_dim)

        out = out + residual
        return out


class Attention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        self.key = nn.Linear(emb_size, emb_size, bias=True)
        self.value = nn.Linear(emb_size, emb_size, bias=True)
        self.query = nn.Linear(emb_size, emb_size, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

        attn = torch.matmul(q, k) * self.scale

        attn = nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.to_out(out)
        return out




class TANet(nn.Module):
    def __init__(self, sample_len=128, channels_num=64, se_cnn_num=10,  cnn1d_kernel_num=5, cnn_block_len=4,
                 sa_block_num=1, dropout=0.5, is_space_attention=True, is_temporal_attention=True):
        super(TANet, self).__init__()

        # Model hyperparameters
        self.sample_len = sample_len
        self.channels_num = channels_num
        self.cnn_kernel_num = cnn1d_kernel_num
        self.cnn_block_len = cnn_block_len
        self.sa_block_num = sa_block_num  # default to at least 1
        self.is_space_attention = is_space_attention
        self.is_temporal_attention = is_temporal_attention

        self.ln1 = nn.LayerNorm(channels_num)
        self.ln2 = nn.LayerNorm(channels_num)

        self.temporal_attention = Attention(emb_size=64, num_heads=8 ,dropout=0.1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dp =  nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(channels_num, 32),
            nn.ReLU(),
            nn.Linear(32, channels_num),
            nn.Dropout(dropout),
        )

        self.fc2 = nn.Linear(sample_len, channels_num)
        self.head = nn.Linear(channels_num, 2)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)


    def forward(self, x):
        x = x.permute(0, 2, 1)
        batch_size, sample_len, channels_num = x.size()

        resdiual = x

        x = self.temporal_attention(x)
        x = self.dp(x)
        x = x + resdiual

        x = self.ln1(x)
        resdiual = x
        x = self.fc(x)
        x = x + resdiual

        x = self.ln2(x)

        x = self.gap(x)
        # Fully connected layer
        x = x.view(batch_size, -1)
        x = self.fc2(x)
        x = self.head(x)

        return x


# 动态控制测试
def main():
    sample_len_options = [256]  # 可选输入长度
    for sample_len in sample_len_options:
        model = TANet(sample_len=sample_len, se_cnn_num=64, cnn1d_kernel_num=16, sa_block_num= 8, dropout=0.2)
        print(model)
        random_data = torch.rand((16, 64, sample_len))  # Dummy data
        output = model(random_data)
        print(f"Output shape for sample_len={sample_len}: {output.shape}")
        del model


if __name__ == '__main__':
    main()
