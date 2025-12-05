from .conv import Transpose, PointwiseConv1d, DepthwiseConv1d
from dotmap import DotMap
from torch.nn.init import xavier_uniform_
from spikingjelly.activation_based.neuron import LIFNode, ParametricLIFNode
from spikingjelly.activation_based import neuron, layer
import math
import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate
import os
import torch.nn.functional as F
from typing import Callable


class ChannelWisePLIFNodeHead(ParametricLIFNode):
    def __init__(self, channels: int, init_tau: float = 2.0, decay_input: bool = True,
                 v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='m', backend='torch',
                 store_v_seq: bool = False):
        super().__init__(init_tau=init_tau, decay_input=decay_input, v_threshold=v_threshold,
                         v_reset=v_reset, surrogate_function=surrogate_function,
                         detach_reset=detach_reset, step_mode=step_mode,
                         backend=backend, store_v_seq=store_v_seq)
        self.channels = channels

        # 使用 softmax 权重：初始化为对应 tau 的 log 反变换
        # init_w = -math.log(init_tau - 1.0)
        # self.w = nn.Parameter(torch.full((channels,), init_w))  # [C]
        self.w = nn.Parameter(torch.empty(channels), requires_grad=True)
        nn.init.normal_(self.w, mean=1.0, std=0.02)
        self.bias = nn.Parameter(torch.empty(channels), requires_grad=True)
        nn.init.zeros_(self.bias)

    def extra_repr(self):
        with torch.no_grad():
            tau = 1. / F.softmax(self.w, dim=0)
        return super().extra_repr() + f', channel_w=softmax, tau_shape={tau.shape}'

    def neuronal_charge(self, x: torch.Tensor):
        # 输入 x 的形状: [T, B, C]
        w_softmax = F.softmax(self.w, dim=0).view(1, 1, -1)  # 形状变为 [1, 1, C] 以广播到 [T, B, C]

        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v + (x - self.v) * w_softmax + self.bias.view(1, 1, -1)
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * w_softmax + self.bias.view(1, 1, -1)
        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v * (1. - w_softmax) + x + self.bias.view(1, 1, -1)
            else:
                self.v = self.v - (self.v - self.v_reset) * w_softmax + x + self.bias.view(1, 1, -1)

    def forward(self, x: torch.Tensor):
        # x shape: [T, B, C]
        spike = super().forward(x)
        return spike, self.v

class ChannelWisePLIFNode(ParametricLIFNode):
    def __init__(self, channels: int, init_tau: float = 2.0, decay_input: bool = True,
                 v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='m', backend='torch',
                 store_v_seq: bool = False):
        super().__init__(init_tau=init_tau, decay_input=decay_input, v_threshold=v_threshold,
                         v_reset=v_reset, surrogate_function=surrogate_function,
                         detach_reset=detach_reset, step_mode=step_mode,
                         backend=backend, store_v_seq=store_v_seq)
        self.channels = channels


        self.w = nn.Parameter(torch.empty(channels), requires_grad=True)
        nn.init.normal_(self.w, mean=1.0, std=0.02)

        self.bias = nn.Parameter(torch.empty(channels), requires_grad=True)
        nn.init.zeros_(self.bias)

    def extra_repr(self):
        with torch.no_grad():
            tau = 1. / F.softmax(self.w, dim=0)
        return super().extra_repr() + f', channel_w=softmax, tau_shape={tau.shape}'

    def neuronal_charge(self, x: torch.Tensor):
        # 输入 x 的形状: [T, B, N, C]
        w_softmax = F.softmax(self.w, dim=0).view(1, 1, 1, -1)  # 形状变为 [1, 1, 1, C] 以广播到 [T, B, C]

        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v + (x - self.v) * w_softmax + self.bias.view(1, 1, 1, -1)
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * w_softmax + self.bias.view(1, 1, 1, -1)
        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v * (1. - w_softmax) + x +self.bias.view(1, 1, 1, -1)
            else:
                self.v = self.v - (self.v - self.v_reset) * w_softmax + x + self.bias.view(1, 1, 1, -1)

    def forward(self, x: torch.Tensor):
        # x shape: [T, B, C]
        spike = super().forward(x)
        spike= spike.squeeze(1)
        return spike

class SGCM(nn.Module):
    """
    Spiking Gated Channel Mixer
    """
    def __init__(self, args, dim: int, hidden_features:int,  spike_mode) -> None:
        super(SGCM, self).__init__()
        self.args = args
        if spike_mode == "lif":

            self.lif1 = ChannelWisePLIFNode(channels= dim, init_tau=args.init_tau, v_threshold=args.v_threshold,
                            surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
                            step_mode='m', decay_input=True, store_v_seq=False, backend=args.backend)

            self.lif2 = LIFNode(tau=args.init_tau, v_threshold=args.v_threshold,
                            surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
                            step_mode='m', decay_input=False, store_v_seq=False, backend=args.backend)

            self.lif3 = LIFNode(tau=args.init_tau, v_threshold=args.v_threshold,
                            surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
                            step_mode='m', decay_input=False, store_v_seq=False, backend=args.backend)

            self.lif4 = LIFNode(tau=args.init_tau, v_threshold=args.v_threshold,
                            surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
                            step_mode='m', decay_input=False, store_v_seq=False, backend=args.backend)

            self.lif5 = LIFNode(tau=args.init_tau, v_threshold=args.v_threshold,
                            surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
                            step_mode='m', decay_input=False, store_v_seq=False, backend=args.backend)

            self.lif6 = LIFNode(tau=args.init_tau, v_threshold=args.v_threshold,
                            surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
                            step_mode='m', decay_input=False, store_v_seq=False, backend=args.backend)

            self.lif7 = LIFNode(tau=args.init_tau, v_threshold=args.v_threshold,
                            surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
                            step_mode='m', decay_input=False, store_v_seq=False, backend=args.backend)


        # 8 => 16
        self.fc1 = layer.Conv1d(dim, hidden_features, kernel_size=1, padding=0, stride=1, bias=args.bias, step_mode='m')

        # 16=>16
        self.head_conv = layer.Conv1d(hidden_features, hidden_features, 3, stride=1, padding=1, bias=False, step_mode='m')  #
        self.head_conv_bn = layer.BatchNorm1d(hidden_features, step_mode='m')

        self.dw = layer.SeqToANNContainer(DepthwiseConv1d(hidden_features//2, hidden_features//2, 3, stride=1, padding=1, bias=True))
        self.dw_bn = layer.BatchNorm1d(hidden_features//2, step_mode='m')

        self.dw2 = layer.SeqToANNContainer(DepthwiseConv1d(hidden_features//2, hidden_features//2, 3, stride=1, padding=1, bias=True))
        self.dw2_bn = layer.BatchNorm1d(hidden_features//2, step_mode='m')

        self.last_conv = layer.Conv1d(hidden_features//2, hidden_features//2, 3, stride=1, padding=1, bias=False, step_mode='m')
        self.last_conv_bn = layer.BatchNorm1d(hidden_features//2, step_mode='m')

        # 8=> 8
        self.fc2 = layer.Conv1d(hidden_features//2, dim, kernel_size=1, padding=0, stride=1, bias=args.bias, step_mode='m')

    def forward(self, inputs):

        inputs_spike = self.lif1(inputs.permute(0,1,3,2)).permute(0,1,3,2)
        x = self.fc1(inputs_spike)
        x = self.head_conv_bn(self.head_conv(self.lif2(x)))
        x_in, v = x.chunk(2, dim=-2)
        x = self.dw_bn(self.dw(self.lif3(x_in)))
        v = self.dw2_bn(self.dw2(self.lif4(v)))
        x = self.lif5(x)
        v = self.lif6(v)
        x = torch.sum(x, dim=2, keepdim=True)
        v = torch.mul(v, x)
        v = self.lif7(self.last_conv_bn(self.last_conv(v)))
        v = self.fc2(v)
        return v



class MS_SSA(nn.Module):
    def __init__(self, dim, args, num_heads=8, kernel=5, spike_mode="lif", token=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.args = args


        self.attn_scale = 1 / math.sqrt(token//num_heads)

        if spike_mode == "lif":
            self.head_lif = ChannelWisePLIFNode(channels=dim, init_tau=args.init_tau, v_threshold=args.v_threshold,
                                 surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
                                 step_mode='m', decay_input=True, store_v_seq=False, backend=args.backend)
            self.head_lif2 = LIFNode(tau=args.init_tau, v_threshold=args.v_threshold,
                                 surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
                                 step_mode='m', decay_input=False, store_v_seq=False, backend=args.backend)


        self.dw = layer.SeqToANNContainer(DepthwiseConv1d(dim, dim, kernel, stride=1, padding=(kernel - 1) // 2, bias=args.use_dw_bias))
        self.dw_bn = layer.BatchNorm1d(dim, step_mode='m')
        self.q_layer = layer.SeqToANNContainer(DepthwiseConv1d(dim, dim, 3, stride=1, padding=(3 - 1) // 2, bias=args.use_dw_bias))
        self.q_bn = layer.BatchNorm1d(dim, step_mode='m')

        if spike_mode == "lif":
            self.q_lif = LIFNode(tau=args.init_tau, v_threshold=args.v_threshold,
            surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
            step_mode='m', decay_input=False, store_v_seq=False, backend=args.backend)


        self.k_layer = layer.SeqToANNContainer(DepthwiseConv1d(dim, dim, 3, stride=1, padding=(3 - 1) // 2, bias=args.use_dw_bias))
        self.k_bn = layer.BatchNorm1d(dim, step_mode='m')

        if spike_mode == "lif":
            self.k_lif = LIFNode(tau=args.init_tau, v_threshold=args.v_threshold,
            surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
            step_mode='m', decay_input=False, store_v_seq=False, backend=args.backend)


        self.v_layer = layer.SeqToANNContainer(DepthwiseConv1d(dim, dim, 3, stride=1, padding=(3 - 1) // 2, bias=args.use_dw_bias))
        self.v_bn = layer.BatchNorm1d(dim, step_mode='m')

        if spike_mode == "lif":
            self.v_lif = LIFNode(tau=args.init_tau, v_threshold=args.v_threshold,
            surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
            step_mode='m', decay_input=False, store_v_seq=False,backend=args.backend)


        if spike_mode == "lif":
            self.attn_lif = LIFNode(tau=args.init_tau, v_threshold=args.v_threshold,
            surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
            step_mode='m', decay_input=False, store_v_seq=False,backend=args.backend)


        self.proj_conv = layer.SeqToANNContainer(DepthwiseConv1d(dim, dim, 3, stride=1, padding=(3 - 1) // 2, bias=args.use_dw_bias))
        self.proj_bn = layer.BatchNorm1d(dim, step_mode='m')


        if spike_mode == "lif":
            self.proj_lif = LIFNode(tau=args.init_tau, v_threshold=args.v_threshold,
            surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
            step_mode='m', decay_input=False, store_v_seq=False,backend=args.backend)


    def forward(self, x):
        # T => N ; N =>D
        Ts, B, D, N = x.shape
        # x_for_qkv = x.flatten(0, 1)
        x = self.head_lif(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        x = self.dw(x)
        x = self.dw_bn(x)
        x = self.head_lif2(x)

        # q = self.q_lif(x)
        q_conv_out = self.q_layer(x) # (Ts, B, T, N) => (Ts, B, N, T)
        q_conv_out = self.q_bn(q_conv_out).reshape(Ts, B, self.num_heads, N// self.num_heads, self.dim) # (Ts, B, N, T) => (Ts, B, T, N) => (Ts, B, H, T, N//H)
        q_conv_out = self.q_lif(q_conv_out)

        # reshape # (Ts, B, N, T) => (Ts*B, N, T)
        q_conv_out = q_conv_out.reshape(Ts * B, self.num_heads, N// self.num_heads, self.dim)
        q_conv_out = self.rotary_emb.rotate_queries_or_keys(q_conv_out).reshape(Ts, B, self.num_heads, N // self.num_heads, self.dim)  # (Ts*B, N, T) => (Ts, B, H, T, N//H)

        # k = self.k_lif(x)
        k_conv_out = self.k_layer(x)
        k_conv_out = self.k_bn(k_conv_out).reshape(Ts, B, self.num_heads, N // self.num_heads, self.dim)
        k_conv_out = self.k_lif(k_conv_out)

        # reshape # (Ts, B, N, T) => (Ts*B, N, T)
        k_conv_out = k_conv_out.reshape(Ts * B, self.num_heads, N// self.num_heads, self.dim)
        k_conv_out = self.rotary_emb.rotate_queries_or_keys(k_conv_out).reshape(Ts, B, self.num_heads, N // self.num_heads, self.dim)  # (Ts*B, N, T) => (Ts, B, H, T, N//H)


        # v = self.v_lif(x)
        v_conv_out = self.v_layer(x)
        v_conv_out = self.v_bn(v_conv_out).reshape(Ts, B, self.num_heads, N// self.num_heads, self.dim )
        v_conv_out = self.v_lif(v_conv_out)

        # Attention mechanism
        # attn = self.scaled_dot_product_attention(q_conv_out, k_conv_out, v_conv_out)
        attn = self.scaled_dot_product_attention_kv(q_conv_out, k_conv_out, v_conv_out)
        attn = attn.reshape(Ts, B, D, N)
        attn = self.attn_lif(attn)

        output = self.proj_conv(attn)
        output = self.proj_bn(output)

        return output


    def scaled_dot_product_attention(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.attn_scale # (B,H,T,T)
        # attn = (q @ k.transpose(-2, -1))
        x = attn @ v  # (Ts, B,H,T,T) * (Ts, B,H,T,D) => (Ts, B, H, T, D)
        x = x.transpose(2, 3) # (Ts, B, T, H, D)
        return x

    def scaled_dot_product_attention_kv(self, q, k, v):
        attn = (k.transpose(-2, -1) @ v) * self.attn_scale #
        # attn = (q @ k.transpose(-2, -1))
        x = q @ attn  # (Ts, B, H, T ,T) * (Ts, B,H,T,D) => (Ts, B, H, T, D)
        x = x.transpose(2, 3)
        return x




class ConvModule(nn.Module):

    """
    Multi-scaled Convolutional Module
    """
    def __init__(
            self,
            args,
            in_channels: int,
            spike_mode="lif",
    ) -> None:
        super(ConvModule, self).__init__()
        self.args = args
        self.expansion_factor = 3
        self.bn1 = layer.BatchNorm1d(in_channels*self.expansion_factor, step_mode='m')
        self.bn2 = layer.BatchNorm1d(in_channels, step_mode='m')
        self.bn3 = layer.BatchNorm1d(in_channels, step_mode='m')
        self.bn4 = layer.BatchNorm1d(in_channels, step_mode='m')

        self.in_channels = in_channels

        if spike_mode == "lif":
            self.lif_head = ChannelWisePLIFNode(channels= in_channels, init_tau=args.init_tau, v_threshold=args.v_threshold,
                            surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
                            step_mode='m', decay_input=True, store_v_seq=False, backend=args.backend)

            self.lif1 = LIFNode(tau=args.init_tau, v_threshold=args.v_threshold,
                            surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
                            step_mode='m', decay_input=False, store_v_seq=False, backend=args.backend)
            self.lif2 = LIFNode(tau=args.init_tau, v_threshold=args.v_threshold,
                           surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
                           step_mode='m', decay_input=False, store_v_seq=False, backend=args.backend)
            self.lif3 = LIFNode(tau=args.init_tau, v_threshold=args.v_threshold,
                           surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
                           step_mode='m', decay_input=False, store_v_seq=False, backend=args.backend)
            self.lif_last = LIFNode(tau=args.init_tau, v_threshold=args.v_threshold,
                            surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
                            step_mode='m', decay_input=False, store_v_seq=False, backend=args.backend)


        self.dw1 = layer.SeqToANNContainer(DepthwiseConv1d(in_channels, in_channels, 1, stride=1, padding=(1 - 1) // 2, bias=args.use_dw_bias))
        self.dw2 = layer.SeqToANNContainer(DepthwiseConv1d(in_channels, in_channels, 3, stride=1, padding=(3 - 1) // 2, bias=args.use_dw_bias))
        self.dw3 = layer.SeqToANNContainer(DepthwiseConv1d(in_channels, in_channels, 5, stride=1, padding=(5 - 1) // 2, bias=args.use_dw_bias))
        self.pw1 = layer.SeqToANNContainer(PointwiseConv1d(in_channels, in_channels * self.expansion_factor, stride=1, padding=0, bias=args.bias))

    def channel_shuffle(self, x, groups):
        time_steps, batchsize, num_channels, tokens = x.data.size()
        channels_per_group = num_channels // groups
        # reshape
        x = x.view(time_steps, batchsize, groups,
                   channels_per_group, tokens)
        x = torch.transpose(x, 2, 3).contiguous()
        # flatten
        x = x.view(time_steps, batchsize, -1, tokens)
        return x

    def forward(self, inputs):

        x = inputs
        x = self.lif_head(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x = self.pw1(x)
        x = self.bn1(x)

        x1, x2, x3 = torch.split(x, x.size(2) // self.expansion_factor, dim=2)

        x1 = self.lif1(x1)
        x1 = self.dw1(x1)
        x1 = self.bn2(x1)

        x2 = self.lif2(x2)
        x2 = self.dw2(x2)
        x2 = self.bn3(x2)

        x3 = self.lif3(x3)
        x3 = self.dw3(x3)
        x3 = self.bn4(x3)

        out = x1 + x2 + x3
        out = self.channel_shuffle(out, 2)

        return out


class MPTM(nn.Module):
    def __init__(self, args, lif_module: nn.Module, alpha: float):
        """
        Membrane Potential-aware Token Mixer
        """
        super().__init__()
        self.lif_module = lif_module
        self.alpha = alpha
        if args.length == 0.1:
            self.aggregator = layer.MaxPool1d(kernel_size=1, stride=1, padding=0, step_mode='m')
        else:
            self.aggregator = layer.MaxPool1d(kernel_size=3, stride=2, padding=1, step_mode='m')

    def forward(
            self,
            x: torch.Tensor,  # [T, B, D, num_tokens] - 要处理的特征
            global_fusion: torch.Tensor,  # [T, B, D] - 全局融合信息 (均值池化之后的)
            base_x: torch.Tensor,  # [T, B, D, num_tokens] - 做残差/最终拼接时用
            total_tokens: int  # 要在 token 维度上混合的总长度
    ) -> torch.Tensor:
        """
        具体流程:
          1) x_lif = self.lif_module(x)
          2) 利用 alpha 拆分 fuse_len/refine_len
          3) 把 global_fusion 以及 x_lif.mean(-1) 在 token 维度上拼接
          4) 与 x_lif 逐元素相乘并和 x 做残差
          5) 与 base_x 在 token 维度上拼接
        """

        # Step 1) LIF 激活
        x_lif = self.lif_module(x)  # [T, B, D, num_tokens]

        # Step 2) 按照 alpha 划分 fuse_len / refine_len
        fuse_len = int(total_tokens * self.alpha)
        refine_len = int(total_tokens - fuse_len)

        # Step 3) 借助 global_fusion 和 x_lif.mean(-1) 进行拼接
        #         global_fusion: [T, B, D]
        #         => unsqueeze(-1) -> [T, B, D, 1]
        #         => repeat -> [T, B, D, fuse_len]
        fusion_part = global_fusion.unsqueeze(-1).repeat(1, 1, 1, fuse_len)

        # x_lif.mean(-1): [T, B, D]
        # => unsqueeze(-1) -> [T, B, D, 1]
        # => repeat -> [T, B, D, refine_len]
        refine_part = x_lif.mean(-1, keepdim=True).repeat(1, 1, 1, refine_len)

        # 拼接 => [T, B, D, total_tokens]
        fusion_tokens = torch.cat([fusion_part, refine_part], dim=-1)

        # Step 4) 逐元素相乘 + 残差
        out = x_lif * fusion_tokens  # [T, B, D, total_tokens]
        out = out + x  # 与原始特征做残差

        # Step 5) 与 base_x 拼接
        # out: [T, B, D, total_tokens] ，base_x 同形状 => 在 token 维度拼
        out = torch.cat((out, self.aggregator(base_x)), dim=-1)

        return out


class Branch_Block(nn.Module):
    def __init__(
        self,
        args,
        dim,
        num_heads,
        spike_mode="lif",
        alpha=0.5
    ):
        super().__init__()
        self.args = args
        self.alpha = alpha  # 保存到 self

        self.tem_attn = MS_SSA(dim, args, num_heads=num_heads, spike_mode=spike_mode, kernel=3, token=args.tem_tokens)

        self.fre_attn = MS_SSA(dim, args, num_heads=num_heads, spike_mode=spike_mode, kernel=3, token=args.fre_tokens)

        self.tem_conv = ConvModule(args, in_channels=dim, spike_mode=args.spike_mode)

        self.fre_conv = ConvModule(args, in_channels=dim, spike_mode=args.spike_mode)

        self.mlp1 = SGCM(args, dim, hidden_features=args.hidden_dims*2, spike_mode=spike_mode)

        self.mlp2 = SGCM(args, dim, hidden_features=args.hidden_dims*2, spike_mode=spike_mode)

        self.mlp1_lif = ChannelWisePLIFNode(channels=dim, init_tau=args.init_tau, v_threshold=args.v_threshold,
                        surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
                        step_mode='m', decay_input=True, store_v_seq=False, backend=args.backend)

        self.mlp2_lif = ChannelWisePLIFNode(channels=dim, init_tau=args.init_tau, v_threshold=args.v_threshold,
                        surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
                        step_mode='m', decay_input=True, store_v_seq=False, backend=args.backend)

        self.tem_attn_lif = LIFNode(tau=args.init_tau, v_threshold=args.v_threshold,
                        surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
                        step_mode='m', decay_input=False, store_v_seq=False, backend=args.backend)
        self.fre_attn_lif = LIFNode(tau=args.init_tau, v_threshold=args.v_threshold,
                       surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
                       step_mode='m', decay_input=False, store_v_seq=False, backend=args.backend)

        self.tem_conv_lif = LIFNode(tau=args.init_tau, v_threshold=args.v_threshold,
                        surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
                        step_mode='m', decay_input=False, store_v_seq=False, backend=args.backend)

        self.fre_conv_lif = LIFNode(tau=args.init_tau, v_threshold=args.v_threshold,
                       surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
                       step_mode='m', decay_input=False, store_v_seq=False, backend=args.backend)

        # 用于第一阶段 (Attn后) 的 tem/fre
        self.tem_attn_mixer = MPTM(args, self.tem_attn_lif, alpha=self.alpha)
        self.fre_attn_mixer = MPTM(args, self.fre_attn_lif, alpha=self.alpha)

        # 用于第二阶段 (Conv后) 的 tem/fre
        self.tem_conv_mixer = MPTM(args, self.tem_conv_lif, alpha=self.alpha)
        self.fre_conv_mixer = MPTM(args, self.fre_conv_lif, alpha=self.alpha)

    def forward(self, tem, fre):

        # tem_res = tem
        T, B, tem_D, tem_tokens = tem.shape
        # fre_res = fre
        T, B, fre_D, fre_tokens = fre.shape

        # ------------------Attn------------------
        tem_attn = self.tem_attn(tem) + tem
        fre_attn = self.fre_attn(fre) + fre

        fusion = torch.cat((tem_attn, fre_attn), dim=-1)
        fusion = self.mlp1(fusion)

        # Global Average Pooling
        fusion = self.mlp1_lif(fusion.permute(0,1,3,2)).permute(0,1,3,2).mean(-1)  # [T, B, dim]

        tem_out = self.tem_attn_mixer(
            x=tem_attn,
            global_fusion=fusion,
            base_x=tem_attn,
            total_tokens=tem_tokens
        )

        fre_out = self.fre_attn_mixer(
            x=fre_attn,
            global_fusion=fusion,
            base_x=fre_attn,
            total_tokens=fre_tokens
        )

        # ------------------Conv------------------
        tem_conv = self.tem_conv(tem_out) + tem_out
        fre_conv = self.fre_conv(fre_out) + fre_out

        fusion1 = torch.cat((tem_conv, fre_conv), dim=-1)
        fusion1 = self.mlp2(fusion1)

        # Average Pooling (Firing Rate Calculation)
        fusion1 = self.mlp2_lif(fusion1.permute(0,1,3,2)).permute(0,1,3,2).mean(-1)

        if self.args.length ==0.1:
            tem_out = self.tem_conv_mixer(
                x=tem_conv,
                global_fusion=fusion1,
                base_x=tem_conv,
                total_tokens=tem_tokens*2
            )

            fre_out = self.fre_conv_mixer(
                x=fre_conv,
                global_fusion=fusion1,
                base_x=fre_conv,
                total_tokens=fre_tokens*2
            )
        else:
            tem_out = self.tem_conv_mixer(
                x=tem_conv,
                global_fusion=fusion1,
                base_x=tem_conv,
                total_tokens=tem_tokens * 1.5
            )

            fre_out = self.fre_conv_mixer(
                x=fre_conv,
                global_fusion=fusion1,
                base_x=fre_conv,
                total_tokens=fre_tokens * 1.5
            )
        return tem_out, fre_out


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Reshape x to combine the batch and time steps
        t, b, n, d = x.size()  # (4, 32, 16, 256)
        x = x.view(b * t, n, d)  # Shape: (4 * 32, 16, 256)

        # Apply positional encoding
        x = x + self.pe[:, :x.size(1)]

        # Reshape back to original dimensions
        x = x.view(t, b, n, d)  # Shape: (4, 32, 16, 256)

        return x



class SPS(nn.Module):
    def __init__(self, config, in_channels=2, embed_dims=16):
        super().__init__()

        self.proj_conv = layer.Conv2d(in_channels, embed_dims*4, kernel_size=3, stride=1, padding=2, dilation=2, bias=False, step_mode='m')
        self.proj_bn = layer.BatchNorm2d(embed_dims*4, step_mode='m')
        if config.spike_mode == "lif":
            self.proj_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
                                   surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                                   step_mode='m', decay_input=False, store_v_seq=False, backend=config.backend)


        self.proj_conv1 = layer.Conv2d(embed_dims*4, embed_dims*2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False, step_mode='m')
        self.proj_bn1 = layer.BatchNorm2d(embed_dims*2, step_mode='m')
        if config.spike_mode == "lif":
            self.proj_lif1 = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
                                   surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                                   step_mode='m', decay_input=False, store_v_seq=False, backend=config.backend)


        self.proj_conv2 = layer.Conv2d(embed_dims * 2, embed_dims, kernel_size=3, stride=1, padding=2, dilation=2, bias=False, step_mode='m')
        self.proj_bn2 = layer.BatchNorm2d(embed_dims, step_mode='m')
        if config.spike_mode == "lif":
            self.proj_lif2 = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
                                   surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                                   step_mode='m', decay_input=False, store_v_seq=False, backend=config.backend)


        self.proj_conv3 = layer.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=2, dilation=2, bias=False, step_mode='m')
        self.proj_bn3 = layer.BatchNorm2d(embed_dims,step_mode='m')
        if config.spike_mode == "lif":
            self.proj_lif3 = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
                                   surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                                   step_mode='m', decay_input=False, store_v_seq=False, backend=config.backend)

        self.maxpool1 = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False,step_mode='m')

        self.rpe_conv = layer.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1, padding=0, bias=False, step_mode='m')
        self.rpe_bn = layer.BatchNorm2d(embed_dims,step_mode='m')
        if config.spike_mode == "lif":
            self.rpe_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
                                   surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                                   step_mode='m', decay_input=False, store_v_seq=False, backend=config.backend)


    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_conv(x) # have some fire value
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x).contiguous()

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif1(x).contiguous()

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif2(x).contiguous()
        # x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif3(x).contiguous()
        x = self.maxpool1(x)

        x_feat = x.reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.rpe_lif(x)
        x = x + x_feat

        x = x.flatten(-2) # T,B,C,N
        return x



class TokenEmbedding(nn.Module):
    def __init__(self, config, c_in, d_model):
        super().__init__()
        self.config = config
        self.proj_conv = layer.Conv2d(1, d_model*2, kernel_size=(1, 8), stride=1, padding='same', bias=True, step_mode='m') # config.bias
        self.proj_bn = layer.BatchNorm2d(d_model*2, step_mode='m')
        if config.spike_mode == "lif":
            self.proj_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
                                   surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                                   step_mode='m', decay_input=False, store_v_seq=False, backend=config.backend)


        self.proj0_conv = layer.Conv2d(d_model*2, d_model*4, kernel_size=(1, 16), stride=1, padding='same', bias=True, step_mode='m') # config.bias
        self.proj0_bn = layer.BatchNorm2d(d_model*4, step_mode='m')
        if config.spike_mode == "lif":
            self.proj0_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
                                   surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                                   step_mode='m', decay_input=False, store_v_seq=False, backend=config.backend)


        self.proj1_conv = layer.Conv2d(d_model*4, d_model, kernel_size=(c_in, 1), stride=1, padding=0, bias=config.bias, step_mode='m')
        self.proj1_bn = layer.BatchNorm2d(d_model, step_mode='m')
        if config.spike_mode == "lif":
            self.proj1_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
                                   surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                                   step_mode='m', decay_input=False, store_v_seq=False, backend=config.backend)


        self.proj_res_conv = layer.Conv2d(d_model*4, d_model, kernel_size=(c_in, 1), stride=1, padding=0, bias=config.bias, step_mode='m')
        self.proj_res_bn = layer.BatchNorm2d(d_model, step_mode='m')
        if config.spike_mode == "lif":
            self.proj_res_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
                                   surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                                   step_mode='m', decay_input=False, store_v_seq=False, backend=config.backend)



    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.proj_conv(x)
        x = self.proj_bn(x)
        x = self.proj_lif(x)

        #
        x = self.proj0_conv(x)
        x = self.proj0_bn(x)
        x = self.proj0_lif(x)

        x_feat = x
        x = self.proj1_conv(x)
        x = self.proj1_bn(x)
        x = self.proj1_lif(x).squeeze(3)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat)
        x_feat = self.proj_res_lif(x_feat).squeeze(3)

        x = x + x_feat # shortcut
        return x



class BranchModule(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.blocks = nn.ModuleList(
            [
                Branch_Block(
                    args=self.args,
                    dim=self.args.embed_dim,
                    num_heads=self.args.num_heads,
                    spike_mode=self.args.spike_mode,
                )
                for _ in range(self.args.depths)
            ]
        )

    def forward(self, tem, fre):

        for module in self.blocks:
            tem, fre = module(tem, fre)
        return tem, fre

class SpikingBranchformer(nn.Module):
    def __init__(self, args):
        super(SpikingBranchformer, self).__init__()
        self.args = args
        self.tem_token_embeddings = TokenEmbedding(config=args, c_in=args.csp_comp, d_model=args.embed_dim)

        self.fre_token_embeddings = SPS(config=args, in_channels=args.in_channels, embed_dims=args.embed_dim)

        self.tem_pos_embedding = PositionalEmbedding(args.embed_dim)
        self.fre_pos_embedding = PositionalEmbedding(args.embed_dim)


        self.branchformer = BranchModule(args)
        self.head = layer.Linear(args.embed_dim, args.n_outputs, step_mode='m')

        self.head_lif_last = ChannelWisePLIFNodeHead(channels=args.embed_dim,  init_tau=args.init_tau, v_threshold=1e9, #  v_threshold=1e9， args.v_threshold
                               surrogate_function=args.surrogate_function, detach_reset=args.detach_reset,
                               step_mode='m', decay_input=True, store_v_seq=False, backend=args.backend)


        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, tem, fre):

        tem = tem.repeat(self.args.ts, 1, 1, 1)
        fre = fre.repeat(self.args.ts, 1, 1, 1, 1)

        tem_embed = self.tem_token_embeddings(tem)
        fre_embed = self.fre_token_embeddings(fre)

        T, B, tem_D, tem_tokens = tem_embed.shape
        T, B, fre_D, fre_tokens = fre_embed.shape


        tem_bran_output, fre_bran_output = self.branchformer(tem_embed, fre_embed)


        branch_output = torch.cat((tem_bran_output, fre_bran_output), dim=-1)
        branch_output = branch_output.mean(-1)

        output ,v = self.head_lif_last(branch_output)
        v = self.head(v.squeeze(0))

        return v


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':

    # set_random_seeds(200)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # 设置参数
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = DotMap()
    length = 0.1
    args.length = length
    args.overlap = 0.5
    args.batch_size = 32
    args.max_epoch = 500
    args.patience = 35
    args.log_interval = 20
    args.image_size = 32
    args.eeg_channel = 64
    args.csp_comp = 64
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

    ### SNN ###
    args.ts = 4
    args.init_tau = 2.0
    args.v_threshold = 1.0
    args.surrogate_function = surrogate.ATan(alpha=5.0)
    # args.surrogate_function = surrogate.Sigmoid(alpha=4.0)
    args.detach_reset = True
    args.backend = 'torch'
    args.spike_mode = "lif"
    # args.dim = 8
    args.embed_dim = 8 # 模态投影维度
    args.hidden_dims = 16  # 单模态MLP隐藏层维度  # 128 , 512
    args.token_dim = 64
    args.channel_dim = 32

    args.kernel_size = 15  # 卷积核大小  # 31, 15
    args.num_heads = 4  # 注意力头数量
    args.depths = 1  # Transformer 深度
    args.n_outputs = 2  # 输出维度 (假设任务是 10 分类)
    args.dropout_l = 0.0  # dropout 概率
    args.use_dp = True  # 是否使用 dropout
    args.use_dw_bias = True  # 深度卷积是否使用 bias
    args.in_channels = 5
    args.image_size = 32
    args.bias = True
    args.patch_size = 2
    args.total_tokens = math.ceil(128 * length) + (args.image_size // args.patch_size) **2
    args.tem_tokens = math.ceil(128 * length)
    args.fre_tokens = (args.image_size // args.patch_size) **2
    args.dataset = "DTUDataset"


    # 初始化模型
    model = SpikingBranchformer(args).to(device)
    param_m = count_parameters(model) / 1e6
    print(count_parameters(model))
    print(model)
    # 创建输入张量，形状为 (time_step, batch_size, csp_dim, sample_points)
    # 此处将 batch_size = 32, csp_dim = 64, sample_points = 256
    tem_tensor = torch.rand((32, 64, 12)).to(device)  # (batch_size, csp_dim, sample_points)
    fre_tensor = torch.rand([32, 5, 32, 32]).to(device)  # (batch_size, csp_dim, sample_points)
    # 前向传播
    output = model(tem_tensor, fre_tensor)
    print(output)
    print("Output shape:", output.shape)
    print("Model size: {:.2f}M".format(param_m))  # Model size: 0.05M

