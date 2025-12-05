import torch
import torch.nn as nn
from spikingjelly.activation_based.neuron import LIFNode,ParametricLIFNode, surrogate

class MS_SPS(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=4,
        in_channels=2,
        embed_dims=256,
        pooling_stat="1111",
        spike_mode="lif",
    ):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = (patch_size,patch_size)
        self.patch_size = patch_size
        self.pooling_stat = pooling_stat

        self.C = in_channels
        self.H, self.W = (
            self.image_size[0] // patch_size[0],
            self.image_size[1] // patch_size[1],
        )
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(
            in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        if spike_mode == "lif":
            self.proj_lif = LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(alpha=5.0), backend='cupy') 
        elif spike_mode == "plif":
            self.proj_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv1 = nn.Conv2d(
            embed_dims // 8,
            embed_dims // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        if spike_mode == "lif":
            self.proj_lif1 = LIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.proj_lif1 = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv2 = nn.Conv2d(
            embed_dims // 4,
            embed_dims // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        if spike_mode == "lif":
            self.proj_lif2 = LIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.proj_lif2 = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv3 = nn.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        if spike_mode == "lif":
            self.proj_lif3 = LIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.proj_lif3 = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool3 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.rpe_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        if spike_mode == "lif":
            self.rpe_lif = LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(alpha=5.0), backend='cupy') 
        elif spike_mode == "plif":
            self.rpe_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
    def forward(self, x):
        # T 是步数， B是batch size具体为什么是2也不太清楚
        T, B, _, H, W = x.shape #
        ratio = 1
        x = self.proj_conv(x.flatten(0, 1))  # have some fire value
        x = self.proj_bn(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif(x)

        x = x.flatten(0, 1).contiguous()
        # if self.pooling_stat[0] == "1":
        #     x = self.maxpool(x)
        #     ratio *= 2

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif1(x)

        x = x.flatten(0, 1).contiguous()
        # if self.pooling_stat[1] == "1":
        #     x = self.maxpool1(x)
        #     ratio *= 2

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif2(x)

        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[2] == "1":
            x = self.maxpool2(x)
            ratio *= 2

        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        if self.pooling_stat[3] == "1":
            x = self.maxpool3(x)
            ratio *= 2

        x_feat = x
        x = self.proj_lif3(x.reshape(T, B, -1, H // ratio, W // ratio).contiguous())

        x = x.flatten(0, 1).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x)
        x = (x + x_feat).reshape(T, B, -1, H // ratio, W // ratio).contiguous()

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        # x:(4,64,256,8,8) (H,W)=(2,2), hook=None
        return x, (H, W),

class MS_MLP_Conv(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        spike_mode="lif",
        layer=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        if spike_mode == "lif":
            self.fc1_lif = LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(alpha=5.0), backend='cupy') 
        elif spike_mode == "plif":
            self.fc1_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.fc2_conv = nn.Conv2d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm2d(out_features)
        if spike_mode == "lif":
            self.fc2_lif = LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(alpha=5.0), backend='cupy') 
        elif spike_mode == "plif":
            self.fc2_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.c_hidden = hidden_features
        self.c_output = out_features
        self.layer = layer

    def forward(self, x):
        T, B, C, H, W = x.shape
        identity = x
        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
        if self.res:
            x = identity + x
            identity = x
        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()
        x = x + identity
        return x


class MS_SSA_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.dvs = dvs
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.q_lif = LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(alpha=5.0), backend='cupy') 
        elif spike_mode == "plif":
            self.q_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.k_lif = LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(alpha=5.0), backend='cupy') 
        elif spike_mode == "plif":
            self.k_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.v_lif = LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(alpha=5.0), backend='cupy') 
        elif spike_mode == "plif":
            self.v_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        if spike_mode == "lif":
            self.attn_lif = LIFNode(
                tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.attn_lif = ParametricLIFNode(
                init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )
        self.talking_heads = nn.Conv1d(
            num_heads, num_heads, kernel_size=1, stride=1, bias=False
        )
        if spike_mode == "lif":
            self.talking_heads_lif = LIFNode(
                tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.talking_heads_lif = ParametricLIFNode(
                init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy")
        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(dim)

        if spike_mode == "lif":
            self.shortcut_lif = LIFNode(
                tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.shortcut_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy")
        self.mode = mode
        self.layer = layer

    def forward(self, x):
        T, B, C, H, W = x.shape
        identity = x
        N = H * W
        x = self.shortcut_lif(x)

        x_for_qkv = x.flatten(0, 1)
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()
        q_conv_out = self.q_lif(q_conv_out)

        q = (
            q_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, H, W).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        if self.dvs:
            k_conv_out = self.pool(k_conv_out)
        k = (
            k_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4) # ()
            .contiguous()
        )

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, H, W).contiguous()
        v_conv_out = self.v_lif(v_conv_out)


        v = (
            v_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )  # T B head N C//h

        kv = k.mul(v)

        if self.dvs:
            kv = self.pool(kv)
        kv = kv.sum(dim=-2, keepdim=True)
        kv = self.talking_heads_lif(kv)

        x = q.mul(kv)
        if self.dvs:
            x = self.pool(x)


        x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()
        x = (
            self.proj_bn(self.proj_conv(x.flatten(0, 1)))
            .reshape(T, B, C, H, W)
            .contiguous()
        )

        x = x + identity
        return x, v

class MS_Block_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,

        spike_mode="lif",
        layer=0,
    ):
        super().__init__()
        # SDSA
        self.attn = MS_SSA_Conv(
            dim,
            num_heads=num_heads,
            spike_mode=spike_mode,
            layer=layer,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        # MLP
        self.mlp = MS_MLP_Conv(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            spike_mode=spike_mode,
            layer=layer,
        )

    def forward(self, x):
        x_attn, attn= self.attn(x)
        x = self.mlp(x_attn)
        return x, attn



class SpikeDrivenTransformer(nn.Module):
    def __init__(
        self,
        img_size_h=32,
        img_size_w=32,
        patch_size=2,
        in_channels=5,
        num_classes=2,
        embed_dims=64,
        num_heads=8,
        mlp_ratios=4,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=6,
        T=4,
        spike_mode="lif",
        TET=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.T = T
        self.TET = TET
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        # 模型第一步Spiking Patch Splitting，S参照VIT的思路，使用attention 的QKV之前都应给输入进行patch化
        patch_embed = MS_SPS(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            spike_mode=spike_mode,
        )

        # 后面就是N个Spike-driven Encoder Moudle
        # ============> for i in range(n):
        # =================>X + Spike-Driven Self-Attention(SDSA)(X)
        # =======================> X + MLP(X)
        blocks = nn.ModuleList(
            [
                MS_Block_Conv(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    drop=drop_rate,
                    spike_mode=spike_mode,
                    layer=j,
                )
                for j in range(depths)
            ]
        )

        # 这是个用法
        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", blocks)

        # classification head
        if spike_mode in ["lif", "alif", "blif"]:
            self.head_lif = LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(alpha=5.0), backend='cupy') 
        elif spike_mode == "plif":
            self.head_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.head = (
            nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, hook=None):

        #反射属性, getattr(object, name) 从指定的对象中获取与给定名称对应的属性值。
        block = getattr(self, f"block") # 从 self中获取block
        patch_embed = getattr(self, f"patch_embed") # 从 self中获取patch_embed

                                               # size:(Time_Step, Batch_Size, C, H, W)
        x, _, = patch_embed(x) # x:(4, 64, 256, 8, 8)
        for blk in block:
            x, _, = blk(x)

        x = x.flatten(3).mean(3)
        return x

    def forward(self, x, hook=None):
        if len(x.shape) == 4:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        elif len(x.shape) == 5:
            x = (x.squeeze(-1).unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)

        # 主干网络()
        x  = self.forward_features(x, hook=hook)

        # LIF
        x = self.head_lif(x)

        # 分类头
        x = self.head(x)
        x = x.mean(0)


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
    args.csp_dim = 32
    args.fre_tokens = args.eeg_channel
    args.dataset = "DTUDataset"


    # 初始化模型
    model = SpikeDrivenTransformer().to(device)

    print(model)
    fre_tensor = torch.rand([32, 5, 32, 32]).to(device)  # (batch_size, csp_dim, sample_points)
    # 前向传播
    output = model(fre_tensor)
    # print(output)
    print("Output shape:", output.shape)
    param_m = count_parameters(model) / 1e6  # 除以百万单位换算系数
    print("Model size: {:.2f}M".format(param_m))  # Model size: 16.81M
