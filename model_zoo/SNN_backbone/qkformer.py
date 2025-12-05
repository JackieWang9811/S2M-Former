import torch
import torch.nn as nn
from spikingjelly.activation_based.neuron import LIFNode,ParametricLIFNode,surrogate


class Token_QK_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(alpha=5.0), backend='cupy') 

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(alpha=5.0), backend='cupy') 

        self.attn_lif = LIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(alpha=5.0), backend='cupy') 


    def forward(self, x):
        T, B, C, H, W = x.shape

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        q = torch.sum(q, dim = 3, keepdim = True)
        attn = self.attn_lif(q)
        x = torch.mul(attn, k)

        x = x.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        x = self.proj_lif(x)

        return x


class Spiking_Self_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = 0.125
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(alpha=5.0), backend='cupy') 

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(alpha=5.0), backend='cupy') 

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(alpha=5.0), backend='cupy') 
        self.attn_lif = LIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(alpha=5.0), backend='cupy') 

        self.qkv_mp = nn.MaxPool1d(4)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2,-1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        # x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T,B,C,W,H)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x)).reshape(T,B,C,W,H))

        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)
        self.mlp1_lif = LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(alpha=5.0), backend='cupy') 

        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)
        self.mlp2_lif = LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(alpha=5.0), backend='cupy')

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x).reshape(T, B, self.c_hidden, H, W)
        x = self.mlp1_lif(x)

        x = self.mlp2_conv(x.flatten(0, 1))
        x = self.mlp2_bn(x).reshape(T, B, C, H, W)
        x = self.mlp2_lif(x)

        return x


class TokenSpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.tssa = Token_QK_Attention(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features= dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):

        x = x + self.tssa(x)
        x = x + self.mlp(x)

        return x


class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.ssa = Spiking_Self_Attention(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features= dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.ssa(x)
        x = x + self.mlp(x)

        return x


class PatchEmbedInit(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif = LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(alpha=5.0), backend='cupy') 

        self.proj1_conv = nn.Conv2d(embed_dims // 2, embed_dims // 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims // 1)
        self.proj1_lif = LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(alpha=5.0), backend='cupy') 

        self.proj_res_conv = nn.Conv2d(embed_dims//2, embed_dims //1, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(alpha=5.0), backend='cupy') 



    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W)
        x = self.proj_lif(x).flatten(0, 1)

        x_feat = x
        x = self.proj1_conv(x)
        x = self.proj1_bn(x).reshape(T, B, -1, H, W)
        x = self.proj1_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H, W).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat # shortcut

        return x


class PatchEmbeddingStage(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256, downsampling=False):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.downsampling = downsampling
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj3_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.proj3_lif = LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(alpha=5.0), backend='cupy')

        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj4_lif = LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(alpha=5.0), backend='cupy') 
        if self.downsampling:
            self.proj_res_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        else:
            self.proj_res_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(alpha=5.0), backend='cupy') 

    def forward(self, x):
        T, B, C, H, W = x.shape
        # Downsampling + Res

        x = x.flatten(0, 1).contiguous()
        x_feat = x

        x = self.proj3_conv(x)
        x = self.proj3_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj3_lif(x).flatten(0, 1).contiguous()

        x = self.proj4_conv(x)
        x = self.proj4_bn(x)
        if self.downsampling:
            x = self.proj4_maxpool(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
            x = self.proj4_lif(x)

            x_feat = self.proj_res_conv(x_feat)
            x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H//2, W//2).contiguous()
            x_feat = self.proj_res_lif(x_feat)

            x = x + x_feat # shortcut
        else:
            x = x.reshape(T, B, -1, H, W).contiguous()
            x = self.proj4_lif(x)
            x_feat = self.proj_res_conv(x_feat)
            x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H, W).contiguous()
            x_feat = self.proj_res_lif(x_feat)

            x = x + x_feat  # shortcut

        return x


class QKFormer(nn.Module):
    def __init__(self,
                 img_size_h=32, img_size_w=32, patch_size=2, in_channels=5, num_classes=2,
                 embed_dims=64, num_heads= [8, 8, 8], mlp_ratios=4, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=6,  T=4,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed1 = PatchEmbedInit(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims // 4)

        stage1 = TokenSpikingTransformer( dim=embed_dims // 4, num_heads=num_heads[0], mlp_ratio=mlp_ratios)

        patch_embed2 = PatchEmbeddingStage(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims // 2,
                                    downsampling=True)

        stage2 = TokenSpikingTransformer(dim=embed_dims // 2, num_heads=num_heads[1], mlp_ratio=mlp_ratios)

        patch_embed3 = PatchEmbeddingStage(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims,
                                       downsampling=True)

        block = nn.ModuleList([SpikingTransformer(
            dim=embed_dims, num_heads=num_heads[2], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer)
            for j in range(depths - 2)])

        setattr(self, f"patch_embed1", patch_embed1)
        setattr(self, f"patch_embed2", patch_embed2)
        setattr(self, f"patch_embed3", patch_embed3)
        setattr(self, f"stage1", stage1)
        setattr(self, f"stage2", stage2)
        setattr(self, f"block", block)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        stage1 = getattr(self, f"stage1")
        patch_embed1 = getattr(self, f"patch_embed1")
        stage2 = getattr(self, f"stage2")
        patch_embed2 = getattr(self, f"patch_embed2")
        block = getattr(self, f"block")
        patch_embed3 = getattr(self, f"patch_embed3")

        x = patch_embed1(x)
        x = stage1(x)

        x = patch_embed2(x)
        x = stage2(x)

        x = patch_embed3(x)
        for blk in block:
            x = blk(x)

        return x.flatten(3).mean(3)

    def forward(self, x):
        if len(x.shape) == 4:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        elif len(x.shape) == 5:
            x = (x.squeeze(-1).unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head(x.mean(0))

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
    model = QKFormer().to(device)

    print(model)
    fre_tensor = torch.rand([32, 5, 32, 32]).to(device)  # (batch_size, csp_dim, sample_points)
    # 前向传播
    output = model(fre_tensor)
    # print(output)
    print("Output shape:", output.shape)
    param_m = count_parameters(model) / 1e6  # 除以百万单位换算系数
    print("Model size: {:.2f}M".format(param_m))  # Model size: 16.81M