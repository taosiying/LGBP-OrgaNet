import torch
import torch.nn as nn
import torchvision
from timm.models.layers import trunc_normal_
import numpy as np
from utils import *
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):
    def __init__(self, dim, factor, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim * factor),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, embed_dim=96,
                 depths=[2, 2, 6], num_heads=[3, 6, 12],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, **kwargs):

        super().__init__()

        patches_resolution = [img_size // patch_size, img_size // patch_size]
        num_patches = patches_resolution[0] * patches_resolution[1]

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=None)
            self.layers.append(layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


def reshape(x):
    x = torch.transpose(x, 1, 2)
    B, C, new_HW = x.shape
    x = x.view(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
    return x


class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, c2, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = c1 // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=c1, out_features=mid_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=mid_channel, out_features=c2)
        )
        self.act = nn.Sigmoid()
        # self.act=nn.SiLU()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        return self.act(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.act(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, c1):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(c1, c1)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class CorrelationBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dot_conv = nn.Conv2d(ch, ch, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(ch * 2)
        self.conv1 = nn.Conv2d(ch * 2, ch, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(int(ch))
        self.channel_attention = ChannelAttentionModule(ch * 2, ch)
        self.spatial_attention = SpatialAttentionModule()
        self.cbam = CBAM(ch)

    def forward(self, c, sw):


        c = self.cbam(reshape(c))
        sw = self.cbam(reshape(sw))
        residual = sw
        x = torch.cat([c, sw], dim=1)
        x1 = self.channel_attention(x)
        x = self.conv1(x) * x1
        x = x + residual
        x = Rearrange('b c h w -> b (h w) c')(x)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        # self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1, bias=False)
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True))
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        # x1 =self.conv1(x)
        # return x1 + self.conv2(x1)
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


def zero_padding(p, step=4):
    B, N, C = p.shape
    H = int(np.sqrt(N))
    p = p.permute(0, 2, 1).view(B, C, H, H)
    padded_matrix = torch.zeros(B, C, step * H, step * H, dtype=torch.float32).to('cuda')
    padded_matrix[:, :, ::step, ::step] = p
    padded_matrix = Rearrange('b c h w -> b (h w) c')(padded_matrix)
    return padded_matrix


def interp_padding(p, step=4):
    B, N, C = p.shape
    H = int(np.sqrt(N))
    p = p.permute(0, 2, 1).view(B, C, H, H)
    padded_matrix = F.interpolate(p, scale_factor=step, mode='bilinear', align_corners=True)
    padded_matrix = Rearrange('b c h w -> b (h w) c')(padded_matrix)
    return padded_matrix


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, h, c = x.size()
        h = int(np.sqrt(h))
        w = h
        x = Rearrange('b (h w) c -> b c h w', h=h, w=h)(x)

        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        x = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        x = Rearrange('b c h w -> b (h w) c')(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, h, C = x.shape
        width = int(np.sqrt(h))
        height = width
        x = Rearrange('b (h w) c -> b c h w', h=height, w=width)(x)
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, width * height)
        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=-1)
        value = self.value(x).view(batch_size, -1, width * height)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        out = Rearrange('b c h w -> b (h w) c')(out)
        return out


class CrossFusion(nn.Module):
    def __init__(self, dim=(96, 384), num_heads=3):
        super().__init__()
        self.dim = dim
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.downsample = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pos_embed_l = nn.Parameter(torch.zeros(1, 197, dim[0]))
        trunc_normal_(self.pos_embed_l, std=.02)
        self.pos_embed_s = nn.Parameter(torch.zeros(1, 197, dim[1]))
        trunc_normal_(self.pos_embed_s, std=.02)
        self.num_heads = num_heads
        self.norm_l = nn.LayerNorm(dim[0])
        self.norm_s = nn.LayerNorm(dim[1])
        self.projs = nn.ModuleList()
        for d in range(2):
            tmp = [nn.LayerNorm(dim[d]), nn.GELU(), nn.Linear(dim[d], dim[1 - d]), nn.LayerNorm(dim[1 - d])]
            self.projs.append(nn.Sequential(*tmp))
        self.average = nn.AdaptiveAvgPool1d(1)
        self.wls1 = nn.Linear(dim[1] + dim[0], dim[1])
        self.gelu = nn.GELU()
        self.wls2 = nn.Linear(dim[1], dim[1])
        self.wq = nn.Linear(dim[1], dim[1])
        self.wk = nn.Linear(dim[1], dim[1])
        self.wv = nn.Linear(dim[1], dim[1])

        # self.wvls = nn.Linear(2*dim[0],dim[0])
        self.scale_l = (dim[1] // self.num_heads) ** -0.5
        self.fc_out = nn.Linear(dim[1], dim[1])

    def forward(self, pl, ps):
        # torch.save(pl, 'results/hiformer-s/test/pl.pt')
        batch_size, h, C = pl.shape
        width = int(np.sqrt(h))
        height = width
        pl1 = Rearrange('b (h w) c->b c h w', h=height, w=width)(pl)
        pl1 = self.downsample(pl1)
        pl1 = Rearrange('b c h w -> b (h w) c')(pl1)
        pl_pool = self.average(self.norm_l(pl).transpose(1, 2))
        pl_pool = Rearrange('b c 1 -> b 1 c')(pl_pool)
        pl_cls = self.projs[0](pl_pool)
        ps_pool = self.average(self.norm_s(ps).transpose(1, 2))
        ps_pool = Rearrange('b c 1 -> b 1 c')(ps_pool)
        ps_cls = self.projs[1](ps_pool)
        pl1 = torch.cat((ps_cls, self.norm_l(pl1)), dim=1)
        ps1 = torch.cat((pl_cls, self.norm_s(ps)), dim=1)
        pl1 = pl1 + self.pos_embed_l
        ps1 = ps1 + self.pos_embed_s
        # print("pl1,ps1 shape:",pl1.shape,ps1.shape)
        p_concat = torch.cat((self.norm_l(pl1), self.norm_s(ps1)), dim=2)
        p_concat = self.wls1(p_concat)
        p_concat = self.gelu(p_concat)
        p_concat = self.wls2(p_concat)
        # print("pcat shape:",p_concat.shape)
        q = self.wq(ps1)
        k = self.wk(p_concat)
        v = self.wv(p_concat)

        # k = torch.stack((q, k), dim=1).reshape(q.shape[0], -1, q.shape[1])
        # v = torch.stack((q, v), dim=1).reshape(q.shape[0], -1, q.shape[1])
        B, N, C = ps1.shape
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale_l
        attn_weights = F.softmax(attn, dim=-1)
        output = attn_weights @ v
        # torch.save(output, 'results/hiformer-s/test/pl_weight.pt')
        output = output.permute(0, 2, 1, 3).reshape(B, N, C)
        # print(output.shape)
        output = output[:, 1:, :]
        # torch.save(output, 'results/hiformer-s/test/pl_out.pt')
        output = self.fc_out(output)
        # print("outshape:",pl.shape,output.shape)
        return pl, output


class crossattention(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3, embed_dim=(96, 384), norm_layer=nn.LayerNorm):
        super().__init__()
        self.average = nn.AdaptiveAvgPool1d(1)
        self.cross_pos_embed = True
        # self.pyramid = PyramidFeatures(config=config, img_size=img_size, in_channels=in_chans)
        self.norm_l = nn.LayerNorm(embed_dim[0])
        self.norm_s = nn.LayerNorm(embed_dim[1])
        n_p1 = (config.image_size // config.patch_size) ** 2  # default: 3136
        n_p2 = (config.image_size // config.patch_size // 4) ** 2  # default: 196
        num_patches = (n_p1, n_p2)
        self.num_branches = 2

        self.pos_embed = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])

        total_depth = sum([sum(x[-2:]) for x in config.depth])
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, total_depth)]  # stochastic depth decay rule
        dpr_ptr = 0
        self.blocks = nn.ModuleList()
        for idx, block_config in enumerate(config.depth):
            curr_depth = max(block_config[:-1]) + block_config[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(embed_dim, num_patches, block_config, num_heads=config.num_heads,
                                  mlp_ratio=config.mlp_ratio,
                                  qkv_bias=config.qkv_bias, qk_scale=config.qk_scale, drop=config.drop_rate,
                                  attn_drop=config.attn_drop_rate, drop_path=dpr_, norm_layer=norm_layer)
            dpr_ptr += curr_depth
            self.blocks.append(blk)

        self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])

        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                trunc_normal_(self.pos_embed[i], std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def forward(self, pl, ps):
        pl_pool = self.average(self.norm_l(pl).transpose(1, 2))
        pl_pool = Rearrange('b c 1 -> b 1 c')(pl_pool)
        pl = torch.cat((pl_pool, pl), dim=1)
        # pl_pool = self.f_l(pl_pool)
        ps_pool = self.average(self.norm_s(ps).transpose(1, 2))
        ps_pool = Rearrange('b c 1 -> b 1 c')(ps_pool)
        ps = torch.cat((ps_pool, ps), dim=1)
        xs = [pl, ps]
        if self.cross_pos_embed:
            for i in range(self.num_branches):
                xs[i] += self.pos_embed[i]

        for blk in self.blocks:
            xs = blk(xs)
        xs = [self.norm[i](x) for i, x in enumerate(xs)]

        return xs





class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        # x1 = self.conv1(x)
        # return  self.conv2(x1) + x1
        return self.double_conv(x)

class CrossfusionBlock(nn.Module):
    def __init__(self,dim,L):
        super().__init__()
        self.num_heads = 3
        self.patchexpand = PatchExpandInterp(in_dim=dim[1])
        self.pos_embed1 = nn.Parameter(torch.zeros(1, L, dim[0]))
        trunc_normal_(self.pos_embed1, std=.02)
        self.pos_embed2 = nn.Parameter(torch.zeros(1,L, dim[0]))
        trunc_normal_(self.pos_embed2, std=.02)
        self.linear1 = nn.Linear(2*dim[0], dim[0])
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(dim[0], dim[0])
        self.norm1 = nn.LayerNorm(dim[0])
        self.norm2 = nn.LayerNorm(dim[0])
        self.wq = nn.Linear(dim[0], dim[0])
        self.wk = nn.Linear(dim[0], dim[0])
        self.wv = nn.Linear(dim[0], dim[0])
        self.scale = (dim[0] // self.num_heads) ** -0.5
        self.fc_out = nn.Linear(dim[0],dim[0])

    def forward(self,p1,p2):
        p2_up = self.norm1(self.patchexpand(p2))
        p2_up = p2_up + self.pos_embed2
        p1 = self.norm2(p1)
        p1 = p1 + self.pos_embed1
        p1_p2 = torch.cat((p1,p2_up),dim=2)
        p1_p2 = self.linear1(p1_p2)
        p1_p2 = self.gelu(p1_p2)
        p1_p2 = self.linear2(p1_p2)
        q = self.wq(p2_up)
        k = self.wk(p1_p2)
        v = self.wv(p1)
        B, N, C = p1.shape
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        output = attn_weights @ v
        output = output.permute(0, 2, 1, 3).reshape(B, N, C)
        output = self.fc_out(output)
        return output
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

# 如果你的环境有 trunc_normal_（torch >=1.7），可直接用；否则可用 nn.init.normal_ 替代。
try:
    from torch.nn.init import trunc_normal_
except:
    def trunc_normal_(tensor, mean=0., std=1.):
        return nn.init.normal_(tensor, mean=mean, std=std)


class NBPFilterBank(nn.Module):
    def __init__(self, H, W, K=3):
        super().__init__()
        self.K = K
        self.H = H
        # 可学习中心与带宽（初始化为从低、中、高三段开始）
        init_mu = torch.linspace(0, 0.5, K) * (H**2 + W**2) ** 0.5
        init_sigma = torch.full((K,), 0.15 * (H**2 + W**2) ** 0.5)
        self.mu = nn.Parameter(init_mu)
        self.sigma = nn.Parameter(init_sigma)
        self.alpha = nn.Parameter(torch.ones(K))
        # 预生成半径网格
        u = torch.linspace(-0.5, 0.5, H).unsqueeze(1).expand(H, W)
        v = torch.linspace(-0.5, 0.5, W).unsqueeze(0).expand(H, W)
        self.register_buffer('r', torch.sqrt(u**2 + v**2) * (H**2 + W**2) ** 0.5)

    def forward(self, spec):  # spec: (B,C,H,W,2) 复数分离或 (B,C,H,W) 复张量
        filters = []
        for k in range(self.K):
            Bk = torch.exp(-((self.r - self.mu[k]) ** 2) / (2 * (self.sigma[k] ** 2))) ** torch.relu(self.alpha[k])
            filters.append(Bk)
        # (K,H,W)
        Fb = torch.stack(filters, dim=0)
        np.save("saved_weights/filter_"+str(self.H)+".npy", Fb.detach().cpu().numpy())
        return Fb  # 上层用逐带相乘

class BandCrossAttention(nn.Module):
    def __init__(self, c_in, d=64, heads=4):
        super().__init__()
        self.q = nn.Conv2d(c_in, d, 1)
        self.k = nn.Conv2d(c_in, d, 1)
        self.v = nn.Conv2d(c_in, d, 1)
        self.proj = nn.Conv2d(d, c_in, 1)
        self.heads = heads
        self.scale = (d // heads) ** -0.5

    def forward(self, Ft_band, Fc_band):  # (B,C,H,W) 频带后的幅/实部特征
        B,C,H,W = Fc_band.shape
        Q = self.q(Ft_band).view(B, self.heads, -1, H*W)        # (B,h,dh,HW)
        K = self.k(Fc_band).view(B, self.heads, -1, H*W)
        V = self.v(Fc_band).view(B, self.heads, -1, H*W)
        attn = (Q.transpose(2,3) @ K) * self.scale               # (B,h,HW,HW)
        attn = attn.softmax(dim=-1)
        out = attn @ V.transpose(2,3)                            # (B,h,HW,dh)
        out = out.transpose(2,3).contiguous().view(B, -1, H, W)
        return self.proj(out)

class NBPFusion(nn.Module):
    def __init__(self, ch, H, K=3, heads=4):
        super().__init__()
        self.proj_c = nn.Conv2d(ch, ch, 1)
        self.proj_t = nn.Conv2d(ch, ch, 1)
        self.nbp = NBPFilterBank(H, H, K=K)
        self.cross = nn.ModuleList([
            BandCrossAttention(ch, d=min(128, ch), heads=heads)
            for _ in range(K)
        ])
        # self.gate = nn.Sequential(nn.Conv2d(ch*2, ch, 1), nn.Sigmoid())
        self.gate = nn.Sequential(nn.Conv2d(ch * 3, ch, 1), nn.Sigmoid())
        self.fuse = nn.Conv2d(ch*2, ch, 3, padding=1)
        self.proj_cf = nn.Conv2d(ch*2, ch, 1)

    def forward(self, Fc, Ft):
        Fc = self.proj_c(Fc)
        Ft = self.proj_t(Ft)
        B, C, H, W = Fc.shape

        # 复频谱（B,C,H,W）
        Fc_spec = fft.fftshift(fft.fft2(Fc, norm="ortho"))
        Ft_spec = fft.fftshift(fft.fft2(Ft, norm="ortho"))

        # 频带滤波
        Bk = self.nbp(Fc)  # (K,H,W)
        out_spec = torch.zeros(B, C, H, W, dtype=torch.complex64, device=Fc.device)
        band_feats = []

        for k in range(Bk.shape[0]):
            filt = Bk[k][None, None, ...]  # (1,1,H,W)
            Fc_k = Fc_spec * filt
            Ft_k = Ft_spec * filt

            # 回到实域做注意力
            Fc_k_sp = fft.ifft2(fft.ifftshift(Fc_k), norm="ortho").real
            Ft_k_sp = fft.ifft2(fft.ifftshift(Ft_k), norm="ortho").real

            band_out = self.cross[k](Fc_k_sp, Ft_k_sp)
            band_feats.append(band_out)

            # 累加复数频谱，保留相位
            out_spec = out_spec + fft.fftshift(fft.fft2(band_out, norm="ortho"))

        # 频域聚合→空间，取幅值
        tildeF = torch.abs(fft.ifft2(fft.ifftshift(out_spec), norm="ortho"))
        # np.save("saved_weights/fused_"+str(H)+".npy", tildeF.detach().cpu().numpy())

        # 门控融合
        # gate = self.gate(torch.cat([Ft, tildeF], dim=1))
        gate = self.gate(torch.cat([Fc, Ft, tildeF], dim=1))
        cf_fused = self.proj_cf(torch.cat([Fc, Ft], dim=1))
        # out = gate * tildeF + (1-gate)*Ft
        # print(torch.mean(gate),torch.max(gate),torch.min(gate))
        out = self.fuse(torch.cat([gate * tildeF, (1 - gate) * cf_fused], dim=1))

        return out, band_feats


class CrossfusionBidirectional(nn.Module):
    def __init__(self, dim, L, num_heads=3):
        """
        dim: int channel dimension (C)
        L: int, sequence length N = H*W (assumed square)
        num_heads: number of attention heads
        """
        super().__init__()
        C = dim
        self.C = C
        self.num_heads = num_heads
        self.head_dim = C // num_heads
        assert self.head_dim * num_heads == C, "C must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5
        self.L = L

        self.patchexpand = PatchExpandInterp(in_dim=2*C)

        # layer norms
        self.norm1 = nn.LayerNorm(C)  # for p2_up
        self.norm2 = nn.LayerNorm(C)  # for p1

        # 小的融合前置 MLP (用于产生跨尺度融合的 KV 源)
        self.pre_linear1 = nn.Linear(2 * C, C)
        self.pre_act = nn.GELU()
        self.pre_linear2 = nn.Linear(C, C)

        # Q/K/V 投影
        self.wq_h = nn.Linear(C, C)  # for p1 as Q (high-res -> low)
        self.wk_h = nn.Linear(C, C)
        self.wv_h = nn.Linear(C, C)

        self.wq_l = nn.Linear(C, C)  # for p2_up as Q (low-res -> high)
        self.wk_l = nn.Linear(C, C)
        self.wv_l = nn.Linear(C, C)

        # 输出变换
        self.fc_out_h2l = nn.Linear(C, C)
        self.fc_out_l2h = nn.Linear(C, C)
        self.fc_fuse = nn.Linear(2 * C, C)

        # gating 融合器（生成形状 B,N,1 的门）
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * C, C),
            nn.GELU(),
            nn.Linear(C, 1)
        )

        # 可学习的缩放残差项，帮助训练（类似 Transformer 的 gamma）
        self.gamma_h2l = nn.Parameter(torch.zeros(1))
        self.gamma_l2h = nn.Parameter(torch.zeros(1))

        # 相对位置编码的表和索引（基于方形 HxW）
        H = W = int(math.sqrt(L))
        assert H * W == L, "L must be a perfect square (H*W)"
        self.H = H
        self.W = W

        # relative position bias table: ( (2H-1)*(2W-1) , num_heads )
        num_rel = (2 * H - 1) * (2 * W - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_rel, num_heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=.02)

        # compute relative_position_index buffer
        coords_h = torch.arange(H)
        coords_w = torch.arange(W)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, H, W
        coords_flatten = coords.reshape(2, -1)  # 2, H*W
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 2
        relative_coords[:, :, 0] += H - 1
        relative_coords[:, :, 1] += W - 1
        relative_coords[:, :, 0] *= 2 * W - 1
        relative_position_index = relative_coords.sum(-1)  # N, N
        self.register_buffer("relative_position_index", relative_position_index)

        # 初始化线性层
        nn.init.xavier_uniform_(self.pre_linear1.weight)
        nn.init.xavier_uniform_(self.pre_linear2.weight)
        nn.init.xavier_uniform_(self.wq_h.weight)
        nn.init.xavier_uniform_(self.wk_h.weight)
        nn.init.xavier_uniform_(self.wv_h.weight)
        nn.init.xavier_uniform_(self.wq_l.weight)
        nn.init.xavier_uniform_(self.wk_l.weight)
        nn.init.xavier_uniform_(self.wv_l.weight)

    def _get_rel_pos_bias(self):
        # 返回形状 (num_heads, N, N)
        # table: (num_rel, num_heads)
        # index: (N, N)
        rp = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.L, self.L, -1)  # N, N, num_heads
        rp = rp.permute(2, 0, 1).contiguous()  # num_heads, N, N
        return rp

    def _qkv_multihead(self, x, proj):
        # x: B,N,C, proj: linear
        B, N, C = x.shape
        q = proj(x)  # B,N,C
        q = q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B,heads,N,hd
        return q

    def forward(self, p1, p2):
        """
        p1: (B, N, C)  # high-res
        p2: (B, N2, C) # lower-res, will be upsampled
        """
        B = p1.shape[0]

        # 1) p2 upsample -> p2_up
        p2_up = self.patchexpand(p2)  # should return (B, N, C)
        p2_up = self.norm1(p2_up)
        p1_n = self.norm2(p1)

        # 2) pre-fuse
        p1_p2 = torch.cat((p1_n, p2_up), dim=2)  # B, N, 2C
        p1_p2 = self.pre_linear1(p1_p2)
        p1_p2 = self.pre_act(p1_p2)
        p1_p2 = self.pre_linear2(p1_p2)  # B, N, C

        # 3) prepare Q/K/V for both directions
        # Direction A: high-res (p1) queries low-res up (p2_up) as K/V (高 -> 低)
        q_h = self._qkv_multihead(p1_n, self.wq_h)  # B,heads,N,hd
        k_h = self._qkv_multihead(p2_up, self.wk_h)
        v_h = self._qkv_multihead(p2_up, self.wv_h)

        # Direction B: low-res up (p2_up) queries fused (p1_p2) as K/V (低 -> 高)
        q_l = self._qkv_multihead(p2_up, self.wq_l)
        k_l = self._qkv_multihead(p1_p2, self.wk_l)
        v_l = self._qkv_multihead(p1_p2, self.wv_l)

        # 4) attention compute with relative position bias
        # q @ k^T -> (B, heads, N, N)
        attn_h2l = (q_h @ k_h.transpose(-2, -1)) * self.scale  # B,heads,N,N
        attn_l2h = (q_l @ k_l.transpose(-2, -1)) * self.scale  # B,heads,N,N

        # add relative position bias
        rel_bias = self._get_rel_pos_bias()  # heads, N, N
        attn_h2l = attn_h2l + rel_bias.unsqueeze(0)
        attn_l2h = attn_l2h + rel_bias.unsqueeze(0)

        attn_h2l = F.softmax(attn_h2l, dim=-1)
        attn_l2h = F.softmax(attn_l2h, dim=-1)

        out_h2l = (attn_h2l @ v_h).permute(0, 2, 1, 3).contiguous().view(B, self.L, self.C)  # B,N,C
        out_l2h = (attn_l2h @ v_l).permute(0, 2, 1, 3).contiguous().view(B, self.L, self.C)

        out_h2l = self.fc_out_h2l(out_h2l) * self.gamma_h2l  # scaled residual style
        out_l2h = self.fc_out_l2h(out_l2h) * self.gamma_l2h

        # 5) gating 融合 (按位置自适应)
        gate_input = torch.cat([out_h2l, out_l2h], dim=2)  # B,N,2C
        gate = torch.sigmoid(self.gate_mlp(gate_input))  # B,N,1
        fused = gate * out_h2l + (1.0 - gate) * out_l2h  # B,N,C

        # optional: residual connection back to p1 (或你想要回到 p2_up)
        fused = fused + p1  # 残差，保留原始高分辨率信息

        # 最后一个融合线性（可选）
        fused = self.fc_fuse(torch.cat([fused, p1], dim=2))  # B,N,C

        return fused

class PatchExpandInterp(nn.Module):
    def __init__(self, in_dim, out_dim=None, scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.scale = scale
        self.out_dim = out_dim or in_dim // 2
        self.proj = nn.Linear(in_dim, self.out_dim)
        self.norm = norm_layer(self.out_dim)

    def forward(self, x):
        # x: (B, L, C), L = H*W
        B, L, C = x.shape
        H = int(np.sqrt(L))
        W = H
        x = self.proj(x)                 # (B, L, out_dim)
        x = self.norm(x)

        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, C, H, W)
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        # 还原到 (B, L', C)
        _, _, H2, W2 = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H2 * W2, -1)
        return x

class Encoder(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3, embed_dim=(96, 384), norm_layer=nn.LayerNorm):
        super().__init__()
        self.imagesize = img_size

        self.swin_transformer = SwinTransformer(img_size, in_chans=3)
        resnet = eval(f"torchvision.models.{config.cnn_backbone}(pretrained={config.resnet_pretrained})")
        self.resnet_layers = nn.ModuleList(resnet.children())[:7]
        self.p1_ch = nn.Conv2d(config.cnn_pyramid_fm[0], config.swin_pyramid_fm[0], kernel_size=1)  # 256,256
        self.p1_pm = PatchMerging((img_size // 4, img_size // 4),
                                  config.swin_pyramid_fm[0])

        self.norm_1 = nn.LayerNorm(config.swin_pyramid_fm[0])
        self.avgpool_1 = nn.AdaptiveAvgPool1d(1)
        self.p2 = self.resnet_layers[5]
        self.p2_ch = nn.Conv2d(config.cnn_pyramid_fm[1], config.swin_pyramid_fm[1], kernel_size=1)
        self.p2_pm = PatchMerging(
            (img_size // 8, img_size // 8),
            config.swin_pyramid_fm[1])

        self.norm_2 = nn.LayerNorm(config.swin_pyramid_fm[1])
        self.avgpool_2 = nn.AdaptiveAvgPool1d(1)

        self.p3 = self.resnet_layers[6]
        self.p3_ch = nn.Conv2d(config.cnn_pyramid_fm[2], config.swin_pyramid_fm[2], kernel_size=1)
        self.norm_3 = nn.LayerNorm(config.swin_pyramid_fm[2])
        self.avgpool_3 = nn.AdaptiveAvgPool1d(1)
        self.NBPfusion1 = NBPFusion(ch=96,H=img_size//4)
        self.NBPfusion2 = NBPFusion(ch=192, H=img_size // 8)
        self.NBPfusion3 = NBPFusion(ch=384, H=img_size // 16)
        # self.correlation1 = CorrelationBlock(ch=96)
        # self.correlation2 = CorrelationBlock(ch=192)
        # self.correlation3 = CorrelationBlock(ch=384)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=config.patch_size, in_chans=in_chans, embed_dim=embed_dim[0],
            norm_layer=norm_layer)

    def forward(self, x):
        # torch.save(x,'results/hiformer-s/test/x.pt')
        fm1 = x
        for i in range(5):
            fm1 = self.resnet_layers[i](fm1)
        fm1_ch = self.p1_ch(fm1)
        fm1_reshaped = Rearrange('b c h w -> b (h w) c')(fm1_ch)
        # sw1 = self.patch_embed(x)
        sw1 = self.swin_transformer.layers[0](fm1_reshaped)
        sw1_reshaped = Rearrange("b (h w) c -> b c h w",h=self.imagesize // 4,w=self.imagesize // 4)(sw1)

        sw1_skipped,bandfeats1 = self.NBPfusion1(fm1_ch,sw1_reshaped)
        band_feats_np = [f.detach().cpu().numpy() for f in bandfeats1]
        # np.save("saved_weights/band_feats1.npy", band_feats_np, allow_pickle=True)
        # np.save("saved_weights/sw1.npy", sw1_reshaped.detach().cpu().numpy())
        # np.save("saved_weights/cnn1.npy", fm1_ch.detach().cpu().numpy())
        # np.save("saved_weights/fused1.npy", sw1_skipped.detach().cpu().numpy())
        sw1_skipped = Rearrange('b c h w -> b (h w) c')(sw1_skipped)

        # sw1_skipped = self.correlation1(fm1_reshaped, sw1)
        sw1_up = self.p1_pm(sw1_skipped)

        # Level 2
        sw2 = self.swin_transformer.layers[1](sw1_up)
        sw2_reshaped = Rearrange("b (h w) c -> b c h w",h=self.imagesize // 8,w=self.imagesize // 8)(sw2)
        fm2 = self.p2(fm1)
        fm2_ch = self.p2_ch(fm2)
        sw2_skipped,bandfeats2 = self.NBPfusion2(fm2_ch,sw2_reshaped)
        band_feats_np = [f.detach().cpu().numpy() for f in bandfeats2]
        # np.save("saved_weights/band_feats2.npy", band_feats_np, allow_pickle=True)
        # np.save("saved_weights/sw2.npy", sw2_reshaped.detach().cpu().numpy())
        # np.save("saved_weights/cnn2.npy", fm2_ch.detach().cpu().numpy())
        # np.save("saved_weights/fused2.npy", sw2_skipped.detach().cpu().numpy())
        sw2_skipped = Rearrange('b c h w -> b (h w) c')(sw2_skipped)
        # fm2_reshaped = Rearrange('b c h w -> b (h w) c')(fm2_ch)
        # fm2_sw2_skipped = self.correlation2(fm2_reshaped, sw2)
        sw2_up = self.p2_pm(sw2_skipped)

        # Level 3
        sw3 = self.swin_transformer.layers[2](sw2_up)
        sw3_reshaped = Rearrange("b (h w) c -> b c h w", h=self.imagesize // 16, w=self.imagesize // 16)(sw3)
        fm3 = self.p3(fm2)
        fm3_ch = self.p3_ch(fm3)
        sw3_skipped,bandfeats3 = self.NBPfusion3(fm3_ch,sw3_reshaped)
        band_feats_np = [f.detach().cpu().numpy() for f in bandfeats3]
        # np.save("saved_weights/band_feats3.npy", band_feats_np, allow_pickle=True)
        # np.save("saved_weights/sw3.npy", sw3_reshaped.detach().cpu().numpy())
        # np.save("saved_weights/cnn3.npy", fm3_ch.detach().cpu().numpy())
        # np.save("saved_weights/fused3.npy", sw3_skipped.detach().cpu().numpy())
        sw3_skipped = Rearrange('b c h w -> b (h w) c')(sw3_skipped)
        # fm3_reshaped = Rearrange('b c h w -> b (h w) c')(fm3_ch)
        # sw3_skipped = self.correlation3(fm3_reshaped, fm2_sw3)

        return [sw1_skipped, sw2_skipped, sw3_skipped]

class Decoder(nn.Module):
    def __init__(self,img_size = 224, n_class=2):
        super().__init__()
        # self.crossattention = MultiScaleBlock(dim=(96, 192, 384),depth=[1,1,1], num_heads=(3, 3, 3), mlp_ratio=(1., 1., 1.))
        l = img_size // 4 * img_size // 4

        self.img_size = img_size
        self.fusion1 = CrossfusionBidirectional(dim=192,L=l//4)
        self.fusion2 = CrossfusionBidirectional(dim=96,L=l)
        # self.fusion1 = CrossfusionBlock(dim=(192, 384), L=l//4)
        # self.fusion2 = CrossfusionBlock(dim=(96, 192), L=l)
        self.upsample = nn.Upsample(scale_factor=2)
        self.norm0 = nn.BatchNorm2d(96)
        self.norm1 = nn.BatchNorm2d(192)
        self.norm11 = nn.BatchNorm2d(192)
        self.norm2 = nn.BatchNorm2d(384)
        self.norm01 = nn.BatchNorm2d(96)
        self.conv01 = DoubleConv(in_channels=288, out_channels=96)
        self.conv11 = DoubleConv(in_channels=576, out_channels=192)
        self.conv02 = DoubleConv(in_channels=384, out_channels=96)
        self.out_conv = nn.Sequential(
            nn.Conv2d(
                96, 16,
                kernel_size=1, stride=1,
                padding=0, bias=True),
            # nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )
        self.segmentation_head = nn.Conv2d(in_channels=16, out_channels=n_class, kernel_size=3, padding=1)

    def forward(self, x):
        # x = self.crossattention(x)
        p0, p1, p2 = x[0],x[1],x[2]
        # np.save("saved_weights/p1_before.npy", p1.detach().cpu().numpy())
        # np.save("saved_weights/p2_before.npy", p2.detach().cpu().numpy())
        # np.save("saved_weights/p0_before.npy", p0.detach().cpu().numpy())
        p1 = self.fusion1(p1, p2)
        # np.save("saved_weights/p1_after.npy", p1.detach().cpu().numpy())
        p0 = self.fusion2(p0, p1)
        # print("shape p0,p1,p2",p0.shape,p1.shape,p2.shape)
        # np.save("saved_weights/p0_after.npy", p0.detach().cpu().numpy())

        p0 = Rearrange('b (h w) c -> b c h w', h=self.img_size//4, w=self.img_size//4)(p0)
        p1 = Rearrange('b (h w) c -> b c h w', h=self.img_size//8, w=self.img_size//8)(p1)
        p2 = Rearrange('b (h w) c -> b c h w', h=self.img_size//16, w=self.img_size//16)(p2)
        p0 = self.norm0(p0)
        p1 = self.norm1(p1)
        p2 = self.norm2(p2)
        p1_up = self.upsample(p1)
        p01 = torch.cat((p1_up, p0), dim=1)
        p01 = self.conv01(p01)
        p01 = self.norm01(p01)
        p2_up = self.upsample(p2)
        p11 = torch.cat((p2_up, p1), dim=1)
        p11 = self.conv11(p11)
        p11 = self.norm11(p11)
        p11_up = self.upsample(p11)
        p02 = torch.cat((p0, p01, p11_up), dim=1)
        p02 = self.conv02(p02)
        out = self.out_conv(p02)
        out = self.segmentation_head(out)
        # print("shape out:",out.shape)
        return out

class LGBPOrga(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3, n_classes=2):
        super().__init__()
        self.encoder = Encoder(config,img_size,in_chans)
        self.decoder = Decoder(img_size=img_size,n_class=n_classes)
    def forward(self,x):
        # torch.save(x,"results/SROrga/test/x.pt")
        x = self.encoder(x)
        out = self.decoder(x)
        return out


