# Imports
import math
import numpy as np

import torch
import torchvision
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from e2cnn import gspaces
from e2cnn import nn as enn


# helpers ViT
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes ViT
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, return_attention=False):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        # interesting_head = (attn[0, :, 0, 0].view(-1, 1, 1) != attn[0, :, :, :]).any(1).any(1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        if return_attention:
            return self.to_out(out), attn
        else:
            return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

    def get_last_selfattention(self, x):
        for it, (attn, ff) in enumerate(self.layers):
            if it < len(self.layers) - 1:
                x = attn(x) + x
                x = ff(x) + x
            else:
                _, att = attn.fn(attn.norm(x), return_attention=True)  # b, h, i, j
                return att

class CustomConvInputModel(nn.Module):
    def __init__(self, strides=None, channels=None):
        super(CustomConvInputModel, self).__init__()

        self.conv1 = nn.Conv2d(3, channels[0], 3, stride=strides[0], padding=1)
        self.batchNorm1 = nn.BatchNorm2d(channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=strides[1], padding=1)
        self.batchNorm2 = nn.BatchNorm2d(channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=strides[2], padding=1)
        self.batchNorm3 = nn.BatchNorm2d(channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=strides[3], padding=1)
        self.batchNorm4 = nn.BatchNorm2d(channels[3])

    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = self.batchNorm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.batchNorm4(x)
        x = F.relu(x)

        x = x.view(x.shape[0], x.shape[1], -1)  # B x C x W*H
        x = x.permute(0, 2, 1)  # B x W*H x C
        return x

class EquivariantConvModel(nn.Module):
    def __init__(self, strides=None, channels=None):
        super().__init__()

        # the model is equivariant under rotations by 45 degrees, modelled by C8
        self.r2_act = gspaces.Rot2dOnR2(N=8)

        # the input image is a scalar field, corresponding to the trivial representation
        in_type = enn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])

        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 24 feature fields, each transforming under the regular representation of C8
        out_type = enn.FieldType(self.r2_act, channels[0] * [self.r2_act.regular_repr])
        self.block1 = enn.SequentialModule(
            # enn.MaskModule(in_type, 29, margin=1),
            enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, stride=strides[0]),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True)
        )

        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        out_type = enn.FieldType(self.r2_act, channels[1] * [self.r2_act.regular_repr])
        self.block2 = enn.SequentialModule(
            # enn.MaskModule(in_type, 29, margin=1),
            enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, stride=strides[1]),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True)
        )
        # self.pool1 = nn.SequentialModule(
        #     nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        # )

        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 48 regular feature fields of C8
        out_type = enn.FieldType(self.r2_act, channels[2] * [self.r2_act.regular_repr])
        self.block3 = enn.SequentialModule(
            # enn.MaskModule(in_type, 29, margin=1),
            enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, stride=strides[2]),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True)
        )

        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type = enn.FieldType(self.r2_act, channels[3] * [self.r2_act.regular_repr])
        self.block4 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, stride=strides[3]),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True)
        )

        # self.gpool = enn.GroupPooling(out_type)

        # number of output channels
        # c = self.gpool.out_type.size

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = enn.GeometricTensor(input, self.input_type)

        # apply each equivariant block

        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # pool over the group
        # x = self.gpool(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor

        x = x.view(x.shape[0], x.shape[1], -1)  # B x C x W*H
        x = x.permute(0, 2, 1)  # B x W*H x C

        return x

class Permute(nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.perm = args

    def forward(self, x):
        return x.permute(self.perm)

class ViT(nn.Module):
    def __init__(
        self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
        pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,
        use_conv = False, conv_strides = None, conv_channels = None, 
        equivariant = False, pretrained = None
        ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        if use_conv:
            if conv_strides is None:
                conv_strides = [2, 2, 2, 2]
            if conv_channels is None:
                conv_channels = [32, 64, 128, 256]
            self.patch_res = 4 * 2 ** (8 - sum(conv_strides) + 1)
            print('Conv strides: {}; Num patches: {} x {}'.format(conv_strides, self.patch_res, self.patch_res))
            num_patches = self.patch_res ** 2
            patch_dim = conv_channels[-1] if not equivariant else conv_channels[-1] * 8
            if pretrained == 'rn':
                # self.patch_res = 8
                conv = CustomConvInputModel(strides=conv_strides, channels=[24, 24, 24, 24])
                self.to_patch_embedding = nn.Sequential(
                    conv,
                    nn.Linear(24, dim)
                )
                path = 'pretrained_models/original_fp_epoch_493.pth'
                pretrained_model = torch.load(path, map_location='cpu')
                conv_pretrained_dict = {k.replace('module.conv.', '', 1): v for k, v in pretrained_model.items() if
                                        '.conv.' in k}
                conv.load_state_dict(conv_pretrained_dict)
                print('Using pre-trained RN ConvNet')
            elif isinstance(pretrained, str) and 'resnet' in pretrained:
                self.patch_res = 8  # TODO HEREEEEE. Discover what is the shape in output from the cut resnet
                cut_info = pretrained.split('-')
                cut_point = int(cut_info[1])
                if len(cut_info) == 3:
                    self.patch_res = int(cut_info[2])
                assert (cut_point == 3 and self.patch_res == 8) or cut_point == 2
                num_patches = self.patch_res ** 2
                resnet = torchvision.models.resnet50(pretrained=True)
                conv = nn.Sequential(*list(resnet.children())[:cut_point + 4])  # cut the resnet to the first "cut_point" bottlenecks
                if cut_point == 3:
                    conv_dim = 1024
                elif cut_point == 2:
                    conv_dim = 512
                self.to_patch_embedding = nn.Sequential(
                    conv,
                    torch.nn.AvgPool2d(2, stride=2) if cut_point == 2 and self.patch_res == 8 else torch.nn.Identity(),
                    nn.Flatten(start_dim=2, end_dim=3),
                    Permute(0, 2, 1),
                    nn.Linear(conv_dim, dim)
                )
                print('Using pre-trained ResNet (sliced to first {} modules)'.format(cut_point))
            else:
                self.to_patch_embedding = nn.Sequential(
                    CustomConvInputModel(strides=conv_strides,
                                         channels=conv_channels) if not equivariant else EquivariantConvModel(
                        strides=conv_strides, channels=conv_channels),
                    nn.Linear(patch_dim, dim)
                )

            print('Conv Model: {}'.format(type(self.to_patch_embedding[0])))
        else:
            num_patches = (image_height // patch_height) * (image_width // patch_width)
            patch_dim = channels * patch_height * patch_width
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                nn.Linear(patch_dim, dim),
            )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        if depth > 0:
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        else:
            self.transformer = None

        self.pool = pool
        self.to_latent = nn.Identity()
        self.num_classes = num_classes

        if num_classes > 0:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes)
            )

    def forward(self, img, return_last_attention=False):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        if self.transformer is not None:
            if return_last_attention:
                x = self.transformer.get_last_selfattention(x)
                return x
            else:
                x = self.transformer(x)

        if self.num_classes > 0:
            x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

            x = self.to_latent(x)
            return self.mlp_head(x)
        else:
            return x    # B x S x dim

    def get_last_selfattention(self, img):
        return self.forward(img, True)


# helpers UTEncoder
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """

    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth,
                 num_heads, bias_mask=None, dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiHeadAttention, self).__init__()
        # Checks borrowed from
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
        if total_key_depth % num_heads != 0:
            raise ValueError("Key depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_key_depth, num_heads))
        if total_value_depth % num_heads != 0:
            raise ValueError("Value depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_value_depth, num_heads))

        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5
        self.bias_mask = bias_mask

        # Key and query depth will be same
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3] * self.num_heads)

    def forward(self, queries, keys, values, src_mask=None):

        # Do a linear for each component
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)

        # Split into multiple heads
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        # Scale queries
        queries *= self.query_scale

        # Combine queries and keys
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))

        if src_mask is not None:
            logits = logits.masked_fill(src_mask, -np.inf)

        # Add bias to mask future values
        if self.bias_mask is not None:
            logits += self.bias_mask[:, :, :logits.shape[-2], :logits.shape[-1]].type_as(logits.data)

        # Convert to probabilites
        weights = nn.functional.softmax(logits, dim=-1)

        # Dropout
        weights = self.dropout(weights)

        # Combine with values to get context
        contexts = torch.matmul(weights, values)

        # Merge heads
        contexts = self._merge_heads(contexts)
        # contexts = torch.tanh(contexts)

        # Linear to get output
        outputs = self.output_linear(contexts)

        return outputs, weights

class Conv(nn.Module):
    """
    Convenience class that does padding and convolution for inputs in the format
    [batch_size, sequence length, hidden size]
    """

    def __init__(self, input_size, output_size, kernel_size, pad_type):
        """
        Parameters:
            input_size: Input feature size
            output_size: Output feature size
            kernel_size: Kernel width
            pad_type: left -> pad on the left side (to mask future data),
                      both -> pad on both sides
        """
        super(Conv, self).__init__()
        padding = (kernel_size - 1, 0) if pad_type == 'left' else (kernel_size // 2, (kernel_size - 1) // 2)
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=0)

    def forward(self, inputs):
        inputs = self.pad(inputs.permute(0, 2, 1))
        outputs = self.conv(inputs).permute(0, 2, 1)

        return outputs

class PositionwiseFeedForward(nn.Module):
    """
    Does a Linear + RELU + Linear on each of the timesteps
    """

    def __init__(self, input_depth, filter_size, output_depth, layer_config='ll', padding='left', dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            filter_size: Hidden size of the middle layer
            output_depth: Size last dimension of the final output
            layer_config: ll -> linear + ReLU + linear
                          cc -> conv + ReLU + conv etc.
            padding: left -> pad on the left side (to mask future data),
                     both -> pad on both sides
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(PositionwiseFeedForward, self).__init__()

        layers = []
        sizes = ([(input_depth, filter_size)] +
                 [(filter_size, filter_size)] * (len(layer_config) - 2) +
                 [(filter_size, output_depth)])

        for lc, s in zip(list(layer_config), sizes):
            if lc == 'l':
                layers.append(nn.Linear(*s))
            elif lc == 'c':
                layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            else:
                raise ValueError("Unknown layer type {}".format(lc))

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)

        return x

class LayerNorm(nn.Module):
    # Borrowed from jekbradbury
    # https://github.com/pytorch/pytorch/issues/1959
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class EncoderLayer(nn.Module):
    """
    Represents one Encoder layer of the Transformer Encoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask=None, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                       hidden_size, num_heads, bias_mask, attention_dropout)

        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='cc', padding='both',
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs, return_attn=False):
        x = inputs

        # Layer Normalization
        x_norm = self.layer_norm_mha(x)

        # Multi-head attention
        y, attn = self.multi_head_attention(x_norm, x_norm, x_norm)

        # Dropout and residual
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)

        # Dropout and residual
        y = self.dropout(x + y)

        if return_attn:
            return y, attn
        else:
            return y

def _gen_bias_mask(max_length):
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).type(torch.FloatTensor)

    return torch_mask.unsqueeze(0).unsqueeze(1)


def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]],
                    'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)

### CONVERTED FROM https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/universal_transformer_util.py#L1062
class ACT_basic(nn.Module):
    def __init__(self, hidden_size):
        super(ACT_basic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size, 1)
        self.p.bias.data.fill_(1)
        self.threshold = 1 - 0.1

    def forward(self, state, inputs, fn, time_enc, pos_enc, max_hop, encoder_output=None):
        # init_hdd
        ## [B, S]
        halting_probability = torch.zeros(inputs.shape[0], inputs.shape[1] - 1).cuda()
        ## [B, S
        remainders = torch.zeros(inputs.shape[0], inputs.shape[1] - 1).cuda()
        ## [B, S]
        n_updates = torch.zeros(inputs.shape[0], inputs.shape[1] - 1).cuda()
        ## [B, S, HDD]
        previous_state = torch.zeros_like(inputs).cuda()
        step = 0
        # for l in range(self.num_layers):
        while (((halting_probability < self.threshold) & (n_updates < max_hop)).byte().any()):
            # Add timing signal
            state = state + time_enc[:, :inputs.shape[1], :].type_as(inputs.data)
            state = state + pos_enc[:, step, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)

            p = self.sigma(self.p(state[:, 1:, :])).squeeze(-1)     # the CLS token is not used for computing the stopping prob. CLS is always running
            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).float()

            # Mask of inputs which halted at this step
            new_halted = (halting_probability + p * still_running > self.threshold).float() * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (halting_probability + p * still_running <= self.threshold).float() * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = p * still_running + new_halted * remainders

            if (encoder_output):
                state, _ = fn((state, encoder_output))
            else:
                # apply transformation on the state
                for f in fn:
                    state = f(state)

            # update running part in the weighted state and keep the rest. Do not consider the CLS
            previous_state = torch.cat([
                        state[:, 0, :].unsqueeze(1),
                        (state[:, 1:, :] * update_weights.unsqueeze(-1)) + (previous_state[:, 1:, :] * (1 - update_weights.unsqueeze(-1)))
            ], dim=1)
            ## previous_state is actually the new_state at end of hte loop
            ## to save a line I assigned to previous_state so in the next
            ## iteration is correct. Notice that indeed we return previous_state
            state = previous_state  # This was missing I think
            step += 1
        return previous_state, (remainders, n_updates)

# UTEncoder
class UEncoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=100, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, act=False, internal_enc_layers=1):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(UEncoder, self).__init__()

        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        ## for t
        self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.num_layers = num_layers
        self.act = act
        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length) if use_mask else None,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        self.proj_flag = False
        if (embedding_size != hidden_size):
            self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
            self.proj_flag = True

        self.encs = nn.ModuleList([EncoderLayer(*params) for _ in range(internal_enc_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        if (self.act):
            self.act_fn = ACT_basic(hidden_size)

    def forward(self, inputs, intermediate_outputs=False, return_last_attn=False):

        # Add input dropout
        outputs = []
        x = self.input_dropout(inputs)

        if (self.proj_flag):
            # Project to hidden size
            x = self.embedding_proj(x)

        if (self.act):
            x, (remainders, n_updates) = self.act_fn(x, inputs, self.encs, self.timing_signal, self.position_signal,
                                                     self.num_layers)
            return x, (remainders, n_updates)
        elif return_last_attn:
            assert len(self.encs) == 1
            attns = []
            for l in range(self.num_layers):
                x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                # if l < self.num_layers - 1:
                #     x = self.encs[0](x)
                # else:
                x, attn = self.encs[0](x, return_attn=return_last_attn)
                attns.append(attn)
            return attns
        else:
            for l in range(self.num_layers):
                x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                for e in self.encs:
                    x = e(x)
                if intermediate_outputs:
                    outputs.append(x)
            if intermediate_outputs:
                return torch.stack(outputs, dim=1), None
            else:
                return x, None

    def get_last_selfattention(self, x):
        return self(x, return_last_attn=True)

# The actual class!
class ConvViUT(nn.Module):
    def __init__(self, img_size=128, patch_size=16, vit_depth=2, max_hops=6, vit_conv_strides=None, vit_conv_channels=None, act=False, equivariant=False,
                 vit_dim=512, vit_heads=16, u_heads=8, mlp_dim=1024, internal_enc_layers=1, multiloss=False, pretrained=None, dropout=0.5, uncertainty_weighting="task-dependent"):
        super().__init__()
        self.vit = ViT(
            image_size=128,
            patch_size=16,
            num_classes=-1,
            dim=vit_dim,
            depth=vit_depth,
            heads=vit_heads,
            mlp_dim=mlp_dim,
            dropout=0.1,
            emb_dropout=0.1,
            use_conv=True,
            conv_strides=vit_conv_strides,
            conv_channels=vit_conv_channels,
            equivariant=equivariant,
            pretrained=pretrained
        )

        print('Vit Depth: {}; U Transf Depth: {}'.format(vit_depth, max_hops))

        self.max_hops = max_hops
        if self.max_hops == 0:
            self.u_transformer = None
        else:
            self.u_transformer = UEncoder(embedding_size=vit_dim, hidden_size=vit_dim, num_layers=max_hops, num_heads=u_heads,
                                      total_key_depth=mlp_dim, total_value_depth=mlp_dim, filter_size=mlp_dim, max_length=(img_size // patch_size)**2 + 1,
                                      act=act, internal_enc_layers=internal_enc_layers)
        self.to_latent = nn.Identity()
        self.cls_head = nn.Linear(vit_dim, 3 if uncertainty_weighting == 'data-dependent' else 2)
        self.dropout = nn.Dropout(dropout)
        self.multiloss = multiloss
        self.loss = nn.CrossEntropyLoss()
        self.uncertainty_weighting = uncertainty_weighting
        if uncertainty_weighting == 'task-dependent':
            self.uncertainty_loss_weights = nn.Parameter(torch.ones(max_hops) * -2.3)

    def forward(self, img, targets):
        x = self.vit(img)   # B x S x dim
        if self.u_transformer is not None:
            x, aux = self.u_transformer(x, intermediate_outputs=self.multiloss)   # B x S x dim
            if aux is not None:
                _, n_updates = aux
                n_updates = float(n_updates.mean().item())
            else:
                n_updates = self.max_hops
        if self.multiloss:
            # raise NotImplementedError
            # x is B x num_hops x S x dim
            out = x[:, :, 0, :] # take the CLS from every computational hop  # B x num_hops x dim
            out = self.to_latent(out)
            out = self.dropout(out)
            out = self.cls_head(out)
            loss = self.compute_loss(out, targets)
            # take the most likely prediction
            out = out[..., :2]
            probs = torch.softmax(out, dim=2)
            higher_probs_idx = torch.argmax(torch.abs(probs[:, :, 0] - 0.5), dim=1)
            out = [out[b, i, :] for b, i in enumerate(higher_probs_idx)]
            out = torch.stack(out)
            return loss, out, float(higher_probs_idx.float().mean())

        else:
            out = x[:, 0, :]
            out = self.to_latent(out)
            out = self.dropout(out)
            out = self.cls_head(out)
            # out = out[:, :2]    #TODO: to remove! (in case we erroneously left 5 classes)
            loss = self.loss(out, targets)
            return loss, out, 0

    def compute_loss(self, out, targets):
        if self.multiloss:
            bs, seq_length = out.shape[:2]
            if self.uncertainty_weighting == 'data-dependent':
                out = out.view(-1, out.shape[2])
                out, log_variance = out[:, :2], out[:, 2]
                targets = targets.unsqueeze(1).expand(-1, seq_length).flatten()    # the target is the same at every timestep
                bce_loss = F.cross_entropy(out, targets, reduction='none')
                final_loss = (bce_loss * torch.exp(-log_variance) + log_variance) * 0.5
                final_loss = final_loss.mean()
                return final_loss
            elif self.uncertainty_weighting == 'task-dependent':
                losses = [self.loss(out[:, i, :], targets) for i in range(out.shape[1])]
                if self.uncertainty_loss_weights is not None:
                    losses = [(l * torch.exp(-w) + w) * 0.5 for l, w in zip(losses, self.uncertainty_loss_weights)]
                return sum(losses) / len(losses)
            elif self.uncertainty_weighting == 'mean':
                targets = targets.unsqueeze(1).expand(-1, seq_length)  # the target is the same at every timestep
                return self.loss(out.view(-1, 2), targets.flatten())
            else:
                raise ValueError('Uncertainty weighting mode not recognized!')
        else:
            return self.loss(out, targets)

    def get_last_selfattention(self, img):
        x = self.vit(img)
        attn = self.u_transformer.get_last_selfattention(x)
        return attn

def transformer_econvviut_hires_multiloss_medium(
    pretrained=False,
    map_location=None,
    depth=6,
    u_depth=6
    ):
    model = ConvViUT(
        vit_depth=depth,
        vit_conv_strides=[1, 2, 2, 2],
        patch_size=8,
        equivariant=True,
        vit_conv_channels=[16, 24, 32, 48],
        mlp_dim=512,
        vit_heads=4,
        u_heads=4,
        vit_dim=512,
        max_hops=u_depth,
        multiloss=True,
        pretrained=pretrained)

    return model

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res
