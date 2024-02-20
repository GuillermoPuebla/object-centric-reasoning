from typing import Tuple
from typing import Union

import torch
from torch import nn
from torch.nn import functional as F

from itertools import permutations

"""
The SlotAttention model comes from: https://github.com/untitled-ai/slot_attention
"""
def conv_transpose_out_shape(in_size, stride, padding, kernel_size, out_padding, dilation=1):
    return (in_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + out_padding + 1

def assert_shape(actual: Union[torch.Size, Tuple[int, ...]], expected: Tuple[int, ...], message: str = ""):
    assert actual == expected, f"Expected shape: {expected} but passed shape: {actual}. {message}"

def build_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)

class SlotAttention(nn.Module):
    def __init__(self, in_features, num_iterations, num_slots, slot_size, mlp_hidden_size, epsilon=1e-8):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size  # number of hidden layers in slot dimensions
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(self.in_features)
        # I guess this is layer norm across each slot? should look into this
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_mlp = nn.LayerNorm(self.slot_size)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_k = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_v = nn.Linear(self.slot_size, self.slot_size, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

        self.register_buffer(
            "slots_mu",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )
        self.register_buffer(
            "slots_log_sigma",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )

    def forward(self, inputs):
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        batch_size, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(k.size(), (batch_size, num_inputs, self.slot_size))
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(v.size(), (batch_size, num_inputs, self.slot_size))

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        slots_init = torch.randn((batch_size, self.num_slots, self.slot_size))
        slots_init = slots_init.type_as(inputs)
        slots = self.slots_mu + self.slots_log_sigma.exp() * slots_init

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            assert_shape(q.size(), (batch_size, self.num_slots, self.slot_size))

            attn_norm_factor = self.slot_size ** -0.5
            attn_logits = attn_norm_factor * torch.matmul(k, q.transpose(2, 1))
            attn = F.softmax(attn_logits, dim=-1)
            # `attn` has shape: [batch_size, num_inputs, num_slots].
            assert_shape(attn.size(), (batch_size, num_inputs, self.num_slots))

            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.matmul(attn.transpose(1, 2), v)
            # `updates` has shape: [batch_size, num_slots, slot_size].
            assert_shape(updates.size(), (batch_size, self.num_slots, self.slot_size))

            # Slot update.
            # GRU is expecting inputs of size (N,H) so flatten batch and slots dimension
            slots = self.gru(
                updates.view(batch_size * self.num_slots, self.slot_size),
                slots_prev.view(batch_size * self.num_slots, self.slot_size),
            )
            slots = slots.view(batch_size, self.num_slots, self.slot_size)
            assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))
            slots = slots + self.mlp(self.norm_mlp(slots))
            assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))

        return slots


class SlotAttRelNetModel(nn.Module):
    """
    Modified Slot Attention model with relation network module for the same-different task.
    """
    def __init__(
        self,
        resolution: Tuple[int, int],
        num_slots: int,
        num_iterations,
        in_channels: int = 3,
        kernel_size: int = 5,
        slot_size: int = 64,
        hidden_dims: Tuple[int, ...] = (64, 64, 64, 64),
        decoder_resolution: Tuple[int, int] = (8, 8), #(8, 8),
        empty_cache=False,
        f_out_d=2
        ):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.slot_size = slot_size
        self.empty_cache = empty_cache
        self.hidden_dims = hidden_dims
        self.decoder_resolution = decoder_resolution
        self.out_features = self.hidden_dims[-1]

        # RN pars
        self.g_hidden_d=slot_size*4
        self.g_out_d = slot_size*2
        self.f_hidden_d=slot_size*4
        self.f_out_d = f_out_d

        # Build Encoder
        modules = []
        channels = self.in_channels
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        out_channels=h_dim,
                        kernel_size=self.kernel_size,
                        stride=1,
                        padding=self.kernel_size // 2
                        ),
                    nn.LeakyReLU()
                    )
                )
            channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.encoder_pos_embedding = SoftPositionEmbed(self.in_channels, self.out_features, resolution)
        self.encoder_out_layer = nn.Sequential(
            nn.LayerNorm(self.out_features),
            nn.Linear(self.out_features, self.out_features),
            nn.LeakyReLU(),
            nn.Linear(self.out_features, self.out_features)
            )

        # Build Decoder
        modules = []
        in_size = decoder_resolution[0]
        out_size = in_size
        # print(out_size)
        # print()

        for i in range(len(self.hidden_dims) - 1, -1, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.hidden_dims[i],
                        self.hidden_dims[i - 1],
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
            out_size = conv_transpose_out_shape(out_size, 2, 2, 5, 1)
            # print(out_size)
        
        # print()
        # print(resolution)
        # print((out_size, out_size))
        # input()
        assert_shape(
            resolution,
            (out_size, out_size),
            message="Output shape of decoder did not match input resolution. Try changing `decoder_resolution`.",
            )

        # same convolutions
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    self.out_features, self.out_features, kernel_size=5, stride=1, padding=2, output_padding=0,
                    ),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(self.out_features, 4, kernel_size=3, stride=1, padding=1, output_padding=0,),
                )
            )

        assert_shape(resolution, (out_size, out_size), message="")

        self.decoder = nn.Sequential(*modules)
        self.decoder_pos_embedding = SoftPositionEmbed(self.in_channels, self.out_features, self.decoder_resolution)

        # Slot Attention module
        self.slot_attention = SlotAttention(
            in_features=self.out_features,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size= self.slot_size * 2 #128
            )
        
        # g function RN
        self.g_theta = nn.Sequential(
            nn.Linear(self.slot_size*2, self.g_hidden_d),
            nn.LeakyReLU(),
            nn.Linear(self.g_hidden_d, self.g_out_d)
            )

        # f function RN
        self.f_phi = nn.Sequential(
            nn.Linear(self.g_out_d, self.f_hidden_d),
            nn.LeakyReLU(),
            nn.Linear(self.f_hidden_d, self.f_out_d)
            )

    def forward(self, x):
        if self.empty_cache:
            torch.cuda.empty_cache()

        batch_size, num_channels, height, width = x.shape
        encoder_out = self.encoder(x)
        encoder_out = self.encoder_pos_embedding(encoder_out)
        # `encoder_out` has shape: [batch_size, filter_size, height, width]
        encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
        # `encoder_out` has shape: [batch_size, filter_size, height*width]
        encoder_out = encoder_out.permute(0, 2, 1)
        encoder_out = self.encoder_out_layer(encoder_out)
        # `encoder_out` has shape: [batch_size, height*width, filter_size]

        slots = self.slot_attention(encoder_out)
        assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))
        # `slots` has shape: [batch_size, num_slots,  ].
        batch_size, num_slots, slot_size = slots.shape

        # The slots feed directly to the relation network
        # Pass all non-repeating pairs of slots to g_theta
        slot_pairs = permutations(list(range(self.num_slots)), 2)
        all_g = []
        for pair in slot_pairs:
            g_out = self.g_theta(torch.cat([slots[:,pair[0],:], slots[:,pair[1],:]], dim=1))
            all_g.append(g_out)
        # Stack and sum all outputs from g_theta
        all_g = torch.stack(all_g, 1).sum(1)
        # Apply f_phi
        f_phi =  self.f_phi(all_g)

        # Decoder
        slots = slots.view(batch_size * num_slots, slot_size, 1, 1)
        # print('slots.shape:', slots.shape)
        decoder_in = slots.repeat(1, 1, self.decoder_resolution[0], self.decoder_resolution[1])
        # print('decoder_in.shape:', decoder_in.shape)
        out = self.decoder_pos_embedding(decoder_in)
        # print('out.shape:', out.shape)
        out = self.decoder(out)
        # print('out.shape:', out.shape)
        # input()
        # `out` has shape: [batch_size*num_slots, num_channels+1, height, width].
        assert_shape(out.size(), (batch_size * num_slots, num_channels + 1, height, width))

        out = out.view(batch_size, num_slots, num_channels + 1, height, width)
        recons = out[:, :, :num_channels, :, :]
        masks = out[:, :, -1:, :, :]
        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)

        return recon_combined, recons, masks, slots, f_phi

    def loss_function(self, input):
        """Loss function for unsupervised object discovery (not used in this simulation)"""
        recon_combined, recons, masks, slots = self.forward(input)
        loss = F.mse_loss(recon_combined, input)
        return {
            "loss": loss,
        }


class SoftPositionEmbed(nn.Module):
    def __init__(self, num_channels: int, hidden_size: int, resolution: Tuple[int, int]):
        super().__init__()
        self.dense = nn.Linear(in_features=num_channels + 1, out_features=hidden_size)
        self.register_buffer("grid", build_grid(resolution))

    def forward(self, inputs):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        assert_shape(inputs.shape[1:], emb_proj.shape[1:])
        return inputs + emb_proj

