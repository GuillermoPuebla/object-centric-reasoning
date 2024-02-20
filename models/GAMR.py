# Imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Encoders
class DoubleConv(nn.Module):
	"""(convolution => [BN] => ReLU) * 2"""

	def __init__(self, in_channels, out_channels, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
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

class Enc_Conv_v0_16(nn.Module):
	def __init__(self):
		super().__init__()

		self.inc = DoubleConv(3, 64) # 64x128x128
		self.down1 = Down(64, 64) # 64x64x64
		self.down1s = DoubleConv(64, 64)
		self.down2 = Down(64, 128) # 128x32x32
		self.down2s = DoubleConv(128, 128)
		self.down3 = Down(128, 128) # 128x16x16
		self.down3s = DoubleConv(128, 128)
		self.down4 = Down(128, 128) # 128x8x8
		self.down4s = DoubleConv(128, 256)
		self.down5 = DoubleConv(256, 256) # 128x8x8
		self.down5s = DoubleConv(256, 256)
		self.relu = nn.ReLU()

		self.conv_kv = nn.Conv2d(256, 128, 1, stride=1, padding=0)

	def forward(self, x):
		x = self.inc(x)

		x = self.down1s(self.down1(x))
		x = self.down2s(self.down2(x))
		x = self.down3s(self.down3(x))
		x = self.down4s(self.down4(x))
		conv_out = self.down5s(self.down5(x))
		z_kv = self.relu(self.conv_kv(conv_out))      # BxCxHxW 'relu' required 
		z_kv = torch.transpose(z_kv.view(x.shape[0], 128, -1), 2, 1) # B, C, HW

		return z_kv

class Enc_Conv_v0_16_ESBN(nn.Module):
	def __init__(self):
		super().__init__()

		self.inc = DoubleConv(3, 64) # 64x128x128
		self.down1 = Down(64, 64) # 64x64x64
		self.down1s = DoubleConv(64, 64)
		self.down2 = Down(64, 128) # 128x32x32
		self.down2s = DoubleConv(128, 128)
		self.down3 = Down(128, 128) # 128x16x16
		self.down3s = DoubleConv(128, 128)
		self.down4 = Down(128, 128) # 128x8x8
		self.down4s = Down(128, 256)
		self.down5 = Down(256, 256) # 128x8x8
		self.down5s = Down(256, 256)
		self.relu = nn.ReLU()

		self.conv_kv = nn.Conv2d(256, 128, 1, stride=1, padding=0)
	
	def forward(self, x):
		x = self.inc(x)
		
		x = self.down1s(self.down1(x))
		x = self.down2s(self.down2(x))
		x = self.down3s(self.down3(x))
		x = self.down4s(self.down4(x))
		conv_out = self.down5s(self.down5(x))
		z_kv = self.conv_kv(conv_out) # BxCxHxW
		z_kv = self.relu(torch.squeeze(z_kv))
		return z_kv

class Encoder_conv(nn.Module):
	def __init__(self):
		super(Encoder_conv, self).__init__()
		# Convolutional layers
		self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
		# Fully-connected layers
		self.fc1 = nn.Linear(8192, 256)
		self.fc2 = nn.Linear(256, 128)
		# Nonlinearities
		self.relu = nn.ReLU()
		# Initialize parameters
		for name, param in self.named_parameters():
			# Initialize all biases to 0
			if 'bias' in name:
				nn.init.constant_(param, 0.0)
			# Initialize all pre-ReLU weights using Kaiming normal distribution
			elif 'weight' in name:
				nn.init.kaiming_normal_(param, nonlinearity='relu')
	def forward(self, x):
		# Convolutional layers
		conv1_out = self.relu(self.conv1(x))
		conv2_out = self.relu(self.conv2(conv1_out))
		conv3_out = self.relu(self.conv3(conv2_out))
		# Flatten output of conv. net
		conv3_out_flat = torch.flatten(conv3_out, 1)
		# Fully-connected layers
		fc1_out = self.relu(self.fc1(conv3_out_flat))
		fc2_out = self.relu(self.fc2(fc1_out))
		# Output
		z = fc2_out
		return z

class Encoder_conv_deepfc(nn.Module):
	def __init__(self,):
		super(Encoder_conv_deepfc, self).__init__()
		# Convolutional layers
		self.conv1 = DoubleConv(1, 64) # 64x64x32
		self.conv2 = Down(64, 64) # 64x32x16
		self.conv3 = Down(64, 128) # 128x16x8
		self.conv4 = Down(128, 128) # 128x8x4
		self.conv5 = Down(128, 128) # 128x4x2

		# Nonlinearities
		self.relu = nn.ReLU()

	def forward(self, x):
		# Convolutional layers
		conv_out = self.conv1(x)
		conv_out = self.conv2(conv_out)
		conv_out = self.conv3(conv_out)
		conv_out = self.conv4(conv_out)

		# Output
		z = self.conv5(conv_out)
		return torch.transpose(z.view(x.shape[0], 128, -1), 2, 1) # bringing it to dimension B, 8, 128

# MAREO model
class TransformerEncoderLayer_qkv(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model=128, nhead=8, dim_feedforward=128, dropout=0.1, activation="relu", mlp = False):
        super(TransformerEncoderLayer_qkv, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.mlp = mlp
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        if mlp:
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)
            self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer_qkv, self).__setstate__(state)

    def forward(self, src, k, v, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2, _ = self.self_attn(src, k, v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if self.mlp:
            src2 = self.activation(self.linear1(src))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class MAREO(nn.Module):
    def __init__(self, encoder, norm_type, steps=4):
        super(MAREO, self).__init__()
        # Encoder
        if encoder == 'original':
            self.encoder = Encoder_conv_deepfc()
            self.n_channels = 1
        elif encoder == 'custom':
            self.encoder = Enc_Conv_v0_16()
            self.n_channels = 3
        else:
            raise ValueError('Unrecognized encoder!') 
        
        # LSTM and output layers
        self.z_size = 128
        self.key_size = 256
        self.hidden_size = 512
        self.lstm = nn.LSTM(self.z_size, self.hidden_size, batch_first=True)
        self.g_out = nn.Linear(self.hidden_size, self.z_size)
        self.y_out = nn.Linear(self.hidden_size+256, 2)

        # New addition
        self.query_size = 128
        self.query_w_out = nn.Linear(self.hidden_size, self.query_size)

        # time step:
        self.time_step = steps

        # Transformer
        self.ga = TransformerEncoderLayer_qkv(d_model=128, nhead=8)

        # RN module terms:
        self.g_theta_hidden = nn.Linear((self.z_size+self.z_size), 512)
        self.g_theta_out = nn.Linear(512, 256)
        
        # Context normalization
        if norm_type == 'contextnorm' or norm_type == 'tasksegmented_contextnorm':
            self.contextnorm = True
            self.gamma1 = nn.Parameter(torch.ones(self.z_size))
            self.beta1 = nn.Parameter(torch.zeros(self.z_size))
        else:
            self.contextnorm = False
        if norm_type == 'tasksegmented_contextnorm':
            self.task_seg = None
        else:
            self.task_seg = [np.arange(8)] # as the output is of dim B, 32, 128

        # Nonlinearities
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        # Initialize parameters
        for name, param in self.named_parameters():
            # Encoder parameters have already been initialized
            if not ('encoder' in name) and not ('confidence' in name):
                # Initialize all biases to 0
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                else:
                    if 'lstm' in name:
                        # Initialize gate weights (followed by sigmoid) using Xavier normal distribution
                        nn.init.xavier_normal_(param[:self.hidden_size*2,:])
                        nn.init.xavier_normal_(param[self.hidden_size*3:,:])
                        # Initialize input->hidden and hidden->hidden weights (followed by tanh) using Xavier normal distribution with gain = 
                        nn.init.xavier_normal_(param[self.hidden_size*2:self.hidden_size*3,:], gain=5.0/3.0)
                    elif 'key_w' in name:
                        # Initialize weights for key output layer (followed by ReLU) using Kaiming normal distribution
                        nn.init.kaiming_normal_(param, nonlinearity='relu')
                    elif 'query_w' in name:
                        # Initialize weights for query output layer using Kaiming normal distribution
                        nn.init.kaiming_normal_(param)
                    elif 'g_out' in name:
                        # Initialize weights for gate output layer (followed by sigmoid) using Xavier normal distribution
                        nn.init.xavier_normal_(param)
                    elif 'y_out' in name:
                        # Initialize weights for multiple-choice output layer (followed by softmax) using Xavier normal distribution
                        nn.init.xavier_normal_(param)
                    elif 'transformer' in name:
                        # Initialize attention weights using Xavier normal distribution
                        if 'self_attn' in name:
                            nn.init.xavier_normal_(param)
                        # Initialize feedforward weights (followed by ReLU) using Kaiming normal distribution
                        if 'linear' in name:
                            nn.init.kaiming_normal_(param, nonlinearity='relu')

    def forward(self, x_in, device):
        # translating them independently:
        # for t in range(x_seq.shape[0]):
        #     for seq_i in range(x_seq.shape[1]):
        #         x_coord = torch.randint(-5, 5, (1,))
        #         y_coord = torch.randint(-5, 5, (1,))
        #         translation = torch.tensor([[x_coord[0], y_coord[0]]]).to(device)
        #         x_seq[t,seq_i,:,:] = translate(x_seq[t,seq_i,:,:].unsqueeze(0), \
        #             translation.float(), padding_mode = 'border')
        # x_in = torch.cat((x_seq[:,0,:,:], x_seq[:,1,:,:]), 1).unsqueeze(1)
        # x_in = x_in.unsqueeze(1)
        z_img = self.encoder(x_in) # B, 8, 128 each
        z_img = z_img.squeeze(1)
        
        self.task_seg = [np.arange(z_img.shape[1])]
        # (Mohit addition)
        if self.contextnorm:
            # for keys:
            z_seq_all_seg = []
            for seg in range(len(self.task_seg)):
                z_seq_all_seg.append(self.apply_context_norm(z_img[:,self.task_seg[seg],:], self.gamma1, self.beta1))
            z_img = torch.cat(z_seq_all_seg, dim=1) # (M): cat --> stack

        # Initialize hidden state
        hidden = torch.zeros(1, x_in.shape[0], self.hidden_size).to(device)
        cell_state = torch.zeros(1, x_in.shape[0], self.hidden_size).to(device)
        
        # Initialize retrieved key vector
        key_r = torch.zeros(x_in.shape[0], 1, self.z_size).to(device)

        # Memory model (extra time step to process key retrieved on final time step)
        for t in range(self.time_step):
            # Controller
            # LSTM
            lstm_out, (hidden, cell_state) = self.lstm(key_r, (hidden, cell_state))
            # Key & query output layers
            query_r = self.query_w_out(lstm_out)
            g = self.relu(self.g_out(lstm_out))
            w_z = self.ga(z_img, query_r, query_r).sum(1).unsqueeze(1) # [32, 8, 128]
            z_t = (z_img * w_z).sum(1).unsqueeze(1)
            # Read from memory
            if t == 0:
                M_v = z_t
            else:
                M_v = torch.cat([M_v, z_t], dim=1)
            
            w_k = w_z.sum(dim=2) 
            key_r = g * (M_v * w_k.unsqueeze(2)).sum(1).unsqueeze(1)

        # Task output layer
        all_g = []
        for z1 in range(M_v.shape[1]):
            for z2 in range(M_v.shape[1]):
                g_hidden = self.relu(self.g_theta_hidden(torch.cat([M_v[:,z1,:], M_v[:,z2,:]], dim=1)))
                g_out = self.relu(self.g_theta_out(g_hidden))
                all_g.append(g_out) # total length 4

        # Stack and sum all outputs from G_theta
        all_g = torch.stack(all_g, 1).sum(1) # B, 256

        # Task output layer
        y_pred_linear = self.y_out(torch.cat([lstm_out.squeeze(),all_g], dim=1)).squeeze()
        y_pred = y_pred_linear.argmax(1)
        
        return y_pred_linear, y_pred

    def apply_context_norm(self, z_seq, gamma, beta):
        eps = 1e-8
        z_mu = z_seq.mean(1)
        z_sigma = (z_seq.var(1) + eps).sqrt()
        z_seq = (z_seq - z_mu.unsqueeze(1)) / z_sigma.unsqueeze(1)
        z_seq = (z_seq * gamma) + beta
        return z_seq

