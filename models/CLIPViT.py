import torch
import torch.nn as nn
import clip


def get_CLIP_ViT16(device, freeze_encoder=False):
	model, preprocess = clip.load("ViT-B/16", device=device)
	# Freeze encoder
	if freeze_encoder:
		for name, param in model.visual.named_parameters():
			param.requires_grad = False
	# Add classification head to vision encoder
	in_features = model.visual.proj.shape[1]
	fc = nn.Linear(in_features, 2).to(device)
	model = nn.Sequential(model.visual, fc).float()
	return model