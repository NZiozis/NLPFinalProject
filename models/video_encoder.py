import torch
import torch.nn as nn
import torchvision

class VideoEncoder(nn.Module):
	def __init__(self, lstm_hidden_size=512):
		super(VideoEncoder, self).__init__()

	self.backbone = torchvision.models.resnet50(pretrained=True)
	self.lstm = nn.LSTM(hidden_size=lstm_hidden_size, bidirectional=True)
	self.maxpool = nn.MaxPool1d()

	def forward(self, inputs):
		# Get resnet features per frame
		backbone_feats = self.backbone(inputs)

		# Pass features into bidirectional LSTM
		lstm_feats = self.lstm(backbone_feats)

		# Apply max pool over timesteps
		out = self.maxpool(lstm_feats)

		return out