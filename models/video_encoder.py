import torch
import torch.nn as nn
import torchvision

class VideoEncoder(nn.Module):
	def __init__(self, input_size=512, lstm_hidden_size=512):
		super(VideoEncoder, self).__init__()

		self.backbone = torchvision.models.resnet50(pretrained=True)
		self.backbone.fc = torch.nn.Linear(in_features=2048, out_features=512, bias=True)
		self.lstm = nn.LSTM(input_size=512, hidden_size=512, batch_first=True, bidirectional=True)
		self.maxpool = nn.MaxPool1d(kernel_size=512)

	def forward(self, inputs):
		"""
		@param inputs (torch.Tensor): Shape (1, number_frames, height, width, channels)
		"""
		# Remove batch dimension of 1
		inputs = torch.squeeze(inputs, dim=0)

		# Get resnet features per frame
		backbone_feats = self.backbone(inputs)
		
		# Add batch dimension back
		backbone_feats = torch.unsqueeze(backbone_feats, dim=0)

		# Pass features into bidirectional LSTM
		lstm_feats = self.lstm(backbone_feats)[0]

		# Apply max pool over timesteps
		out = self.maxpool(lstm_feats)
		print("maxpool shape ", out.shape)

		return out