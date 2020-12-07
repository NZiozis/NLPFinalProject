import torch
import torch.nn as nn
import torchvision

class VideoEncoder(nn.Module):
	def __init__(self, input_size=512, lstm_hidden_size=512):
		super(VideoEncoder, self).__init__()

		self.backbone = torchvision.models.resnet50(pretrained=True)
		self.backbone.fc = torch.nn.Linear(in_features=2048, out_features=512, bias=True) # For resnet 50
		#self.backbone.fc = torch.nn.Linear(in_features=512, out_features=512, bias=True) # For resnet34 or resnet18
		self.lstm = nn.LSTM(input_size=512, hidden_size=512, batch_first=True, bidirectional=True)
		self.maxpool = nn.MaxPool1d(kernel_size=1024)

	def forward(self, inputs):
		"""
		@param inputs (torch.Tensor): Shape (1, number_frames, height, width, channels)
		@return out (torch.Tensor): Shape (1, 1024) where 1024 = dimension output by LSTM
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
		lstm_feats = lstm_feats.permute(0, 2, 1)
		num_steps = lstm_feats.shape[2]
		pool = nn.MaxPool1d(num_steps)
		out = pool(lstm_feats)
		out = torch.squeeze(out, dim=2)

		return out