import torch
import torch.nn as nn

class RecipeEncoder(nn.Module):
	def __init__(self, ingredients_one_hot_len=3769, ingredients_vec_dim=1024, lstm_hidden_size=512):
		super(RecipeEncoder, self).__init__()

	self.fc = nn.Linear(ingredients_one_hot_len, ingredients_vec_dim)
	self.lstm = nn.LSTM(hidden_size=lstm_hidden_size)

	def forward(self, inputs):
		# Apply fully connected layer to ingredients vector to get first hidden state
		h_0 = self.fc(inputs[0])
		inputs = torch.cat(h_0, inputs[1:])

		# Pass features into RNN
		lstm_feats = self.lstm(inputs)

		return lstm_feats