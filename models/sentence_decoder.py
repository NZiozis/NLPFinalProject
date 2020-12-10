import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence

class SentenceDecoder(nn.Module):
    def __init__(self, args, max_seq_length=30):
        """Set the hyper-parameters and build the layers."""
        super(SentenceDecoder, self).__init__()
        self.sentDec_inDim = args.sentDec_inDim
        # Set this to size of word/ingredient embeddings -> 100 for now
        self.word_dim = 100
        self.sentDec_hiddens = args.sentDec_hiddens
        self.vocab_len = 12269
        self.sentDec_nlayers = args.sentDec_nlayers

        self.lstm = nn.LSTM(self.word_dim, self.sentDec_hiddens, self.sentDec_nlayers, batch_first=True)
        self.linear = nn.Linear(self.sentDec_hiddens, self.vocab_len)
        self.max_seg_length = max_seq_length
        self.linear_project = nn.Linear(self.sentDec_inDim, self.word_dim)

    def forward(self, recipe_enc, sent_lens, word_embs):
        """Decode sentence feature vectors and generates sentences."""
        # recipe_enc --> [Nb, 1024]
        # word_embs  --> [Nb, Ns, 256]
        # len(sent_lens)  --> Nb

        '''
        features = self.linear_project(recipe_enc)  # [Nb, 256]
        features = features.unsqueeze(0)
        out, _ = self.lstm(features)
        outputs = self.linear(out[0])  # [sum(sent_lens), Nw] -- Nw = number of words in the vocabulary
        '''
        features = self.linear_project(recipe_enc)  # [Nb, 256]
        features = features.squeeze(0).unsqueeze(1)
        word_embs = torch.cat((features, word_embs), 1)  # torch.Size([Nb, Ns + 1, 256])
        packed = pack_padded_sequence(word_embs, sent_lens, batch_first=True, enforce_sorted=False)
        
        out, _ = self.lstm(packed)
        outputs = self.linear(out[0])  # [sum(sent_lens), Nw] -- Nw = number of words in the vocabulary

        return outputs
