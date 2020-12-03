import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence

def hotfix_pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
    lengths = torch.as_tensor(lengths, dtype=torch.int64)
    #lengths = torch.from_numpy(lengths.copy())
    lengths = lengths.cpu()
    if enforce_sorted:
        sorted_indices = None
    else:
        lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_indices = sorted_indices.to(input.device)
        batch_dim = 0 if batch_first else 1
        input = input.index_select(batch_dim, sorted_indices)

    data, batch_sizes = \
        torch._C._VariableFunctions._pack_padded_sequence(input, lengths, batch_first)
    return PackedSequence(data, batch_sizes, sorted_indices)


class SentenceEncoder(nn.Module):
    def __init__(self, args):
        """Set the hyper-parameters and build the layers."""
        super(SentenceEncoder, self).__init__()
        self.word_emb_dim = 100
        self.enc_lstm_dim = args.sentEnd_hiddens
        self.pool_type = "max"
        self.dpout_model = 0.
        self.vocab_len = args.vocab_len
        self.sentEnd_nlayers = args.sentEnd_nlayers
        self.sentences_sorted = args.sentences_sorted

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, self.sentEnd_nlayers, bidirectional=True,
                                dropout=self.dpout_model)
        self.proj_enc = nn.Linear(2 * self.enc_lstm_dim, 2 * self.enc_lstm_dim, bias=False)

    def forward(self, sent, sent_len):

        sent = sent.permute(1, 0, 2)  # [Ns, Nb, 256]
        bsize = sent.size(1)  # Nb

        if not self.sentences_sorted:
            # Sort by length (keep idx)
            sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
            idx_sort = torch.LongTensor(idx_sort).cuda()
            sent = sent.index_select(1, idx_sort)  # sent --> [Ns, Nb, 256]

        sent_out = self.enc_lstm(sent)[0]

        sent_out = sent_out.view(-1, 2 * self.enc_lstm_dim)  # (Ns*Nb) X 1024
        sent_out = self.proj_enc(sent_out)  # (Ns*Nb) X 1024
        sent_out = sent_out.view(-1, bsize, 2 * self.enc_lstm_dim)  # [Ns, Nb, 1024]

        # Pooling
        emb = torch.max(sent_out, 0)[0].squeeze(0)  # [Nb, 1024]

        return emb
