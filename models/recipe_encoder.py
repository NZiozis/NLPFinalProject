import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class RecipeEncoder(nn.Module):
    def __init__(self, args):
        """Set the hyper-parameters and build the layers."""
        super(RecipeEncoder, self).__init__()
        self.recipe_inDim = args.recipe_inDim
        self.recipe_hiddens = args.recipe_hiddens
        self.recipe_nlayers = args.recipe_nlayers

        self.lstm = nn.LSTM(self.recipe_inDim, self.recipe_hiddens, self.recipe_nlayers, batch_first=True)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.recipe_nlayers, bsz, self.recipe_hiddens).zero_()),
                Variable(weight.new(self.recipe_nlayers, bsz, self.recipe_hiddens).zero_()))

    def forward(self, ingredient_feats, recipes_v, rec_lens, use_teacherF):
        # ingredient_feats -> [N, 1, 1024]
        # recipes_v -> [N, rec_lens[0], 1024]
        # len(rec_lens -> N

        if use_teacherF == False:
            recipes_v_i = torch.cat((ingredient_feats, recipes_v), 1)  # [N, rec_lens[0] + 1, 1024]
            recipes_v_packed = pack_padded_sequence(recipes_v_i, rec_lens, batch_first=True)
            out, _ = self.lstm(recipes_v_packed)  # [0] --> [sum(rec_lens), 1024],  [1] --> [rec_lens[0]]
        else:
            inputs = ingredient_feats  # [N, 1, 1024]
            states = self.init_hidden(recipes_v.shape[0])
            # [0]-> [1, len(rec_lens), 1024], [1] -> [1, len(rec_lens), 1024]

            sampled_instructions = []
            for di in range(rec_lens[0]):  # max recipe len.
                hiddens, states = self.lstm(inputs, states)
                if random.random() < (0.5):
                    inputs = hiddens  # [N, 1, 1024]
                else:
                    inputs = recipes_v[:, di, :].unsqueeze(1)  # [N, 1, 1024]
                sampled_instructions.append(hiddens.squeeze(1))

            sampled_instructions = torch.stack(sampled_instructions, 1)  # torch.Size([N, rec_lens[0], 1024])

            out = pack_padded_sequence(sampled_instructions, rec_lens, batch_first=True)
            # [0] --> [sum(rec_lens), 1024],  [1] --> [rec_lens[0]]

        return out[0]  # [sum(rec_lens), 1024]

