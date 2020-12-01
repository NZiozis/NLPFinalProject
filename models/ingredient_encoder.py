import torch
import torch.nn as nn

class IngredientEncoder(nn.Module):
    def __init__(self, ingredient_projection_dim, ingredient_dim):
        """Set the hyper-parameters and build the layers."""
        super(IngredientEncoder, self).__init__()
        self.ing_prj_dim = ingredient_projection_dim
        self.ingredient_dim = ingredient_dim
        self.linear = nn.Linear(self.ingredient_dim, self.ing_prj_dim)
        self.bn = nn.BatchNorm1d(self.ing_prj_dim, momentum=0.01)

    def forward(self, feats):
        # feats --> [N, Nv]
        features = self.linear(feats)  # [N, 1024]
        # Don't include batch normalization if size of batch is 1
        if feats.shape[0] > 1:
            features = self.bn(features)  # [N, 1024]
        return features