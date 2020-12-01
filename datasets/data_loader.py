# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data
from .RECIPE1M import RECIPE1M
from .video_datasets import TastyVideoDataset


class Recipe1MDataset(data.Dataset):
    """RECIPE1M Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, args, vocab):
        self.recipe1m = RECIPE1M(args)
        self.ids = list(self.recipe1m.ingredients.keys())
        self.vocab = vocab

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """Returns one data pair (ingredients and recipe)."""
        recipe1m = self.recipe1m
        vocab = self.vocab
        ann_id = self.ids[index]
        ingredients = torch.tensor(recipe1m.ingredients[ann_id], dtype=torch.float)

        tokens = recipe1m.sentences[ann_id]
        target_captions = []
        for x in range(0, len(tokens)):
            caption = [vocab('<start>'), vocab('<end>')]
            caption.extend([vocab(token) for token in tokens[x]])
            target_captions.append(torch.Tensor(caption))
            
        return ingredients, target_captions


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (ingredients, recipes).
    Args:
        data: list of tuple (ingredients, recipes).
            - ingredients: torch tensor of shape
            - recipes: torch tensor of shape (?); variable length.
    Returns:
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    ingredients, target_captions = zip(*data)

    ingredients_v = torch.stack(ingredients, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in target_captions]

    list_of_sents = []
    list_sent_encoding = []
    counter = 0
    for i in range(len(target_captions)):
        cap = target_captions[i]
        counts = i
        for x in range(0, len(cap)):
            numinc = len([v for v in lengths if v > x])
            list_of_sents.append((cap[x], counts))
            counts = counts + numinc
            list_sent_encoding.append((cap[x], counter))
            counter = counter + 1

    list_of_sents.sort(key=lambda x: len(x[0]), reverse=True)
    captions, indices = zip(*list_of_sents)
    list_sent_encoding.sort(key=lambda x: len(x[0]), reverse=True)
    _, indices_encoder = zip(*list_sent_encoding)

    lengths_captions = [len(cap) for cap in captions]
    captions_v = torch.zeros(len(captions), max(lengths_captions)).long()
    for i, cap in enumerate(captions):
        end = lengths_captions[i]
        captions_v[i, :end] = cap[:end]

    return ingredients_v, lengths, captions_v, torch.LongTensor(lengths_captions), \
           torch.LongTensor(indices), torch.LongTensor(indices_encoder)


def get_loader(args, batch_size, vocab, shuffle, num_workers, use_video=False):
    """Returns torch.utils.data.DataLoader for custom Recipe1M or TVD dataset."""

    if use_video:
        tasty_videos = TastyVideoDataset()
        data_loader = torch.utils.data.DataLoader(dataset=tasty_videos,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    else:
        recipe1m = Recipe1MDataset(args, vocab)
        data_loader = torch.utils.data.DataLoader(dataset=recipe1m,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  collate_fn=collate_fn)
    return data_loader
