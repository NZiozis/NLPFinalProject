import torch
from torch.utils import data
import collections
import os
from bs4 import BeautifulSoup
import re
from PIL import Image
import numpy as np
import json
import codecs

class TastyVideoDataset(data.Dataset):
    """ Implements Pytorch dataset for training baseline and joint models on the Tasty Video Dataset
    """
    def __init__(self, root='/nfs/bigiris/cristinam/Tasty_Videos_Dataset', split="train", embedding_type='fasttext', img_transform=None):
        '''
        @param root (str): Path to root directory for Tasty Video Dataset
        @param split (str): Can be 'train', 'val' or 'test'
        @param embedding_type (str): Specifies what type of word embeddings to use. Can be 'fasttext' or 'glove'.
        @param img_transform (torch.Transform): Apply a Pytorch image transformation to all images
        '''
        assert split in ["train", "val", "test"]
        self.root = root
        self.split = split
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        # Define the spatial resolution of the images
        self.crop_size = 128

        # Load word embeddings file
        if embedding_type == 'fasttext':
            with open('data/fasttext_embeds_codecs.txt', 'r') as embedFile:
                embeds = embedFile.readlines()
        elif embedding_type == 'glove':
            with open('data/vocab_glove_embeds.txt', 'r') as embedFile:
                embeds = embedFile.readlines()
        else:
            print("Embedding type ", embedding_type, " not supported")
            embeds = None

        # Create dictionary mapping word (str) to embedding vector
        self.embed_dict = dict()
        for embed in embeds:
            spl = embed.split(' ')
            float_list = [float(i) for i in spl[1:-1]]
            self.embed_dict[spl[0]] = float_list

        # Load ingredient dict
        with open('data/ingredient_dict.json', 'r') as ingDictFile:
            self.ingredient_dict = json.load(ingDictFile)
        self.num_ingredients = len(self.ingredient_dict.keys())

        # Load steps words dict
        with open('data/steps_vocab_dict.json', 'r') as stepsDictFile:
            self.steps_vocab_dict = json.load(stepsDictFile)
        self.vocab_size = len(self.steps_vocab_dict.keys())

        # Load recipes from the dataset
        with open('data/Tasty_Videos_Dataset/all_recipes_processed.txt', 'r') as recipeFile:
            recipe_dicts = json.load(recipeFile)

        for name, recipe_dict in recipe_dicts.items():
            if recipe_dict["split"] == self.split:
                frames, steps = [], []
                count = 0
                max_num_frames = len(os.listdir(os.path.join(self.root, 'ALL_RECIPES_without_videos', name, 'frames')))
                for elt in recipe_dict["annotations"]:
                    # Check if interval exists
                    if len(elt) == 2:
                        start, end = elt[0], elt[1]
                        # If end interval is larger than max number of frames in folder, set it to max
                        if max_num_frames < end:
                            end = max_num_frames-1
                        # Get video frames spaced every 50 frames in range
                        frames_list = [os.path.join(self.root, 'ALL_RECIPES_without_videos', name, 'frames', (str(i)+'.jpg').zfill(9)) for i in range(start, end+1, 50)]
                        if len(frames_list) > 0:
                            frames.append(frames_list)
                            # Get corresponding step text
                            steps.append(recipe_dict["steps"][count])
                    count+=1

                # Remove duplicate ingredients
                ingredients = list(set(recipe_dict["ingredients"]))

                # Add recipe sample to dataset:
                # frames is a list of lists. Each nonempty list contains file paths to frames corresponding to a recipe step.
                # steps is a list of strings, one string for each recipe step.
                # ingredients is a list of strings, one for each ingredient in the recipe.
                self.files[split].append({
                    "frames": frames,
                    "sentences": steps,
                    "ingredients": ingredients
                })


    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        '''
        @return (frames, steps_indices, steps_embeds, ingredients) where
        frames is a list of lists. Each list contains numpy arrays for resized frames corresponding to a recipe step, i.e.,
        frames = [[frame1_recipestep1, frame2_recipestep1, ...], [frame1_recipestep2, frame2_recipestep2], ...]
        
        steps_indices is a list of torch.FloatTensor. Each Tensor contains indices of words in the vocabulary, i.e.,
        steps_indices = [[recipestep1_word1_index, recipestep1_word2_index, ...], [recipestep2_word1_index, recipestep2_word2_index, ...], ...]

        steps_embeds is a list of lists of torch.FloatTensors. Each Tensor contains word embedding, i.e.,
        steps_embeds = [[recipestep1_word1_embedding, recipestep1_word2_embedding, ...], [recipestep2_word1_embedding, recipestep2_word2_embedding, ...], ...]
        
        ingredients is a torch.FloatTensor with 1 at index of ingredient in recipe and 0 otherwise, i.e.,
        ingredients = torch.FloatTensor([0, 0, ..., 1, 0, 1, ...])
        If ingredients has a 1 at index N then the ingredient corresponding to index N is in the recipe.
        '''
        datafiles = self.files[self.split][index]
        frames = []
        for interval in datafiles["frames"]:
            interval_frames = []
            for f in interval:
                img = Image.open(f).convert('RGB')
                img = img.resize((self.crop_size, self.crop_size))
                np3ch = np.array(img)
                # If image is grayscale, convert to "color" by replicating channels
                if np3ch.shape != (self.crop_size, self.crop_size, 3):
                    np3ch = np3ch.repeat(3,1,1)
                # Permute dimensions to make channel first
                np3ch = np.moveaxis(np3ch, -1, 0)
                interval_frames.append(np3ch)
            frames.append(np.array(interval_frames))

        # Ingredient vector has 1's at index of each ingredient
        ingredient_words = datafiles["ingredients"]
        ingredient_indices = [self.ingredient_dict[ing] for ing in ingredient_words]
        ingredients = []
        count = 0
        while count < self.num_ingredients:
            if count in ingredient_indices:
                ingredients.append(1)
            else:
                ingredients.append(0)
            count+=1
        ingredients = torch.FloatTensor(ingredients)

        # Get word embeddings for every word in each step sentence
        steps_words = datafiles["sentences"]
        steps_embeds, steps_indices = [], []
        for step in steps_words:
            split_step = step.split(' ')
            # TODO: replace zero vector with UNK embedding
            embeds = [self.embed_dict.get(i, [0] * 100) for i in split_step]
            steps_embeds.append(torch.FloatTensor(embeds))
            # Also get index of words from step
            indices = [self.steps_vocab_dict[i] for i in split_step]
            steps_indices.append(torch.FloatTensor(indices))

        return frames, steps_indices, steps_embeds, ingredients




