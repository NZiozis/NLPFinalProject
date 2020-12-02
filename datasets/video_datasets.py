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
    '''
    @param root (str): Path to root directory for Tasty Video Dataset
    @param split (str): Can be 'train' or 'test'
    @param img_transform (torch.Transform): Apply a Pytorch image transformation to all images
    '''
    def __init__(self, root='/nfs/bigiris/cristinam/Tasty_Videos_Dataset', split="train", embedding_type='fasttext', img_transform=None):
        assert split in ["train", "val", "test"]
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.crop_size = 256

        # Load embeddings file
        if embedding_type == 'fasttext':
            with open('data/fasttext_embeds_codecs.txt', 'r') as embedFile:
                embeds = embedFile.readlines()
        elif embedding_type == 'glove':
            with open('data/vocab_glove_embeds.txt', 'r') as embedFile:
                embeds = embedFile.readlines()
        else:
            print("Embedding type ", embedding_type, " not supported")
            embeds = None
        # Create dictionary mapping word to embedding vector
        self.embed_dict = dict()
        for embed in embeds:
            spl = embed.split(' ')
            float_list = [float(i) for i in spl[1:-1]]
            self.embed_dict[spl[0]] = float_list

        # Load ingredient dict
        with open('data/ingredient_dict.json', 'r') as ingDictFile:
            self.ingredient_dict = json.load(ingDictFile)
        self.num_ingredients = len(self.ingredient_dict.keys())

        # Load word dict
        with open('data/Tasty_Videos_Dataset/id2word_tasty.txt', 'rb') as idFile:
            data = idFile.read()
        id_dict = eval(data)
        id_dict_decoded = dict()
        for key, val in id_dict.items():
            id_dict_decoded[key] = codecs.decode(val)

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
                        # Get video frames spaced every 10 frames in range
                        frames_list = [os.path.join(self.root, 'ALL_RECIPES_without_videos', name, 'frames', (str(i)+'.jpg').zfill(9)) for i in range(start, end+1, 20)]
                        if len(frames_list) > 0:
                            frames.append(frames_list)
                            # Get corresponding step text
                            steps.append(recipe_dict["steps"][count])
                    count+=1

                # Remove duplicate ingredients
                ingredients = list(set(recipe_dict["ingredients"]))
                '''
                # Check if all ingredients exist in vocabulary
                num_ing_missing = 0
                missing_ing = []
                ing_single_words = []
                for ing in ingredients:
                    split_ing = ing.split(' ')
                    ing_single_words.extend(split_ing)
                for v in ing_single_words:
                    if v not in id_dict_decoded.values():
                        num_ing_missing +=1
                        missing_ing.append(v)
                if num_ing_missing != 0:
                    print(name, num_ing_missing)
                    print(missing_ing)
                '''
                self.files[split].append({
                    "frames": frames,
                    "sentences": steps,
                    "ingredients": ingredients
                })


    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]
        frames = []
        for interval in datafiles["frames"]:
            interval_frames = []
            for f in interval:
                img = Image.open(f).convert('RGB')
                img = img.resize((self.crop_size, self.crop_size))
                np3ch = np.array(img)
                # If image is grayscale, convert to "color" by replicating channels
                if np3ch.shape != (256, 256, 3):
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
        steps_embeds = []
        for step in steps_words:
            split_step = step.split(' ')
            # TODO: replace zero vector with UNK embedding
            embeds = [self.embed_dict.get(i, [0] * 100) for i in split_step]
            steps_embeds.append(torch.FloatTensor(embeds))

        # TODO: Get indices in vocab for each word in each sentence
        steps = None

        #print("num intervals ", len(frames))
        #print("shape of intervals ", [frames[i].shape for i in range(len(frames))])
        #print("len steps ", len(steps))
        #print("shape of steps ", [steps[i].shape for i in range(len(steps))])
        #print("shape ingredients ", ingredients.shape)
        return frames, steps, steps_embeds, ingredients




