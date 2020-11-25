import torch
from torch.utils import data
import collections
import os
from bs4 import BeautifulSoup
import re
from PIL import Image
import numpy as np


class TastyVideoDataset(data.Dataset):
    '''
    @param root (str): Path to root directory for Tasty Video Dataset
    @param split (str): Can be 'train' or 'test'
    @param img_transform (torch.Transform): Apply a Pytorch image transformation to all images
    '''
    def __init__(self, root='/nfs/bigiris/cristinam/Tasty_Videos_Dataset', split="train", img_transform=None):
        assert split in ["train", "val"]
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.crop_size = 256

        imgsets_dir = os.path.join(self.root, 'SPLITS/split_2511', '{}ing_set.txt'.format(split))
        with open(imgsets_dir) as imgset_file:
            names = imgset_file.readlines()
        # Remove \n from video names
        names = [name[:-1] for name in names]
        # Remove files with no alignment data
        names.remove('mixed-berry-smoothie-meal-prep')
        names.remove('peach-feta-and-mint-stuffed-avocado')

        recipes_mismatch = []
        for name in names:
            vid_folder = os.path.join(self.root, 'ALL_RECIPES_without_videos', name)
            with open(os.path.join(vid_folder, 'csvalignment.dat'), 'r') as alignmentFile:
                alignment_file = alignmentFile.readlines()
            alignment_file = [(int(a[:-1].split(',')[0]), int(a[:-1].split(',')[1])) for a in alignment_file]
            with open(os.path.join(vid_folder, 'recipe.xml'), 'r') as recipeFile:
                recipe_data = recipeFile.read()
            all_recipe_steps = []
            recipe = BeautifulSoup(recipe_data, "xml") 
            steps = recipe.find('steps')
            for elt in steps.children:
                if elt.string != '\n':
                    all_recipe_steps.append(elt.string)

            # Separate sentences in recipe steps
            recipe_steps = []
            for s in all_recipe_steps:
                split_sentence = s.split('.')
                recipe_steps.extend(elt for elt in split_sentence if re.search('[a-zA-Z]', elt) != None)
            

            if len(alignment_file) != len(recipe_steps):
                recipes_mismatch.append(name)
            else:
                # Uncomment the following assertion when sure that alignments and recipe steps are the same
                #assert len(alignment_file) == len(recipe_steps)
                frames, sentences, ingredients = [], [], []
                # For each interval in alignment file
                count = 0
                for start, end in alignment_file:
                    # If interval not empty
                    if start != end:
                        # Get video frames
                        frames.append([os.path.join(vid_folder, 'frames', (str(i)+'.jpg').zfill(9)) for i in range(start, end+1)])
                        # Get sentences for this step
                        sentences.append(recipe_steps[count])
                    count+=1

                ing = []
                ingredients = recipe.find('ingredients')
                for elt in ingredients.children:
                    ing.append(elt.string)
                ing = [i for i in ing if i != '\n']

                self.files[split].append({
                        "frames": frames,
                        "sentences": sentences,
                        "ingredients": ing
                })

        # TODO: Ignore mismatched sentence/frame alignments for now
        #print("recipes_mismatch ", recipes_mismatch)
        #print("len recipes mismatch ", len(recipes_mismatch))

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

        return frames, datafiles["sentences"], datafiles["ingredients"]




