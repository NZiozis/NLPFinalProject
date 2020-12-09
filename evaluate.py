# -*- coding: utf-8 -*-
import os
import time
import random
import numpy as np
import argparse
import pdb
import json
from math import log
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.autograd import Variable
from scipy.special import softmax

from datasets.video_datasets import TastyVideoDataset
from models.sentence_encoder import SentenceEncoder
from models.video_encoder import VideoEncoder
from models.ingredient_encoder import IngredientEncoder
from models.sentence_decoder import SentenceDecoder
from models.recipe_encoder import RecipeEncoder

import wandb
from rouge import FilesRouge

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def decode_sentence(sentence, index2vocab):
    """ Given a vector representing a sentence using indices, output the sentence string.
    """
    if torch.is_tensor(sentence[0]):
        words = [index2vocab[str(int(elt.cpu().item()))].strip() for elt in sentence]
    else:
        words = [index2vocab[str(elt)].strip() for elt in sentence]
    return " ".join(words)


def generate(args, saved_model_folder, epochs, split, results_path,
             indx2vocab_dict, use_video):
    # Build the models
    encoder_recipe = RecipeEncoder(args)
    encoder_recipe_state_dict =\
        torch.load(os.path.join(saved_model_folder,
                                'encoder_recipe-{}.ckpt'.format(epochs)),
                   map_location=torch.device('cpu'))
    encoder_recipe.load_state_dict(encoder_recipe_state_dict)
    encoder_recipe.eval()
    if device != 'cpu':
        encoder_recipe.cuda()
        encoder_recipe = nn.DataParallel(encoder_recipe)

    if use_video:
        encoder_video = VideoEncoder(args.sentEnd_hiddens)
        encoder_video_state_dict = torch.load(os.path.join(saved_model_folder, 'encoder_video-{}.ckpt'.format(epochs)), map_location=torch.device('cpu'))
        encoder_video.load_state_dict(encoder_video_state_dict)
        encoder_video.eval()
        if device != 'cpu':
            encoder_video.cuda()
            encoder_video = nn.DataParallel(encoder_video)

    encoder_ingredient = IngredientEncoder(1024, 3925)
    encoder_ingredient_state_dict = torch.load(os.path.join(saved_model_folder, 'encoder_ingredient-{}.ckpt'.format(epochs)), map_location=torch.device('cpu'))
    encoder_ingredient.load_state_dict(encoder_ingredient_state_dict)
    encoder_ingredient.eval()
    if device != 'cpu':
        encoder_ingredient.cuda()
        encoder_ingredient = nn.DataParallel(encoder_ingredient)

    decoder_sentences = SentenceDecoder(args)
    decoder_sentences_state_dict = torch.load(os.path.join(saved_model_folder, 'decoder_sentences-{}.ckpt'.format(epochs)), map_location=torch.device('cpu'))
    decoder_sentences.load_state_dict(decoder_sentences_state_dict)
    decoder_sentences.eval()
    if device != 'cpu':
        decoder_sentences.cuda()
        decoder_sentences = nn.DataParallel(decoder_sentences)

    encoder_sentences = SentenceEncoder(args)
    encoder_sentences_state_dict = torch.load(os.path.join(saved_model_folder, 'encoder_sentences-{}.ckpt'.format(epochs)), map_location=torch.device('cpu'))
    encoder_sentences.load_state_dict(encoder_sentences_state_dict)
    encoder_sentences.eval()
    if device != 'cpu':
        encoder_sentences.cuda()
        encoder_sentences = nn.DataParallel(encoder_sentences)

    # Build data loader
    tasty_video_dataset = TastyVideoDataset(split=split, video=use_video)
    test_loader = torch.utils.data.DataLoader(dataset=tasty_video_dataset,
                                              batch_size=1,
                                              shuffle=False)

    # Train the models
    number_samples = len(test_loader)
    print("Total number of samples in ", split, number_samples)

    outputs, gt = [], []
    if use_video:
        for i, (vid_intervals, sentences_indices, sentences_emb, ingredients_v, name) in enumerate(test_loader):
            # Move data to gpu
            vid_intervals = [j.float().cuda() for j in vid_intervals]
            sentences_emb = [j.float().cuda() for j in sentences_emb]
            sentences_indices = [j.cuda() for j in sentences_indices]
            ingredients_v = ingredients_v.float().cuda()
            # Get sizes of sentences in recipes
            sent_lens = [s.shape[1] for s in sentences_emb]

            # Prep word embeddings for sentences
            sentences_emb = [elt.squeeze(0) for elt in sentences_emb]
            sentences_emb = pad_sequence(sentences_emb, batch_first=True)

            # Get video encoder features for each sentence in recipe
            video_features = [encoder_video(j) for j in vid_intervals]
            video_features = torch.stack(video_features)
            video_features = video_features.permute(1, 0, 2) # (batch_size, num_sentences, hidden_dim)
            # Get lengths of recipes
            rec_lens = torch.tensor([video_features.shape[1]])
            
            # Get ingredient features using ingredient encoder
            ingredient_feats = encoder_ingredient(ingredients_v).unsqueeze(0)
            ingredient_feats = ingredient_feats.cuda()

            # Get recipe encoder output using features from video encoder
            recipe_enc = encoder_recipe(ingredient_feats, video_features, rec_lens, False)
            recipe_enc = recipe_enc.unsqueeze(0)
            
            # Get sentence decoder output
            sentence_dec = decoder_sentences(recipe_enc, sent_lens, sentences_emb)
            #_, predicted = sentence_dec.max(1) # Greedy decode
            predictions = beam_search_decoder(sentence_dec, 5)[0][0] # Beam search
            predicted = predictions[-1][0]

            # Construct ground truth from sentences
            sentences_indices = [elt.squeeze(0) for elt in sentences_indices]
            sentences_indices = torch.cat(sentences_indices, dim=0)

            predicted_sentence = decode_sentence(predicted, indx2vocab_dict)
            print("recipe ", name, " predicted sentence ", predicted_sentence)
            outputs.append(decode_sentence(predicted, indx2vocab_dict))
            gt.append(decode_sentence(sentences_indices, indx2vocab_dict))

    else:
        for i, (_, sentences_indices, sentences_emb, ingredients_v, name) in enumerate(tqdm(test_loader)):
            # Move data to gpu
            sentences_emb = [j.float().cuda() for j in sentences_emb]
            sentences_indices = [j.cuda() for j in sentences_indices]
            ingredients_v = ingredients_v.float().cuda()
            # Get sizes of sentences in recipes
            sent_lens = [s.shape[1] for s in sentences_emb]
            # Get lengths of recipes
            rec_lens = torch.tensor([len(sentences_emb)])

            # Prep word embeddings for sentences
            sentences_emb = [elt.squeeze(0) for elt in sentences_emb]
            sentences_emb = pad_sequence(sentences_emb, batch_first=True)

            # Get sentence encoder features
            sentence_features = encoder_sentences(sentences_emb, sent_lens)
            sentence_features = sentence_features.unsqueeze(0)
            
            # Get ingredient features using ingredient encoder
            ingredient_feats = encoder_ingredient(ingredients_v).unsqueeze(0)
            ingredient_feats = ingredient_feats.cuda()

            # Get recipe encoder output using features from video encoder
            recipe_enc = encoder_recipe(ingredient_feats, sentence_features, rec_lens, False)
            recipe_enc = recipe_enc.unsqueeze(0)
            
            # Get sentence decoder output
            sentence_dec = decoder_sentences(recipe_enc, sent_lens, sentences_emb)
            # _, predicted = sentence_dec.max(1) # Greedy approach 
            predictions = beam_search_decoder(sentence_dec, 5) # Beam search approach
            predicted = predictions[-1][0]

            # Construct ground truth from sentences
            sentences_indices = [elt.squeeze(0) for elt in sentences_indices]
            sentences_indices = torch.cat(sentences_indices, dim=0)

            predicted_sentence = decode_sentence(predicted, indx2vocab_dict)
            print("recipe ", name, " predicted sentence ", predicted_sentence)
            outputs.append(decode_sentence(predicted, indx2vocab_dict))
            gt.append(decode_sentence(sentences_indices, indx2vocab_dict))

    # Write outputs and ground truth to txt files with one sentence per line
    with open(os.path.join(results_path, 'outputs_{}.txt'.format(split)), 'w') as outputFile:
        for sent in outputs:
            outputFile.write(sent+'\n')
    with open(os.path.join(results_path, 'gt_{}.txt'.format(split)), 'w') as gtFile:
        for sent in gt:
            gtFile.write(sent+'\n')


def beam_search_decoder(data, k):
    """
    @param data (torch.Tensor) A sequence of distributions over the vocab (single recipe)
    @param k (int) Hyperparameter for number of non-repetitions
    """
    # Softmax the distributions
    soft_max = nn.Softmax(dim=1)
    data = soft_max(data)

    data = data.cpu().data.numpy()

    sequences = [[list(), 0.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score - log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    saved_model_folder = '/home/cristinam/cse538/project/NLPFinalProject/saved_models/train_joint_model/models_e1024_he512_hre1024_hd512_ep50_b1_l0_001'
    epochs = 5
    split = 'val'
    results_path = os.path.join(saved_model_folder, 'results')
    video = False

    # model parameters
    parser.add_argument('--vocab_len', type=int, default=12269, help='')
    parser.add_argument('--inredient_dim', type=int, default=3925, help='')
    parser.add_argument('--word_dim', type=int, default=256, help='')
    parser.add_argument('--sentEnd_hiddens', type=int, default=512, help='')
    parser.add_argument('--sentEnd_nlayers', type=int, default=1, help='')
    parser.add_argument('--recipe_inDim', type=int, default=1024, help='')
    parser.add_argument('--recipe_hiddens', type=int, default=1024, help='')
    parser.add_argument('--recipe_nlayers', type=int, default=1, help='')
    parser.add_argument('--sentDec_inDim', type=int, default=1024, help='')
    parser.add_argument('--sentDec_hiddens', type=int, default=512, help='')
    parser.add_argument('--sentDec_nlayers', type=int, default=1, help='')
    parser.add_argument('--sentences_sorted', type=int, default=1, help='')

    args = parser.parse_args()

    # Load dictionary mapping index to vocab string (decoded)
    with open('data/index_to_vocab.json', 'r') as vocabFile:
        index_to_vocab = json.load(vocabFile)

    # Generate results files with one recipe per line
    generate(args, saved_model_folder, epochs, split, results_path,
             index_to_vocab, video)

    # Calculate rouge score
    files_rouge = FilesRouge()
    outputs_path = os.path.join(results_path, 'outputs_{}.txt'.format(split))
    ref_path = os.path.join(results_path, 'gt_{}.txt'.format(split))
    scores = files_rouge.get_scores(outputs_path, ref_path, avg=True)
    print("ROUGE scores ", scores)

