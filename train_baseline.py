# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import socket
import argparse
from datetime import datetime
import wandb

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from models.sentence_encoder import SentenceEncoder
from models.video_encoder import VideoEncoder
from models.ingredient_encoder import IngredientEncoder
from models.sentence_decoder import SentenceDecoder
from models.recipe_encoder import RecipeEncoder
from datasets.video_datasets import TastyVideoDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

COMP_PATH = '/home/cristinam/cse538/project/NLPFinalProject'


def train(args, name_repo):
    timestamp = name_repo  # unique identifier with descriptive name
    # current timestamp with second precision to avoid overwriting old stuff
    timestamp += datetime.now().strftime("%y%m%d_%H%M%S_")
    # add computer name to allow easier finding data scattered over multiple computers
    timestamp += socket.gethostname()

    out_dir = args.model_path

    # Build the models
    encoder_recipe = RecipeEncoder(args).to(device)
    encoder_ingredient = IngredientEncoder(1024, 3925)
    decoder_sentences = SentenceDecoder(args).to(device)
    encoder_sentences = SentenceEncoder(args).to(device)

    # Loss and optimizer
    criterion_sent = nn.CrossEntropyLoss()
    params = list(encoder_ingredient.parameters()) + \
             list(encoder_sentences.parameters()) + \
             list(encoder_recipe.parameters()) + \
             list(decoder_sentences.parameters())
             
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Build data loader
    tasty_video_dataset = TastyVideoDataset(split='train', video=False)
    train_loader = torch.utils.data.DataLoader(dataset=tasty_video_dataset,
                                              batch_size=1,
                                              shuffle=True)

    # Train the models
    use_teacherF = False
    total_step = len(train_loader)
    for epoch in range(args.num_epochs):

        for i, (_, sentences_indices, sentences_emb, ingredients_v, recipe_name) in enumerate(train_loader):

            sentences_emb = [j.float().cuda() for j in sentences_emb]
            # Get sizes of sentences in recipes
            sent_lens = [sentences_emb[i].shape[1] for i in range(len(sentences_emb))]
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

            # Get recipe encoder output using features from sentence encoder
            if epoch >= 5:
                use_teacherF = random.random() < (0.5)
            recipe_enc = encoder_recipe(ingredient_feats, sentence_features, rec_lens, use_teacherF)
            recipe_enc = recipe_enc.unsqueeze(0)
            
            # Get sentence decoder output
            sentence_dec = decoder_sentences(recipe_enc, sent_lens, sentences_emb)
            
            # Construct ground truth from sentences
            sentences_indices = [elt.squeeze(0) for elt in sentences_indices]
            sentences_indices = pad_sequence(sentences_indices, batch_first=True)
            sentence_target = pack_padded_sequence(sentences_indices, sent_lens, batch_first=True, enforce_sorted=False)[0]
            sentence_target = sentence_target.type(torch.LongTensor)
            sentence_target = sentence_target.cuda()

            # Calculate losses
            all_loss = criterion_sent(sentence_dec, sentence_target)

            encoder_sentences.zero_grad()
            encoder_recipe.zero_grad()
            decoder_sentences.zero_grad()
            
            all_loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.10f} '.format(epoch, args.num_epochs, i, total_step,
                                                                           all_loss.item()))
            wandb.log({"Decoder Loss ": all_loss.item()})

            del all_loss

        if (epoch + 1) % 5 == 0:  # Save the model checkpoints
            save_models(args, (encoder_recipe, encoder_ingredient, encoder_sentences, decoder_sentences),
                        epoch + 1)
        
    # Save the final models
    save_models(args, (encoder_recipe, encoder_ingredient, encoder_sentences, decoder_sentences),
                epoch + 1)


def save_models(args, all_models, epoch_val):
    if epoch_val == 0:
        num_epochs = ''
    else:
        num_epochs = '-' + str(epoch_val)

    (encoder_recipe, encoder_ingredient, encoder_sentences, decoder_sentences) = all_models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.save(encoder_recipe.state_dict(),
               os.path.join(args.model_path, 'encoder_recipe{}.ckpt'.format(num_epochs)))
    torch.save(encoder_ingredient.state_dict(),
               os.path.join(args.model_path, 'encoder_ingredient{}.ckpt'.format(num_epochs)))
    torch.save(decoder_sentences.state_dict(),
               os.path.join(args.model_path, 'decoder_sentences{}.ckpt'.format(num_epochs)))
    torch.save(encoder_sentences.state_dict(),
               os.path.join(args.model_path, 'encoder_sentences{}.ckpt'.format(num_epochs)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    name_repo = 'train_baseline'
    wandb.init(project="cse538-project")

    intermediate_fd = '/home/cristinam/cse538/project/NLPFinalProject/saved_models/'+name_repo+'/'

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

    # training parameters
    parser.add_argument('--log_step', type=int, default=20, help='step size for printing log info')
    parser.add_argument('--save_step', type=int, default=45, help='step size for saving trained models')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()
    param_all = ('e' + str(args.recipe_inDim) + '_he' + str(args.sentEnd_hiddens) + '_hre' + str(
        args.recipe_hiddens) + '_hd' + str(args.sentDec_hiddens) + '_ep' + str(args.num_epochs) + '_b' + str(
        args.batch_size) + '_l' + str(args.learning_rate).replace(".", "_"))
    parser.add_argument('--model_path', type=str, default=(intermediate_fd + 'models_' + param_all + '/'),
                        help='path for saving trained models')
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    train(args, name_repo)

