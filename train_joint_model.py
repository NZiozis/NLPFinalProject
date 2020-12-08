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
    encoder_video = VideoEncoder(args.sentEnd_hiddens).to(device)
    encoder_ingredient = IngredientEncoder(1024, 3925).to(device)
    decoder_sentences = SentenceDecoder(args).to(device)
    encoder_sentences = SentenceEncoder(args).to(device)

    # Loss and optimizer
    criterion_sent = nn.CrossEntropyLoss()
    criterion_latent = nn.CosineEmbeddingLoss()
    params = list(encoder_ingredient.parameters()) + \
             list(encoder_video.parameters()) + \
             list(encoder_sentences.parameters()) + \
             list(encoder_recipe.parameters()) + \
             list(decoder_sentences.parameters())
             
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Build data loader
    tasty_video_dataset = TastyVideoDataset(split='train', video=True)
    train_loader = torch.utils.data.DataLoader(dataset=tasty_video_dataset,
                                              batch_size=1,
                                              shuffle=False)
    # Train the models
    use_teacherF = False
    total_step = len(train_loader)
    for epoch in range(args.num_epochs):

        for i, (vid_intervals, sentences_indices, sentences_emb, ingredients_v, name) in enumerate(train_loader):
            print(torch.cuda.memory_allocated(0))
            print(name)
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

            # Get sentence encoder features
            sentence_features = encoder_sentences(sentences_emb, sent_lens)
            sentence_features = sentence_features.unsqueeze(0)
            if len(sentence_features.shape) == 2:
                sentence_features = sentence_features.unsqueeze(0)

            # Calculate embedding loss
            # When y = 1 this tells Cosine loss that embeddings should be similar
            y = torch.Tensor([1]).cuda()
            embedding_loss = criterion_latent(video_features, sentence_features, y)
            video_features = []
            
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
            sentence_target = torch.cat(sentences_indices, dim=0)
            sentence_target = sentence_target.type(torch.LongTensor)
            sentence_target = sentence_target.cuda()

            # Calculate cross entropy loss
            sentence_loss = criterion_sent(sentence_dec, sentence_target)
            total_loss = sentence_loss + embedding_loss

            
            total_loss.backward()
            optimizer.step()

            encoder_video.zero_grad()
            encoder_sentences.zero_grad()
            encoder_recipe.zero_grad()
            decoder_sentences.zero_grad()

            if i % args.log_step == 0:  # Print log info
                print('Epoch [{}/{}], Step [{}/{}], Decoder Loss: {:.10f}, Embedding Loss: {:.10f}'.format(epoch, args.num_epochs, i, total_step,
                                                                           sentence_loss.item(), embedding_loss.item()))
            #wandb.log({"Decoder Loss ": sentence_loss.item(), "Embedding Loss": embedding_loss.item()})

            # Free memory before next loop
            vid_intervals, sentences_emb, sentences_indices, ingredients_v, \
            ingredient_feats, sentence_features, recipe_enc,  \
            sentence_dec, sentence_target = [], [], [], [], [], [], [], [], []
            del total_loss, sentence_loss, embedding_loss, sentence_dec, recipe_enc, video_features, ingredient_feats

            #print('current memory allocated: {}'.format(torch.cuda.memory_cached() / 1024 ** 2))
            #print('max memory allocated: {}'.format(torch.cuda.max_memory_allocated() / 1024 ** 2))
            #print('cached memory: {}'.format(torch.cuda.memory_cached() / 1024 ** 2))

            torch.cuda.empty_cache()

            #print("After emptying cache:")
            #print('current memory allocated: {}'.format(torch.cuda.memory_cached() / 1024 ** 2))
            #print('max memory allocated: {}'.format(torch.cuda.max_memory_allocated() / 1024 ** 2))
            #print('cached memory: {}'.format(torch.cuda.memory_cached() / 1024 ** 2))

            
        if (epoch + 1) % 5 == 0:  # Save the model checkpoints
            save_models(args, (encoder_recipe, encoder_ingredient, encoder_sentences, decoder_sentences, encoder_video),
                        epoch + 1)
        
    # Save the final models
    save_models(args, (encoder_recipe, encoder_ingredient, encoder_sentences, decoder_sentences, encoder_video),
                epoch + 1)


def save_models(args, all_models, epoch_val):
    if epoch_val == 0:
        num_epochs = ''
    else:
        num_epochs = '-' + str(epoch_val)

    (encoder_recipe, encoder_ingredient, encoder_sentences, decoder_sentences, encoder_video) = all_models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.save(encoder_recipe.state_dict(),
               os.path.join(args.model_path, 'encoder_recipe{}.ckpt'.format(num_epochs)))
    torch.save(encoder_ingredient.state_dict(),
               os.path.join(args.model_path, 'encoder_ingredient{}.ckpt'.format(num_epochs)))
    torch.save(encoder_video.state_dict(),
               os.path.join(args.model_path, 'encoder_video{}.ckpt'.format(num_epochs)))
    torch.save(decoder_sentences.state_dict(),
               os.path.join(args.model_path, 'decoder_sentences{}.ckpt'.format(num_epochs)))
    torch.save(encoder_sentences.state_dict(),
               os.path.join(args.model_path, 'encoder_sentences{}.ckpt'.format(num_epochs)))


def generate(sentences_v, vocab, recipe_enc, decoder_sentences, embed_words):
    target_sentence = ids2words(vocab, sentences_v.cpu().numpy())

    recipe_enc_gen = recipe_enc[0, :].view(1, -1)
    pred_ids = decoder_sentences.sample(recipe_enc_gen, embed_words)
    pred_sentence = ids2words(vocab, pred_ids[0].cpu().numpy())

    print('gt   : ', target_sentence)
    print('pred : ', pred_sentence)


def ids2words(vocab, target_ids):
    target_caption = []
    for word_id in target_ids:
        word = vocab.idx2word[word_id]
        if word == '<start>':
            continue
        if word == '<end>':
            break
        target_caption.append(word)
    target_sentence = ' '.join(target_caption)
    return target_sentence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    name_repo = 'train_joint_model'
    #wandb.init(project="cse538-project")

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
    parser.add_argument('--num_workers', type=int, default=1)
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

