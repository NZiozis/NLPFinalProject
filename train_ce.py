# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import os
from datasets.data_loader import get_loader
from models.model_ce import EncoderINGREDIENT, EncoderRECIPE, DecoderSENTENCES
from models.model_ce import BLSTMprojEncoder, SP_EMBEDDING
from models.video_encoder import VideoEncoder
from models.ingredient_encoder import IngredientEncoder
from models.sentence_decoder import SentenceDecoder
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle
from torch.autograd import Variable
import random
import numpy as np
from datetime import datetime
import socket
import argparse
from datasets.Vocabulary import Vocabulary

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
    embed_words = SP_EMBEDDING(args).to(device)
    encoder_recipe = EncoderRECIPE(args).to(device)
    
    if args.video_encoder:
        # Build video encoder
        encoder_video = VideoEncoder(args.sentEnd_hiddens).to(device)
        encoder_sentences = None
        encoder_ingredient = IngredientEncoder(1024, 3925)
        decoder_sentences = SentenceDecoder(args)
    else:
        encoder_sentences = BLSTMprojEncoder(args).to(device)
        encoder_ingredient = EncoderINGREDIENT(args).to(device)
        decoder_sentences = DecoderSENTENCES(args).to(device)
        encoder_video = None

    # Loss and optimizer
    criterion_sent = nn.CrossEntropyLoss()
    params = list(embed_words.parameters()) + \
             list(encoder_recipe.parameters()) + \
             list(decoder_sentences.parameters()) + \
             list(encoder_ingredient.parameters())

    if args.video_encoder:
        params = params + list(encoder_video.parameters())
    else:
        params = params + list(encoder_sentences.parameters())

    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Build data loader
    with open(args.vocab_bin, 'rb') as f:
        vocab = pickle.load(f, encoding='latin1')
    train_loader = get_loader(args, args.batch_size, vocab, shuffle=True, num_workers=args.num_workers)

    # Train the models
    use_teacherF = False
    total_step = len(train_loader)
    for epoch in range(args.num_epochs):
        epoch_loss_all = 0

        # Case where we just have sentence encoder
        if not args.video_encoder:
            # Set mini-batch dataset
            for i, (ingredients_v, rec_lens, sentences_v, sent_lens, indices, indices_encoder) in enumerate(train_loader):
                ingredients_v = ingredients_v.to(device)  # [N, Nv] -> Nv = ingredient vocab. len
                sentences_v = sentences_v.to(device)  # [Nb, Ns] -> [total num sent, max sent len.]
                sent_lens = sent_lens.to(device)  # Nb-> total sent. num, max

                """ 1. encode sentences """
                word_embs = embed_words(sentences_v)  # [Nb, Ns, 256]
                sentence_enc = encoder_sentences(word_embs, sent_lens)  # [Nb, 1024]

                """ reshape sentences wrt the recipe order """
                # sort the indices
                _, orgj_idx = indices_encoder.sort(0, descending=False)  # [Nb]
                orgj_idx = Variable(orgj_idx).cuda()  # [Nb]

                # permute sentence according to instructional order within a batch
                sentence_enc = sentence_enc.index_select(0, orgj_idx)  # [Nb, 1024]

                # split according to the batch, note that the batch is ordered
                sentence_enc_spl = torch.split(sentence_enc, rec_lens, dim=0)

                # pack and pad the ordered sentences
                recipes_v_pckd = pack_sequence(sentence_enc_spl)
                recipes_v = pad_packed_sequence(recipes_v_pckd, batch_first=True)[0]  # [N, rec_lens[0], 1024]

                """ 2. encode ingredient """
                ingredient_feats = encoder_ingredient(ingredients_v).unsqueeze(1)  # [N, 1, 1024]

                """ 3. encode recipe """
                if epoch >= 5:
                    use_teacherF = random.random() < (0.5)
                recipe_enc = encoder_recipe(ingredient_feats, recipes_v, rec_lens, use_teacherF)  # [Nb, 1024]

                """ 4. decode sentences """
                idx = Variable(indices).cuda()
                recipe_enc = recipe_enc.index_select(0, idx)  # [Nb, 1024]

                sentence_dec = decoder_sentences(recipe_enc, word_embs, sent_lens)
                # [sum(sent_lens), Nw] -- Nw = number of words in the vocabulary
                print("sentence lens ", sent_lens)
                print("sentences_v shape ", sentences_v.shape)
                print("sentences_v ", sentences_v)
                sentence_target = pack_padded_sequence(sentences_v, sent_lens, batch_first=True)[0]  # [ sum(sent_lens) ]
                print("sentence dec ", sentence_dec.shape)
                print("sentence target ", sentence_target.shape)
                print("sentence target ", sentence_target)
                """ Compute the loss """
                all_loss = criterion_sent(sentence_dec, sentence_target)
                epoch_loss_all += all_loss

                """ Backpropagation """
                encoder_recipe.zero_grad()
                encoder_ingredient.zero_grad()
                decoder_sentences.zero_grad()
                encoder_sentences.zero_grad()
                embed_words.zero_grad()

                all_loss.backward()
                optimizer.step()

                """ Printing and evaluations """
                if i % args.log_step == 0:  # Print log info
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.10f} '.format(epoch, args.num_epochs, i, total_step,
                                                                               all_loss.item()))

                if (i + 1) % args.save_step == 0:  # Print sentences
                    generate(sentences_v[0, :], vocab, recipe_enc, decoder_sentences, embed_words)

            if (epoch + 1) % 5 == 0:  # Save the model checkpoints
                save_models(args, (encoder_recipe, encoder_ingredient, decoder_sentences, encoder_sentences, embed_words),
                            epoch + 1)
        
        # Case where we just have the video encoder
        else:
            for i, (vid_intervals, sentences_v, ingredients_v) in enumerate(train_loader):
                # Get video encoder features for each sentence in recipe
                vid_intervals = [j.float().cuda() for j in vid_intervals]

                recipes_v = [encoder_video(j) for j in vid_intervals]
                recipes_v = torch.stack(recipes_v)
                recipes_v = recipes_v.permute(1, 0, 2) # (batch_size, num_sentences, hidden_dim)

                # Get ingredient features using ingredient encoder
                ingredient_feats = encoder_ingredient(ingredients_v).unsqueeze(0)
                ingredient_feats = ingredient_feats.cuda()
                rec_lens = torch.tensor([recipes_v.shape[1]])
                sent_lens = torch.tensor([recipes_v.shape[i][0] for i in range(len(recipes_v))]).cuda()

                if epoch >= 5:
                    use_teacherF = random.random() < (0.5)
                recipe_enc = encoder_recipe(ingredient_feats, recipes_v, rec_lens, use_teacherF)  # [Nb, 1024]

                sentence_dec = decoder_sentences(recipe_enc, sent_lens)
                # [sum(sent_lens), Nw] -- Nw = number of words in the vocabulary

                sentence_target = pack_padded_sequence(sentences_v, sent_lens, batch_first=True)[0]  # [ sum(sent_lens) ]

                all_loss = criterion_sent(sentence_dec, sentence_target)
                epoch_loss_all += all_loss

                encoder_recipe.zero_grad()
                decoder_sentences.zero_grad()
                encoder_video.zero_grad()
                embed_words.zero_grad()

                all_loss.backward()
                optimizer.step()

                if i % args.log_step == 0:  # Print log info
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.10f} '.format(epoch, args.num_epochs, i, total_step,
                                                                               all_loss.item()))
            if (epoch + 1) % 5 == 0:  # Save the model checkpoints
                save_models(args, (encoder_recipe, decoder_sentences, encoder_video, embed_words),
                            epoch + 1)
        
    if args.video_encoder:
        # Save the final models
        save_models(args, (encoder_recipe, decoder_sentences, encoder_video, embed_words),
                    epoch + 1)
    else:
        save_models(args, (encoder_recipe, encoder_ingredient, decoder_sentences, encoder_sentences, embed_words),
                    epoch + 1)


def save_models(args, all_models, epoch_val):
    if epoch_val == 0:
        num_epochs = ''
    else:
        num_epochs = '-' + str(epoch_val)

    (encoder_recipe, encoder_ingredient, decoder_sentences, encoder_sentences, embed_words) = all_models
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
    torch.save(embed_words.state_dict(),
               os.path.join(args.model_path, 'embed_words{}.ckpt'.format(num_epochs)))


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
    #name_repo = 'retrain_video_enc'
    name_repo = 'retrain_sentence_enc'

    # TODO: Move DATA to same folder as code
    intermediate_fd = '/home/cristinam/cse538/project/saved_models/'+name_repo+'/'
    json_fd = '/home/cristinam/cse538/project/DATA/Recipe1M/'
    vocab_fd = '/home/cristinam/cse538/project/DATA/vocab/'
    #intermediate_fd = os.path.join(COMP_PATH, 'INTERMEDIATE/' + name_repo + '/')
    #json_fd = os.path.join(COMP_PATH, 'DATA/Recipe1M/')
    #vocab_fd = os.path.join(COMP_PATH, 'DATA/vocab/')

    parser.add_argument('--json_joint', type=str, default=(json_fd + 'layer1_joint.json'), help='path for annotations')
    parser.add_argument('--vocab_bin', type=str, default=(vocab_fd + 'vocab_bin_30171.pkl'), help='')
    parser.add_argument('--vocab_ing', type=str, default=(vocab_fd + 'vocab_ing_3769.pkl'), help='')

    # model parameters
    parser.add_argument('--vocab_len', type=int, default=30171, help='')
    parser.add_argument('--inredient_dim', type=int, default=3769, help='')
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
    parser.add_argument('--video_encoder', type=bool, default=False)

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

