import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import os
import pdb
from tqdm.contrib import tzip
from rouge import FilesRouge


def print_out_rouge_score(predicted_path, expected_path):

    files_rouge = FilesRouge()
    scores = files_rouge.get_scores(predicted_path, expected_path, avg=True)
    print("ROUGE scores ", scores)
    
    return 0


def print_out_bleu_and_meteor_score(predicted_path, expected_path):

    scores = [('BLEU SCORE-1: ', []), ('BLEU SCORE-2: ', []), ('BLEU SCORE-3: ', []), ('BLEU SCORE-4: ', []), ('METEOR SCORE: ', [])]

    with open(predicted_path, 'r') as fp_pred, open(expected_path, 'r') as fp_exp:
        for prediction, expected in tzip(fp_pred, fp_exp): 
            prediction = prediction.split(' ')
            expected_list = expected.split(' ')

            scores[0][1].append(sentence_bleu(prediction, expected_list, weights=(1, 0, 0, 0)))
            scores[1][1].append(sentence_bleu(prediction, expected_list, weights=(0, 1, 0, 0)))
            scores[2][1].append(sentence_bleu(prediction, expected_list, weights=(0, 0, 1, 0)))
            scores[3][1].append(sentence_bleu(prediction, expected_list, weights=(0, 0, 0, 1)))
            scores[4][1].append(meteor_score(prediction, expected))

    for score in scores:
        print(score[0] + str(sum(score[1]) / len(score[1])))

    return 0


def main():

    path_to_results =\
        "/mnt/disks/nlp-small/new_saved_models/" +\
        "train_baseline/models_e1024_he512_hre1024_hd512_ep50_b1_l0_001/results"

    predicted_path = os.path.join(path_to_results, 'outputs_val.txt')
    expected_path = os.path.join(path_to_results, 'gt_val.txt')

    print_out_rouge_score(predicted_path, expected_path)
    # print_out_bleu_and_meteor_score(predicted_path, expected_path)

    return 0


main()
