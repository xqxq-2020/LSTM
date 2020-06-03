#coding: UTF-8
import os
import argparse
import torch
from dataset import Dataset, config
from lstm import LSTMTagger
from lstm_crf import BiLSTM_CRF
import numpy as np
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', default = [], nargs='+', type=str, help='Specify GPU id.')
parser.add_argument( '-e', '--epoch', default = 1, type=int, help='train epoch number')
parser.add_argument( '--checkpoint', default = 'checkpoint/', type=str, help='checkpoint.')
parser.add_argument( '--seed', default = 1, type=int, help='seed for pytorch init')
parser.add_argument( '-b','--bidirection', action='store_true', help='use bi-direction lstm or not')
parser.add_argument('--weighted_tag', action='store_true',help='use wighted loss or not' )
parser.add_argument('--crf', action='store_true',help='use crf or not' )

args = parser.parse_args()

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu[0]

dataset_tool = Dataset(config)

if args.crf:
    print("BiLSTM_CRF model")
    model = BiLSTM_CRF(dataset_tool.word_to_ix_length,config.TAG_to_ix_crf,config.EMBEDDING_DIM,config.HIDDEN_DIM)
else:
    print("BiLSTM / LSTM model")
    model = LSTMTagger(config.EMBEDDING_DIM, config.HIDDEN_DIM, dataset_tool.word_to_ix_length,\
                          len(config.TAG_to_ix), config.LAYER, config.DROP_RATE, 1, args.bidirection)

model.load_state_dict(torch.load(args.checkpoint+"/epoch_max_accuracy.pkl"))
print("Loading model...")

sentence = "越权 违规 里森 主管 炒 期货 资不抵债 巴林 银行 告 终结"
gt = ['v','v','n','v','v','n','l','n','n','v','v']

if __name__ == '__main__':
    word_str = sentence.split()
    print("Sentence:\n",sentence)
    print("Ground Truth:")
    print(gt)
    inputs = dataset_tool.prepare_sequence(word_str,dataset_tool.word_to_ix)
    inputs = torch.tensor(inputs, dtype=torch.long)
    inputs = inputs.cuda(0)
    model = model.cuda(0)
    if args.crf:
        with torch.no_grad():
            score, pred_tag = model(inputs)
        print("Result:")
        result = []
        for item in pred_tag:
            result.append(list(config.TAG_to_ix.keys())[item])
        print(result)
        print("score:",score.cpu().numpy())
    else:
        with torch.no_grad():
            result = []
            score = model(inputs,[],is_test=True)
            for word in score.cpu().numpy():
                pred = np.where(word == np.max(word))
                result.append(list(config.TAG_to_ix.keys())[pred[0][0]])
            print("Result:")
            print(result)
