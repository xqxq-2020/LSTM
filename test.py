#coding: UTF-8
import os
import argparse
import torch
from dataset import Dataset, config
#from lstm import LSTM_Model
from lstm_crf import BiLSTM_CRF

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', default = [], nargs='+', type=str, help='Specify GPU id.')
parser.add_argument( '-e', '--epoch', type=int, help='train epoch number')
parser.add_argument( '--checkpoint', default = './', type=str, help='checkpoint.')
parser.add_argument( '--seed', default = 1, type=int, help='seed for pytorch init')
parser.add_argument( '-b','--bidirection', action='store_true', help='use bi-direction lstm or not')
parser.add_argument('--weighted_tag', action='store_true',help='use wighted loss or not' )
parser.add_argument('--crf', action='store_true',help='use crf or not' )

args = parser.parse_args()
print(args)

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu[0]

dataset_tool = Dataset(config)

model = BiLSTM_CRF(dataset_tool.word_to_ix_length,config.TAG_to_ix_crf,config.EMBEDDING_DIM,config.HIDDEN_DIM)
model.load_state_dict(torch.load(args.checkpoint+"/epoch_max_accuracy.pkl"))
print("load successful")

#"越权/vi  违规/vi  里森/nr  主管/vt  炒/vt  期货/n  资不抵债/l  [巴林/ns  银行/n]nt  告/vt  终结/vi  "

sentence = "越权 违规 里森 主管 炒 期货 资不抵债 巴林 银行 告 终结"

if __name__ == '__main__':
    word_str = sentence.split()
    print(word_str)
    inputs = dataset_tool.prepare_sequence(word_str,dataset_tool.word_to_ix)
    inputs = torch.tensor(inputs, dtype=torch.long)
    inputs.cuda(0)
    model.cuda(0)
    score, pred_tag = model.lstm_crf_model(inputs)

    print(score,pred_tag)

