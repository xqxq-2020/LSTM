#coding: UTF-8
import os
import argparse
import torch
from dataset import Dataset, config
from lstm import LSTM_Model
from lstm_crf import BiLSTM_CRF_model

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu[0]

model = LSTM_Model(args)

#"越权/vi  违规/vi  里森/nr  主管/vt  炒/vt  期货/n  资不抵债/l  [巴林/ns  银行/n]nt  告/vt  终结/vi  "

sentence = "越权 违规 里森 主管 炒 期货 资不抵债 巴林 银行 告 终结"
dataset_tool = Dataset(config)

if __name__ == '__main__':
	word_str = sentence.split()
	print(word_str)
	inputs = dataset_tool.prepare_sequence(word_str,dataset_tool.word_to_ix)
	inputs = torch.tensor(inputs, dtype=torch.long)
	inputs.cuda(0)
	score, pred_tag = model.lstm_crf_model(inputs)

	print(score,pred_tag)

