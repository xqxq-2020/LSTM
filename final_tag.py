#coding: UTF-8
import time
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from net import LSTMTagger
from dataset import Dataset, config

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', default = [], nargs='+', type=str, help='Specify GPU id.')
parser.add_argument( '-e', '--epoch', type=int, help='train epoch number')
parser.add_argument( '--checkpoint', default = './', type=str, help='checkpoint.')
parser.add_argument( '--seed', default = 1, type=int, help='seed for pytorch init')
parser.add_argument( '-b','--bidirection', action='store_true', help='use bi-direction lstm or not')
args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)

class LSTM_Model(object):
    def __init__(self, args, lstmTagger = LSTMTagger, DataSet = Dataset, config = config, weighted_tag=False):
        self.config = config
        self.args = args
        self.epoch = self.args.epoch
        self.batch_size = self.config.BATCH_SIZE

        self.dataset = DataSet(self.config)
        self.training_data = self.dataset.training_data
        self.tag_weights = self.dataset.tag_weights
        self.test_data, self.test_tags = self.dataset.gene_testdata()
        self.word_to_ix_length = self.dataset.word_to_ix_length #词表长度

        self.max_accuracy = 0.

        if weighted_tag: #是否对类别加权
            self.loss_function = nn.NLLLoss(torch.tensor(self.tag_weights,dtype=torch.float))
        else:
            self.loss_function = nn.NLLLoss()

        self.lstm_model = lstmTagger(self.config.EMBEDDING_DIM, self.config.HIDDEN_DIM, self.word_to_ix_length,\
                          len(self.config.TAG_to_ix), self.config.LAYER, self.config.DROP_RATE, self.config.BATCH_SIZE,\
                          self.args.bidirection)
       

    #模型训练
    def train(self):
        self.lstm_model.train() # set mode
        #self.optimizer = optim.SGD(self.lstm_model.parameters(), lr=self.config.LR)
        #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.step, gamma=self.config.gamma)
        self.optimizer = optim.Adam(self.lstm_model.parameters(), lr=self.config.LR)
        # set gpu
        self.lstm_model.cuda(0)
        self.loss_function.cuda(0)

        for epoch in range(self.epoch):
            print("================= Epoch:",epoch,"/",self.epoch,"===============")
            start = time.clock()
            _iter,loss_total = 0,0

            while _iter < len(self.training_data):
                #try:
                    sentences_batch,tags_batch_list,lens_batch = self.dataset.gene_trainbatch(_iter) #（tensor 形式的sentens,tags）
                    # set gpu
                    sentences_batch = sentences_batch.cuda(0)
                    for i in range(len(tags_batch_list)):
                        tags_batch_list[i] = torch.tensor(tags_batch_list[i], dtype=torch.long)
                        tags_batch_list[i] = tags_batch_list[i].cuda(0)

                    #self.lr_scheduler.step() # lr
                    self.lstm_model.zero_grad()
                    
                    # input
                    tag_scores = self.lstm_model(sentences_batch, lens_batch)

                    # tags_batch_list = batch_size * 词个数 (对应词性的索引)
                    # tag_scores = batch_size * 词个数 * 26类
                    loss = 0
                    for i in range(min(self.batch_size,len(tags_batch_list))):
                        loss += self.loss_function(tag_scores[i][:lens_batch[i]], tags_batch_list[i])
                    loss_total += loss

                    if _iter % (100 * self.batch_size) == 0:
                        print("iter ",str(_iter), "loss:",loss_total/500,\
                                " lr:",self.optimizer.state_dict()['param_groups'][0]['lr'])
                        loss_total = 0
                        
                    _iter += self.batch_size
                    loss.backward()
                    self.optimizer.step()
                #except:
                #    print("Wrong!")
                #    continue

            # Test
            self.test()
            print("One epoch use time:",time.clock()-start)

    def test(self):
        print("******Testing...")
        with torch.no_grad(): # test mode
            num = 0
            total_word = 0
            test_labels = []
            test_predicts = []
            for inputs, targets in zip(self.test_data[:3850],self.test_tags[:3850]):
                #try:
                    total_word += len(inputs) #测试集的所有词数量
                    inputs = torch.tensor(inputs, dtype=torch.long)
                    targets = torch.tensor(targets, dtype=torch.long)
                
                    inputs = inputs.cuda(0)
                    targets = targets.cuda(0)

                    tag_scores = self.lstm_model(inputs,[0],is_test=True) # tensor N个词*tag数
                    tag_scores_numpy = tag_scores.cpu().numpy()
                
                    for idx,word in enumerate(tag_scores_numpy):
                        test_tag = np.where(word == np.max(word))
                        if test_tag[0][0] == int(targets[idx]):
                            num += 1
                        test_labels.append(int(targets[idx]))
                        test_predicts.append(test_tag[0][0])
                #except:
                #    print(item)
                #    continue

            # 评测
            accuracy = num / total_word
            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy
                # 计算混淆矩阵
                self.Confusion_matrix(test_labels, test_predicts)
                # save model
                torch.save(self.lstm_model.state_dict(),self.args.checkpoint+'/epoch_max_accuracy.pkl')
                print("Max accuracy's model is saved in ",self.args.checkpoint+'/epoch_max_accuracy.pkl')
            
            print("Acc:",accuracy,"(",num,'/',total_word,")","   Max acc so far:",self.max_accuracy) #单句的准确率
            
    def Confusion_matrix(self, test_labels, test_predicts):
        label_list = self.dataset.TAG_list_test
        cm = confusion_matrix(test_labels, test_predicts)
        cm = cm.astype(np.float32)
        sums = []
        for i in range(len(label_list)-1):#'x'类别没有
            sums.append(np.sum(cm[i]))
        
        for i in range(len(sums)):
            for j in range(len(sums)):
                cm[i][j]=round(float(cm[i][j])/float(sums[i]),2)#*100

        np.savetxt('Con_Matrix.txt', cm, fmt="%.2f", delimiter=',') #保存为2位小数的浮点数，用逗号分隔
        '''
        labels = [ i for i in self.config.TAG_to_ix.keys()] 
        labels[24] = labels[25]
        labels = labels[:-1]
        print(labels)
        sns.set()
        f,ax=plt.subplots()
        sns.heatmap(cm,annot=True,annot_kws={'size':4},ax=ax,cmap=plt.cm.Blues)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels, rotation=0)
        ax.tick_params(axis='y',labelsize=4)
        ax.tick_params(axis='x',labelsize=4)
        plt.savefig("Con_Matrix.png")
        '''
        print("The confusion matrix is saved in Con_Matrix.txt")

#if __name__ == '__main__':
# main

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu[0]

model = LSTM_Model(args)
model.train()
    


