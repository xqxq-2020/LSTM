import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from dataset import Dataset, config
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# 返回vec中每一行最大的那个元素的下标
def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()  # 注：tensor只有一个元素才能调用item方法

def prepare_sequence(seq, to_ix):  # 单词转为索引
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def log_sum_exp(vec):  # 计算一维向量vec与其最大值的log_sum_exp
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    # 减去最大值是为了防止数值溢出
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

START_TAG = "<START>"
STOP_TAG = "<STOP>"

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim  # 嵌入维度
        self.hidden_dim = hidden_dim  # 隐藏层维度
        self.vocab_size = vocab_size  # 词汇大小
        self.tag_to_ix = tag_to_ix  # 标签转为下标
        self.tagset_size = len(tag_to_ix)  # 目标取值范围大小

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)  # 嵌入层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)  # 双向LSTM
        
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)  # LSTM的输出映射到标签空间

        # 转移矩阵的参数tag-->tag
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size)).cuda(0)
        # 限制不能转移到start，end不能转移到其他
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000


    def _forward_alg(self, feats):  # 前向算法，feats是LSTM所有时间步的输出
        init_alphas = torch.full((1, self.tagset_size), -10000.)  # alpha初始为-10000
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.  # start位置的alpha为0

        init_alphas = init_alphas.cuda(0)
        forward_var = init_alphas  # 包装进forward_var变量，以便于自动反向传播

        for feat in feats:  # 对于每个时间步，进行前向计算
            alphas_t = []  # 当前时间步i的前向tensor
            for next_tag in range(self.tagset_size):  # next_tag有target_size个可能的取值
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))

            forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]  # 添加stop_tag
        alpha = log_sum_exp(terminal_var)
        return alpha  # 每个时间步的得分（预测分数）


    def _get_lstm_features(self, sentence):  # LSTM的输出
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats


    def _score_sentence(self, feats, tags):
        # 计算给定标签序列的分数---发射分数+转移分数（真实分数）
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).cuda(0), tags])
        score = score.cuda(0)
        tags = tags.cuda(0)
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score


    def _viterbi_decode(self, feats):  # viterbi算法
        backpointers = []

        # 在log空间初始化维特比变量
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        
        init_vvars = init_vvars.cuda(0)
        # 时间步i的forward_var拥有时间步i-1的维特比变量
        forward_var = init_vvars
        for feat in feats:  # 对于每个时间步
            bptrs_t = []  # 存放当前步的backpointer
            viterbivars_t = []  # 存放当前步的维特比变量

            for next_tag in range(self.tagset_size):
                # next_tag_var[i]包含之前时间步tag i的维特比变量和tag i 到next_tag的转移分数
                # 没有包含发射分数是因为最大值不取决于它（之后添加了它们）
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)  # backpointer存放前一步最好的取值

        # 转移到STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # 根据back pointers解码最好的路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # 断言检查
        best_path.reverse()
        return path_score, best_path


    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)  # LSTM输出
        forward_score = self._forward_alg(feats)  # 前向计算分数
        gold_score = self._score_sentence(feats, tags)  # 真实分数
        # logP(y|x) = gold_score - forward_score
        return forward_score - gold_score


    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)  # 获得BiLSTM的发射分数
        score, tag_seq = self._viterbi_decode(lstm_feats)  # 给定特征，寻找最好的路径
        return score, tag_seq

class BiLSTM_CRF_model(object):
    def __init__(self, args, lstm_crf = BiLSTM_CRF, DataSet = Dataset, config = config, weighted_tag=False):
        self.config = config
        self.args = args
        self.dataset = Dataset(self.config)
        self.lstm_crf_model = lstm_crf(self.dataset.word_to_ix_length, self.config.TAG_to_ix_crf, \
                                self.config.EMBEDDING_DIM,self.config.HIDDEN_DIM)

        self.optimizer = optim.SGD(self.lstm_crf_model.parameters(), lr=self.config.LR, weight_decay=self.config.weight_decay)

        self.train_data, self.train_tags = self.dataset.gene_traindata()
        self.test_data, self.test_tags = self.dataset.gene_testdata()

        self.epoch = self.args.epoch
        self.max_accuracy = 0.

    def train(self):
        self.lstm_crf_model.train()
        self.lstm_crf_model.cuda(0)

        for i in range(self.epoch):
            print("----------- Epoch:",i,"-----------")
            start = time.clock()
            _iter = 0
            loss_total = 0
            for sentence, target in zip(self.train_data, self.train_tags):
                sentence = torch.tensor(sentence, dtype=torch.long)
                target = torch.tensor(target, dtype=torch.long)
                
                sentence = sentence.cuda(0)
                target = target.cuda(0)

                self.lstm_crf_model.zero_grad()
                loss = self.lstm_crf_model.neg_log_likelihood(sentence, target)
                loss.backward()
                self.optimizer.step()
                
                loss_total += loss
                if _iter % 100==0:
                    print("Iter ",_iter,"   loss:",loss_total/100)
                    loss_total = 0

                _iter += 1
            print("Time use:",time.clock()-start)
            self.test()

    def test(self):
        print("**********Testing...")
        self.lstm_crf_model.load_state_dict(torch.load("crf_log_all/epoch_max_accuracy.pkl"))
        print("load successful")
        with torch.no_grad():
            num = 0
            total_word = 0
            test_labels = []
            test_predicts = []

            self.lstm_crf_model.cuda(0)
            for inputs, targets in zip(self.test_data[:3850], self.test_tags[:3850]):
                total_word += len(inputs)

                inputs = torch.tensor(inputs, dtype=torch.long)
                
                inputs = inputs.cuda(0)
                
                score, pred_tag = self.lstm_crf_model(inputs)
                for idx, word in enumerate(pred_tag):
                    if word == targets[idx]:
                        num += 1
                    test_labels.append(targets[idx])
                    test_predicts.append(pred_tag[idx])

            accuracy = num / total_word
            if accuracy >= self.max_accuracy:
                self.max_accuracy = accuracy
                # 计算混淆矩阵
                self.Confusion_matrix(test_labels, test_predicts)
                # save model
                torch.save(self.lstm_crf_model.state_dict(),self.args.checkpoint+'/epoch_max_accuracy.pkl')
                print("Max accuracy's model is saved in ",self.args.checkpoint+'/epoch_max_accuracy.pkl')
            
            self.Confusion_matrix(test_labels, test_predicts)
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
                cm[i][j]=round(float(cm[i][j])/float(sums[i]),2)

        np.savetxt(self.args.checkpoint+'/Con_Matrix.txt', cm, fmt="%.2f", delimiter=',') #保存为2位小数的浮点数，用逗号分隔
        print("The confusion matrix is saved in "+self.args.checkpoint+"/Con_Matrix.txt")




