# Tagging-LSTM

## 介绍
NLP大作业：使用LSTM及CRF进行汉语的词类标注

## 环境

1.  PyTorch 1.0.1

2.  Python 3.7


## 使用说明

#### 1.  实验数据准备

使用北京大学人民日报语料库，已分词的txt文档被存放在 data 路径下, 其中 1998-01-2003_shuf.txt 为经过随机shuffle的数据。

#### 2.  单向LSTM网络模型训练

```python main.py --epoch 100 --checkpoint checkpoint --gpu 0 --seed 1```

参数含义：

--epoch: 训练epoch数

--checkpoint: 模型存储路径

--gpu: GPU序号

--seed: 模型初始化随机种子设置

--weighted_tag：计算损失函数时对类别加权 (可选)

默认为基于mini-batch的模型训练，若要修改batch size大小，可修改 dataset.py 中 Config 类的 BATCH_SIZE 参数。

#### 3.  双向LSTM网络模型

```python main.py --epoch 100 --checkpoint checkpoint --gpu 0 --seed 1 --bidirection```

参数含义：

--bidirection: 使用双向LSTM

默认为基于mini-batch的模型训练，若要修改batch size大小，可修改 dataset.py 中 Config 类的 BATCH_SIZE 参数。

#### 4.  BiLSTM-CRF模型

```python main.py --epoch 100 --checkpoint checkpoint --gpu 0 --seed 1 --bidirection --crf```

参数含义：

--crf: 使用crf层

未实现基于mini-batch的模型训练，训练速度较慢，但训练5个epoch后已经能达到 95% 以上的准确率

## 实验结果

| 模型         | 准确率    |
|------------|--------|
| 单向LSTM     | 88.38% |
| 双向LSTM     | 87.61% |
| BiLSTM-CRF | 95.49% |

## 参考

PyTorch官方代码 LSTM + CRF

