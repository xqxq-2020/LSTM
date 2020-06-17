# Tagging-LSTM

## 介绍
NLP大作业：使用LSTM及CRF进行汉语的词类标注

## 环境

1.  PyTorch 1.0.1

2.  Python 3.7

3.  Numpy, sklearn, matplotlib, seaborn


## 使用说明

#### 1.  实验数据准备

使用北京大学人民日报语料库，已分词的txt文档被存放在 data 路径下, 其中 1998-01-2003_shuf.txt 为经过随机shuffle的数据。


#### 2.  系统使用

加载checkpoint路径中训练好的模型，对输入语句进行测试。

- BiLSTM_CRF模型

```python test.py --bidirection --crf --checkpoint checkpoint_bilstm_crf/ --gpu 0 ```


- BiLSTM模型

```python test.py --bidirection --checkpoint checkpoint_bilstm/ --gpu 0 ```

- LSTM模型

```python test.py --checkpoint checkpoint_lstm/ --gpu 0 ```

默认输出测试语句 "越权 违规 里森 主管 炒 期货 资不抵债 巴林 银行 告 终结" 的词类标注结果:

Result: ['n', 'v', 'n', 'v', 'v', 'n', 'v', 'n', 'n', 'v', 'v']

注：名词(n)、时间词(t)、处所词(s)、方位词(f)、数词(m)、量词(q)、区别词(b)、代词( r)、动词(v)、形容词( a)、状态词(z)、副词(d)、介词( p)、连词( c)、助词(u)、语气词(y)、叹词(e)、拟声词(o)、成语(i)、习用语(l)、简称(j)、前接成分(h)、后接成分(k)、语素(g)、非语素字(x)、标点符号(w)。


#### 2.  单向LSTM网络模型训练

```python main.py --epoch 100 --checkpoint checkpoint_lstm --gpu 0 --seed 1```

参数含义：

--epoch: 训练epoch数

--checkpoint: 模型存储路径

--gpu: GPU序号

--seed: 模型初始化随机种子设置

--weighted_tag：计算损失函数时对类别加权 (可选)

默认为基于mini-batch的模型训练，若要修改batch size大小，可修改 dataset.py 中 Config 类的 BATCH_SIZE 参数。


#### 3.  双向LSTM网络模型

```python main.py --epoch 100 --checkpoint checkpoint_bilstm --gpu 0 --seed 1 --bidirection```

参数含义：

--bidirection: 使用双向LSTM

默认为基于mini-batch的模型训练，若要修改batch size大小，可修改 dataset.py 中 Config 类的 BATCH_SIZE 参数。


#### 4.  BiLSTM-CRF模型

```python main.py --epoch 100 --checkpoint checkpoint_bilstm_crf --gpu 0 --seed 1 --bidirection --crf```

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

