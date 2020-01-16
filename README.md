[TOC]
### 1、项目描述
https://www.biendata.com/competition/falsenews/
task1：虚假新闻识别、文本分类任务

### 2、Requirements
```
python  3.7
pytorch 1.1
tensorboardx
numpy
torchvision
torchtext
transformers # pytorch加载bert工具, 并提供bert预训练转换工具
```
### 3、训练
1、数据：
链接: https://pan.baidu.com/s/1ZE_V_bjyJIFpWqHXfhmh8w 提取码: r3dv 

1、下载bert预训练模型：pytorch版本：

* [中文预训练BERT-wwm（Pre-Trained Chinese BERT with Whole Word Masking）](https://github.com/ymcui/Chinese-BERT-wwm#%E4%B8%AD%E6%96%87%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD)：直接提供pytorch版本

将 vocab.txt pytorch_model.bin config.json 放到对应的目录


remind： 基于bert的文本分类，先下载bert系列(bert、bertwwm、ernie、xlnet、roberta、albert)等， 放到指定的预训练目录(pretrain)

网络上，可以使用基本的bert+mlp方式，也可以在上层再套用CNN、RNN、RCNN、RNN_attention、HAN等网络，模型越复杂，对gpu的要求也越高

3、运行：
```
数据处理：
python run_bert.py --do_data # 数据预处理， 分割训练集和验证集，存储pickle格式

训练：
python run_bert.py --do_train # 注意修改超参数 默认model_name 是bert
python run_bert.py --do_train --use_cnn  1 # 使用bert +textcnn网络，下面的模型也可以在

python run_bert.py --do_train --model ernie
python run_bert.py --do_train --model ernie --use_cnn 1
python run_bert.py --do_train --model xlnet
python run_bert.py --do_train --model albert
python run_bert.py --do_train --model roberta
python run_bert.py --do_train --model bert_wwm

```

### 4 相关的git、预训练下载地址

#### bert：
- 论文：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- 论文中文版： https://github.com/yuanxiaosc/BERT_Paper_Chinese_Translation
- bert官方地址：[https://github.com/google-research/bert](https://github.com/google-research/bert)

#### bertwwm
- 论文：Pre-Training with Whole Word Masking for Chinese BERT
- 中文下载:[https://github.com/ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm) 

#### ernie(百度)
- 论文：ERNIE 1.0: Enhanced Representation through kNowledge IntEgration
- 论文：ERNIE 2.0: A Continual Pre-training Framework for Language Understanding
- 官网：https://github.com/PaddlePaddle/ERNIE/blob/develop/README.zh.md
- 提供中文下载：https://github.com/nghuyong/ERNIE-Pytorch

#### xlnet
- 论文： XLNet: Generalized Autoregressive Pretraining for Language Understanding
- 官网：[https://github.com/zihangdai/xlnet](https://github.com/zihangdai/xlnet)
- 中文下载 [https://github.com/ymcui/Chinese-PreTrained-XLNet](https://github.com/ymcui/Chinese-PreTrained-XLNet)
#### roberta
- 论文：RoBERTa: A Robustly Optimized BERT Pretraining Approach

- 中文下载 :[https://github.com/brightmart/roberta_zh](https://github.com/brightmart/roberta_zh)
-  https://github.com/guoday/CCF-BDCI-Sentiment-Analysis-Baseline/blob/master/README.md

#### albert
- 论文：[ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)
- 官方地址：https://github.com/google-research/ALBERT
- 提供中文下载 [https://github.com/brightmart/albert_zh](https://github.com/brightmart/albert_zh)
- 提供中文下载（pytorch版本） https://github.com/lonePatient/albert_pytorch/blob/master/README_zh.md

