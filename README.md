Code of GCGCN(Global Context-enhanced Graph Convolutional Networks for Document-level Relation Extraction)
===
Source code for Coling 2020 paper [Global Context-enhanced Graph Convolutional Networks for Document-level Relation Extraction](https://www.aclweb.org/anthology/2020.coling-main.461/)<br>
# Method<br>
![](https://github.com/Huiweizhou/GCGCN/blob/master/model_graph.png)
# Environments<br>
* Ubuntu-18.10.1( 4.18.0-25-generic)<br>
* Python(3.6.8)<br>
* Cuda(10.1.243)<br>
# Dependencies<br>
* matplotlib (3.3.2)<br>
* networkx (2.4)<br>
* nltk (3.4.5)<br>
* numpy (1.19.2)<br>
* torch (1.3.0)<br>
# Data<br>
First you should get pretrained Bert_base model from [huggingface](https://github.com/huggingface/transformers) and put it into `./bert/bert-base-uncased/`. <br>
Before running our code you need to obtain the DocRED dataset from the author of the dataset, [Here](https://github.com/thunlp/DocRED).<br>
After downing DocRED, you can use `gen_data_extend_graph.py` to preprocess data for GCGCN_Glove and use `gen_bert_data_extend_graph.py` to preprocess data for GCGCN_Bert. Finally, processed data will be saved into `./prepro_data` and `./prepro_data_bert` respectively.<br> 
# Run code<br>
`train.py` used to start training<br>
`test.py` used to evaluation model's performance on Dev or Test set.<br>
`Config.py` is for training Glove-based model And `Config_bert.py` is used for training Bert_based model
# Evaluation<br>
For Dev set, you can use `test.py` to evaluate you trained model.
For Test set, you should first use `test.py` to get test results which saved in `./result`, and submit it into [Condalab competition](https://competitions.codalab.org/competitions/20717).
# Contacts<br>
If you have any questions for code, please feel free to contact Yibin Xu(19xyb@mail.dlut.edu.cn), we will reply it as soon as possible.
