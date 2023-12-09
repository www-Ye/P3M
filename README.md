# P3M
Code for AAAI 2024 Conference paper [A Positive-Unlabeled Metric Learning Framework for Document-Level Relation Extraction with Incomplete Labeling].

## Requirements
* Python (tested on 3.6.7)
* CUDA (tested on 11.0)
* [PyTorch](http://pytorch.org/) (tested on 1.7.1)
* [Transformers](https://github.com/huggingface/transformers) (tested on 4.18.0)
* numpy (tested on 1.19.5)
* [apex](https://github.com/NVIDIA/apex) (tested on 0.1)
* [opt-einsum](https://github.com/dgasmith/opt_einsum) (tested on 3.3.0)
* ujson
* tqdm

## Dataset
The [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset can be downloaded following the instructions at [link](https://github.com/thunlp/DocRED/tree/master/data).

The [Re-DocRED](https://arxiv.org/abs/2205.12696) dataset can be downloaded following the instructions at [link](https://github.com/tonytan48/Re-DocRED).

The [ChemDisGene](https://arxiv.org/abs/2204.06584) dataset can be downloaded following the instructions at [link](https://github.com/chanzuckerberg/ChemDisGene).
```
P3M
 |-- dataset
 |    |-- docred
 |    |    |-- train_annotated.json
 |    |    |-- train_distant.json
 |    |    |-- train_ext.json
 |    |    |-- train_revised.json
 |    |    |-- dev.json
 |    |    |-- dev_ext.json
 |    |    |-- dev_revised.json
 |    |    |-- test_revised.json
 |    |-- chemdisgene
 |    |    |-- train.json
 |    |    |-- valid.json
 |    |    |-- test.anno_all.json
 |-- meta
 |    |-- rel2id.json
 |    |-- relation_map.json
```

## Training and Evaluation
### DocRED
Train DocRED model with the following command:

```bash
>> sh scripts/run_bert_p3m.sh  # P3M BERT
>> sh scripts/run_roberta_p3m.sh  # P3M RoBERTa
>> sh scripts/run_bert_p3m_ext.sh  # P3M BERT Extremely unlabeled
>> sh scripts/run_roberta_p3m_ext.sh  # P3M RoBERTa Extremely unlabeled
>> sh scripts/run_bert_p3m_full.sh  # P3M BERT Fully supervised
>> sh scripts/run_roberta_p3m_full.sh  # P3M RoBERTa Fully supervised
```

### ChemDisGene
Train ChemDisGene model with the following command:
```bash
>> sh scripts/run_bio_p3m.sh  # P3M PubmedBERT
```
