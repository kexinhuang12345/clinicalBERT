# ClinicalBERT

This repo hosts pretraining and finetuning weights and relevant scripts for ClinicalBERT, a contextual representation for clinical notes. 

### New: Clinical XLNet and Pretraining Script
1. clinical XLNet pretrained model is available at [here](https://github.com/kexinhuang12345/clinicalXLNet).

2. Detailed Step Instructions for pretraining ClinicalBERT and Clinical XLNet from scratch are available [here](https://github.com/kexinhuang12345/clinicalBERT/blob/master/notebook/pretrain.ipynb)

3. The predictive performance result is updated in this [version](https://www.kexinhuang.com/s/main.pdf) using the correct pretraining test splitting method described in pretraining script above. For more clinical outcomes performance comparison with more baselines using the correct split for ClinicalBERT/XLNet, please see the [Clinical XLNet](https://arxiv.org/abs/1912.11975) paper.

## Installation and Requirements

```
pip install pytorch-pretrained-bert
```

## Datasets

We use [MIMIC-III](https://mimic.physionet.org/about/mimic/). As MIMIC-III requires the CITI training program in order to use it, we refer users to the link. However, as clinical notes share commonality, users can test any clinical notes using the ClinicalBERT weight, although further fine-tuning from our checkpoint is recommended. 

File system expected:

```
-data
  -discharge
    -train.csv
    -val.csv
    -test.csv
  -3days
    -train.csv
    -val.csv
    -test.csv
  -2days
    -test.csv
```
Data file is expected to have column "TEXT", "ID" and "Label" (Note chunks, Admission ID, Label of readmission).


## ClinicalBERT Weights
Use [this google link](https://drive.google.com/file/d/1X3WrKLwwRAVOaAfKQ_tkTi46gsPfY5EB) to download pretrained ClinicalBERT along with the readmission task fine-tuned model weights.

The following scripts presume a model folder that has following structure:
```
-model
	-discharge_readmission
		-bert_config.json
		-pytorch_model.bin
	-early_readmission
		-bert_config.json
		-pytorch_model.bin
	-pretraining
		-bert_config.json
		-pytorch_model.bin
		-vocab.txt
```

## Hospital Readmission using ClinicalBERT

Below list the scripts for running prediction for 30 days hospital readmissions.

### Early Notes Prediction
```
python ./run_readmission.py \
  --task_name readmission \
  --readmission_mode early \
  --do_eval \
  --data_dir ./data/3days(2days)/ \
  --bert_model ./model/early_readmission \
  --max_seq_length 512 \
  --output_dir ./result_early
```
### Discharge Summary Prediction
```
python ./run_readmission.py \
  --task_name readmission \
  --readmission_mode discharge \
  --do_eval \
  --data_dir ./data/discharge/ \
  --bert_model ./model/discharge_readmission \
  --max_seq_length 512 \
  --output_dir ./result_discharge
```
### Training your own readmission prediction model from pretraining ClinicalBERT
```
python ./run_readmission.py \
  --task_name readmission \
  --do_train \
  --do_eval \
  --data_dir ./data/(DATA_FILE) \
  --bert_model ./model/pretraining \
  --max_seq_length 512 \
  --train_batch_size (BATCH_SIZE) \
  --learning_rate 2e-5 \
  --num_train_epochs (EPOCHs) \
  --output_dir ./result_new
```
It will use the train.csv from the (DATA_FILE) folder.

The results will be in the output_dir folder and it consists of 

1. 'logits_clinicalbert.csv': logits from ClinicalBERT to compare with other models
2. 'auprc_clinicalbert.png': Precision-Recall Curve 
3. 'auroc_clinicalbert.png': ROC Curve
4. 'eval_results.txt': RP80, accuracy, loss

## Preprocessing
We provide [script](./preprocess.py) for preprocessing clinical notes and merge notes with admission information on MIMIC-III. 

## Notebooks

1. [Attention](
        ./notebook/attention_visualization.ipynb
      ): this notebook is a tutorial to visualize self-attention.

## Gensim Word2Vec and FastText models

Please use [this link](https://drive.google.com/file/d/14EOqvvjJ8qUxihQ_SFnuRsjK9pOTrP-6/view?usp=sharing) to download Word2Vec and FastText models for Clinical Notes.

To use, simply 

```
import gensim
word2vec = gensim.models.KeyedVectors.load('word2vec.model')
weights = (m[m.wv.vocab])
```

## Contact
Please contact kh2383@nyu.edu for help or submit an issue. 

## Citation

Please cite [arxiv](https://arxiv.org/abs/1904.05342):
```
@article{clinicalbert,
author = {Kexin Huang and Jaan Altosaar and Rajesh Ranganath},
title = {ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission},
year = {2019},
journal = {arXiv:1904.05342},
}

```




