# FLAT

Code for the paper [Adversarial Training for Improving Model Robustness? Look at Both Prediction and Interpretation](https://www.aaai.org/AAAI22Papers/AAAI-2735.ChenH.pdf)

### Data
Download [datasets](https://drive.google.com/drive/folders/1707n-X__GXdHNGFv1I2j2GLugIEsLj6J?usp=sharing) and put them in the folder `textattack/my_datasets`.

### Preparation
- Install the packages and toolkits in `requirements.txt`
- `cd` into `CNN_LSTM` and `BERT_DeBERTa` for running experiments for CNN/LSTM and BERT/DeBERTa respectively

### Training Base Models

**Training CNN/LSTM base models**

For IMDB, set `--max_seq_length 250`. Fine-tune hyperparameters (e.g. learning rate, the number of hidden units) on each dataset.
```
python train.py train --gpu_id 2 --model cnn/lstm --dataset sst2/imdb/ag/trec --task base --batch-size 64 --epochs 10 --learning-rate 0.01 --max_seq_length 50
```

**Training BERT/DeBERTa base models**

For IMDB, set `--max_seq_length 250`. Fine-tune hyperparameters (e.g. learning rate, weight decay) on each dataset.
```
python train.py train --gpu_id 2 --model bert/deberta --dataset sst2/imdb/ag/trec --task base --epochs 10 --learning-rate 1e-5 --max_seq_length 50
```

### Adversarial Training

**Adversarial training for CNN/LSTM**

For IMDB, set `--max_seq_length 250`. Fine-tune hyperparameters (e.g. learning rate, the number of hidden units) on each dataset.
```
python train.py train --attack textfooler/pwws --gpu_id 2 --model cnn/lstm --dataset sst2/imdb/ag/trec --task adv --batch-size 64 --epochs 30 --learning-rate 0.01 --max_seq_length 50 --num-clean-epochs 10
```

**Adversarial training for BERT/DeBERTa**

For IMDB, set `--max_seq_length 250`. Fine-tune hyperparameters (e.g. learning rate, weight decay) on each dataset.
```
python train.py train --attack textfooler/pwws --gpu_id 2 --model bert/deberta --dataset sst2/imdb/ag/trec --task adv --epochs 30 --learning-rate 1e-5 --max_seq_length 50 --num-clean-epochs 10
```

### FLAT

Search $\beta, \gamma$ in (0.0001, 0.001, 0.01,...,1000). The optimal hyperparameters vary across different models and datasets.

**FALT for CNN/LSTM**

For IMDB, set `--max_seq_length 250`. Fine-tune hyperparameters (e.g. $\beta, \gamma$, learning rate, the number of hidden units) on each dataset.
```
python train.py train --attack textfooler/pwws --gpu_id 2 --model cnn_mask/lstm_mask --dataset sst2/imdb/ag/trec --task adv_reg --batch-size 64 --epochs 30 --learning-rate 0.005 --max_seq_length 50 --num-clean-epochs 10 --beta 0.1 --gamma 0.001
```

**FLAT for BERT/DeBERTa**

For IMDB, set `--max_seq_length 250`. Fine-tune hyperparameters (e.g. $\beta, \gamma$, learning rate, weight decay) on each dataset.
```
python train.py train --attack textfooler/pwws --gpu_id 2 --model bert_mask/deberta_mask --dataset sst2/imdb/ag/trec --task adv_reg --epochs 30 --learning-rate 1e-5 --max_seq_length 50 --num-clean-epochs 10 --beta 0.1 --gamma 0.001
```

### Adversarial Attack

**Attack CNN/LSTM**

```
python attack.py attack --recipe textfooler/pwws --model path_to_model_checkpoint --dataset sst2/imdb/ag/trec --task base/adv/adv_reg (corresponding to the training strategy of target model) --num-examples 10000 --save_file save_file_name.txt --gpu_id 2
```

**Attack BERT/DeBERTa**

```
python attack.py attack --recipe textfooler/pwws --model path_to_model_checkpoint --dataset sst2/imdb/ag/trec --task base/adv/adv_reg (corresponding to the training strategy of target model) --num-examples 10000 --save_file save_file_name.txt --gpu_id 2
```


### Acknowledgments
The code was built on top of [TextAttack](https://github.com/QData/TextAttack) and [Hugging Face/Transformers](https://github.com/huggingface/transformers)

### Reference:
If you find this repository helpful, please cite our paper:
```bibtex
@inproceedings{chen2022adversarial,
    title={Adversarial Training for Improving Model Robustness? Look at Both Prediction and Interpretation},
    author={Chen, Hanjie and Ji, Yangfeng},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    year={2022}
}
```
