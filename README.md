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

### Acknowledgments
The code for constructing baseline rationales was adapted from [jifan-chen/QA-Verification-Via-NLI](https://github.com/jifan-chen/QA-Verification-Via-NLI/tree/master/seq2seq_converter)


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
