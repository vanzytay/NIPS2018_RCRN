# NIPS 2018

We propose a new Recurrently controlled recurrent networks (RCRN) that shows some improvement over stacked BiLSTMs and BiLSTM across a number of NLP tasks.

This repository contains the Tensorflow model file for RCRN, according with the custom cuda optimized kernel. I will upload running scripts / example usage when I have time (already have tons of backlog ): ) .

# Dependencies

Python 2.7
Tensorflow 1.7

# Acknowledgements

Our CUDA op was adapted from: https://github.com/JonathanRaiman/tensorflow_qrnn

Cudnn RNN was adapted from:
https://github.com/HKUST-KnowComp/R-Net

# Reference

If you find our repository useful, please cite our paper!

```
@inproceedings{nips2018,
  author    = {Yi Tay and
               Luu Anh Tuan and
               Siu Cheung Hui},
  title     = {Recurrently Controlled Recurrent Networks},
  booktitle = {Proceedings of NIPS 2018},
  year      = {2018}
}
```
