# UDiffSE: Unsupervised Diffusion-based Speech Enhancement

This repository contains the PyTorch implementation for the following paper:

> B. Nortier, M. Sadeghi, and R. Serizel, [Unsupervised Speech Enhancement with Diffusion-based Generative Models](https://hal.science/hal-04210707), ICASSP 2024.

## Table of contents
- [Installation](#installation)
- [Training](#training)
- [Pretrained checkpoint](#pretrained-checkpoint)
- [Demo](#demo)
- [Audio samples](#audio-samples)
- [Supplementary material](#supplementary-material)
- [Bibtex](#bibtex)
- [References](#references)

## Installation

Create a virtual environment using `Python 3.8` and install the package dependencies via 
```
pip install -r requirements.txt
```

We find that the line `pypesq==1.2.4` may cause errors in which case we recommend using the [alternative](https://github.com/vBaiCai/python-pesq/blob/master/pypesq/__init__.py) suggestion to install pypesq with the command 
```
pip install https://github.com/vBaiCai/python-pesq/archive/master.zip
```

## Training

A diffusion-based clean speech generative model can be trained using `train.py`:
```
python train.py --transform_type exponent --format wsj0 --gpus 2 --batch_size 14  --resume_from_checkpoint file/to/last.ckpt
```

## Pretrained checkpoint

A pretrained checkpoint for a clean speech generative model trained on the WSJ0 dataset can be downloaded via this [Google drive link](https://drive.google.com/file/d/1sqP9ClhsJRP3Dy1tD7jHVc3wur8ho5Fn/view?usp=sharing).

## Demo

A demo of the UDiffSE framework is provided in `demo.ipynb`. This notebook presents a demonstration of sampling from clean speech prior learned via a diffusion-based generative model, followed by speech enhancement of a test noisy speech signal.

## Audio samples

A collection of audio samples that compare the speech enhancement performances of UDiffSE, RVAE [1] and SGMSE+ [2] over the WSJ0-QUT and TCD-TIMIT datasets may be found on [UDiffSE's webpage](https://team.inria.fr/multispeech/demos/udiffse).

## Supplementary material

Supplementary material, including additional details, discussions, and parameter studies that serve to expand our work is provided in the `docs` directory ([direct link](./docs/UDiffSE_Supplementary.pdf)).

## Bibtex

```bibtex
@inproceedings{nortier2023unsupervised,
  title={Unsupervised speech enhancement with diffusion-based generative models},
  author={Nortier, Bern{\'e} and Sadeghi, Mostafa and Serizel, Romain},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2024},
  organization={IEEE}
}
```

## References

[1] S. Leglaive, X. Alameda-Pineda, L. Girin, and R. Horaud, “A recurrent variational autoencoder for speech enhancement,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2020.

[2] J. Richter, S. Welker, J.-M. Lemercier, B. Lay, and T. Gerkmann, “Speech enhancement and dereverberation with diffusion-based generative models,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2023.
