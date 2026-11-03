## The GAN is dead; long live the GAN! A Modern Baseline GAN (R3GAN)<br><sub>Official PyTorch implementation of the NeurIPS 2024 paper</sub>

![Teaser image](./doc/teaser.png)

**The GAN is dead; long live the GAN! A Modern Baseline GAN**<br>
Nick Huang, [Aaron Gokaslan](https://skylion007.github.io/), [Volodymyr Kuleshov](https://www.cs.cornell.edu/~kuleshov/), [James Tompkin](https://www.jamestompkin.com)

OpenReview: https://openreview.net/forum?id=OrtN9hPP7V

arXiv: https://arxiv.org/abs/2501.05441

HuggingFace: https://huggingface.co/papers/2501.05441

<br>
Abstract: There is a widely-spread claim that GANs are difficult to train, and GAN architectures in the literature are littered with empirical tricks. We provide evidence against this claim and build a modern GAN baseline in a more principled manner. First, we derive a well-behaved regularized relativistic GAN loss that addresses issues of mode dropping and non-convergence that were previously tackled via a bag of ad-hoc tricks. We analyze our loss mathematically and prove that it admits local convergence guarantees, unlike most existing relativistic losses. Second, this loss allows us to discard all ad-hoc tricks and replace outdated backbones used in common GANs with modern architectures. Using StyleGAN2 as an example, we present a roadmap of simplification and modernization that results in a new minimalist baseline. Despite being simple, our approach surpasses StyleGAN2 on FFHQ, ImageNet, CIFAR, and Stacked MNIST datasets, and compares favorably against state-of-the-art GANs and diffusion models.

## Requirements

Our code requires the same packages as the official StyleGAN3 repo. However, we have updated the code so it is compatible with the latest version of the required packages (including PyTorch, etc).

## Getting started
To generate images using a given model, run:

```
# Generate 8 images using pre-trained FFHQ 256x256 model
gen_images.py --seeds=0-7 --outdir=out --network=ffhq-256x256.pkl

# Generate 64 airplane images using pre-trained CIFAR10 model
gen_images.py --seeds=0-63 --outdir=out --class=0 --network=cifar10.pkl
```

To reproduce the main results from our paper, run the following commands:

```
# CIFAR10
python train.py --outdir=./training-runs --data=./datasets/cifar10.zip --gpus=8 --batch=512 --mirror=1 --aug=1 --cond=1 --preset=CIFAR10 --tick=1 --snap=200

# FFHQ 64x64
python train.py --outdir=./training-runs --data=./datasets/ffhq-64x64.zip --gpus=8 --batch=256 --mirror=1 --aug=1 --preset=FFHQ-64 --tick=1 --snap=200

# FFHQ 256x256
python train.py --outdir=./training-runs --data=./datasets/ffhq-256x256.zip --gpus=8 --batch=256 --mirror=1 --aug=1 --preset=FFHQ-256 --tick=1 --snap=200

# ImageNet 32x32
python train.py --outdir=./training-runs --data=./datasets/imagenet-32x32.zip --gpus=32 --batch=4096 --mirror=1 --aug=1 --cond=1 --preset=ImageNet-32 --tick=1 --snap=200

# Imagenet 64x64
python train.py --outdir=./training-runs --data=./datasets/imagenet-64x64.zip --gpus=64 --batch=4096 --mirror=1 --aug=1 --cond=1 --preset=ImageNet-64 --tick=1 --snap=200
```

The easiest way to explore different training settings is to modify [`train.py`](./train.py) directly.

## Pre-trained models

We provide pre-trained models for our proposed training configuration (config E) on each dataset:

- [https://huggingface.co/brownvc/R3GAN-CIFAR10](https://huggingface.co/brownvc/R3GAN-CIFAR10)
- [https://huggingface.co/brownvc/R3GAN-FFHQ-64x64](https://huggingface.co/brownvc/R3GAN-FFHQ-64x64)
- [https://huggingface.co/brownvc/R3GAN-FFHQ-256x256](https://huggingface.co/brownvc/R3GAN-FFHQ-256x256)
- [https://huggingface.co/brownvc/R3GAN-ImgNet-32x32](https://huggingface.co/brownvc/R3GAN-ImgNet-32x32)
- [https://huggingface.co/brownvc/R3GAN-ImgNet-64x64](https://huggingface.co/brownvc/R3GAN-ImgNet-64x64)

## Preparing datasets
We use the same dataset format and dataset preprocessing tool as StyleGAN3 and EDM, refer to their repos for more details.

## Quality metrics
We support the following metrics:

* `fid50k_full`: Fr&eacute;chet inception distance against the full dataset.
* `kid50k_full`: Kernel inception distance against the full dataset.
* `pr50k3_full`: Precision and recall against the full dataset.
* `is50k`: Inception score for CIFAR-10.

Refer to the StyleGAN3 code base for more details.

## License

The code in [`R3GAN/`](./R3GAN) is released under the [MIT license](./R3GAN/LICENSE). The rest of the code derives from NVIDIA's [StyleGAN3](https://github.com/NVlabs/stylegan3) and is released under the [NVIDIA Source Code License](./LICENSE.txt).

The pre-trained models are released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (FFHQ) and [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (CIFAR-10, ImageNet).

## Citation

```
@inproceedings{
huang2024the,
title={The {GAN} is dead; long live the {GAN}! A Modern {GAN} Baseline},
author={Nick Huang and Aaron Gokaslan and Volodymyr Kuleshov and James Tompkin},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=OrtN9hPP7V}
}
```

## Acknowledgements

The authors thank Xinjie Jayden Yi for contributing to the proof and Yu Cheng for helpful discussion. For compute, the authors thank Databricks Mosaic Research. Nick Huang was supported by a Brown University Division of Research Seed Award, and James Tompkin was supported by NSF CAREER 2144956. Volodymyr Kuleshov was supported by NSF CAREER 2145577 and NIH MIRA 1R35GM15124301.
