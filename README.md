# Towards Generalist Models for Multimodal Clinical Diagnostics

This repository contains the official implementation of the Medical Imaging meets NeurIPs workshop paper "Towards Generalist Models for Multimodal Clinical Diagnostics".

## Requirements
The environment are as follows
- python 3.7
- pytorch 1.12.1
- transformer 4.27.1
- scikit-learn 1.0.2
- tqdm
- PIL

## Dataset

Please make sure you have credentialed access to our MMCaD dataset on [PhysioNet](https://physionet.org/). The dataset should be available for download soon. 

Note that we do not explicitly upload the chest X-ray images used in MMCaD. To download the images and you will need credentialed access to [MIMIC CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) and the run the `data/download_images.py` script using your PhysioNet account details:
```shell
python download_images.py
```

After downloading MMCaD, first prepare it for model input by running:
```shell
python prepare_data.py
```
We provide randomly split training, validation and test sets with a ratio of 7:1:2 in `data/train_idx.json`, `data/val_idx.json`, and `data/test_idx.json`, respectively.


## Training
Before training GeMini, you need to download the image and text feature extractors for encoding image and text modalities. For image feature extractor, we used [ViT patch 16](https://huggingface.co/google/vit-base-patch16-224) and [ViT patch 32](https://huggingface.co/google/vit-base-patch32-224-in21k). The text feature extractor is [PubMedBert](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext).

Please configure the data and feature extractor paths in the `run_train.sh` script according to your download directories.

Now everything is well set and you can start training by running the following script:
```shell
sh run_train.sh
```
Note that hyperparameters and training arguments are also specified in the `run_train.sh` script.
Our pre-trained checkpoint for GeMini can be downloaded from [link]().

## Evaluation
To evaluate GeMini on the test set, please update the data and model paths in `sh run_test.sh` accordingly and run:
```shell
sh run_test.sh
```

## Citation
If you use the MMCaD dataset in your work, please consider citing the following two papers:

```
@inproceedings{fu2023towards,
  title={Towards Generalist Models for Multimodal Clinical Diagnostics},
  author={Fu, Yunxiang and Zhou, Hong-Yu and Yu, Yizhou},
  booktitle={Medical Imaging Meets NeurIPS Workshop},
  year={2023}
}
```
```
@article{zhou2023irene,
  title={A transformer-based representation-learning model with unified processing of multimodal input for clinical diagnostics},
  author={Zhou, Hong-Yu and Yu, Yizhou and Wang, Chengdi and Zhang, Shu and Gao, Yuanxu and Pan, Jia and Shao, Jun and Lu, Guangming and Zhang, Kang and Li, Weimin},
  journal={Nature Biomedical Engineering},
  doi={10.1038/s41551-023-01045-x}
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```
