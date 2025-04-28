# ViTPose –– *FPN Architecture Implementation* 

> **This repository is a fork that builds on the original [ViTPose](https://github.com/ViTAE-Transformer/ViTPose).**  
> FPN modifications were implemented by: Adithya Raman, Jacob Klingler, Niva Ranavat and Sarah Jamil as part of ROB499/599 - [website](https://sarahtj.github.io/website/).

> This setup was tested on the GreatLakes computing cluster at the University of Michigan




## Installation

We use PyTorch 1.9.0 and mmcv 1.3.9, instructions for installing mmcv and other ViTPose dependancies are specifiec bellow:
```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.3.9
MMCV_WITH_OPS=1 pip install -e .
cd ..
git clone https://github.com/ViTAE-Transformer/ViTPose.git
cd ViTPose
pip install -v -e .
```

After install the two repos, install timm and einops, i.e.,
```bash
pip install timm==0.4.9 einops
```

## Dataset Preparation
Our experimentation was done using a subset the COCO 2017 dataset. To prepare this subset, the full COCO 2017 dataset and annotations will need to be downloade from [here](https://cocodataset.org/#download).

Once the dataset is downloaded, you can generate the subset using the tools in the utils folder, which includes generate_coco_split.py and coco_split.sh. In both of these files, you'll need to specify the filepath for your downloaded dataset and where the subset will be created. We recomend making the subset inside of a data directory within ViTPose.

## ViTPose-B Backbone Weight Extraction
Additionally, to run our FPN experiments, you will need the backbone weights from the original ViTPose. The ViTPose-B model weights can be downloaded from [here](https://onedrive.live.com/?id=E534267B85818129!163&resid=E534267B85818129!163&e=Q1uZKs&migratedtospo=true&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBaW1CZ1lWN0pqVGxnU01qcDFfTnJWM1ZSU21LP2U9UTF1Wktz&cid=e534267b85818129).

After downloading the weights, you'll need to extract the backbone weights which can be done through the following:
```bash
python utils/extract_backbone.py
```
Keep in mind you will need to edit two paths within the extract_backbone.py file based on where your vitpose-b.pth is located and where you want the destination to be.

## Training/Evaluation
To replicate the experiments from our paper, we have provided a set of configuration files to simplify usage. For each configuration file, you can use this process to train and evaluate:

Training a model:
```bash
python tools/train.py configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/<config file>
```

Evaluating a model:
```bash
python tools/test.py configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/<config file>
```

All configuration files will need a path specifying the location of the backbone weights from the ViTPose-B model to reproduce our results. In each configuration file, edit the top line to reflect:
```bash
load_from = '<path to backbone weights>'
```

Each configuration file has model parameters which can be edited to change stride size, transformer output selection, etc... A list of our configuration models and the respective experiment is specified here:
- simp_fpn_multiple_head.py - Final transformer feature maps generation with multiple feature maps passed into the decoder
- inter_config.py - Intermediate transformer output for feature map generation
- ViTPose_base_coco_256x192.py - ViTPose-B model
- ViTPose_base_coco_256x192_fpn.py - Fpn applied to final encoder layer


