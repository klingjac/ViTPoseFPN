# ViTPose –– *FPN Architecture Implementation* 

> **This repository is a fork that builds on the original [ViTPose](https://github.com/ViTAE-Transformer/ViTPose).**  
> FPN modifications were implemented by: Adithya Raman, Jacob Klingler, Niva Ranavat and Sarah Jamil as part of ROB499/599.

> This setup was tested on the GreatLakes computing cluster at the University of Michigan




## Usage/Installation

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



