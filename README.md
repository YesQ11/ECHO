# Echo in Noise: Exploring Noise-Augmented Conditional Diffusion for Speech Enhancement
The official implementation of Echo in Noise: Exploring Noise-Augmented Conditional Diffusion for Speech Enhancement

## Abstract
Diffusion-based speech enhancement (SE) methods have achieved remarkable success using noise interpolation, yet their underlying mechanism remains theoretically underexplored. In this work, we propose ECHO (Exploring Noise-Augmented Conditional Diffusion for Speech Enhancement), a novel framework that reinterprets noise interpolation as beneficial noise augmentation, unifying it within the broader context of distributional augmentation. This perspective reveals how implicitly introduced noise promotes model generalization. Building on this insight, ECHO introduces a noise re-mixup mechanism that decouples and remixes signals for broader distribution coverage. Additionally, we enhance the network architecture with deformable convolutions to better capture dynamic noise patterns. Experimental results demonstrate ECHO consistently outperforms state-of-the-art methods across matched, cross-dataset, and low-resource scenarios, validating the effectiveness of strategically integrated noise.

## Environment Requirements
```
# create virtual environment
conda create --name echo python=3.9.0

# activate environment
conda activate echo

# install required packages
pip install -r requirements_py39.txt
```
## How to train
python train.py --log_dir <path_to_model> --base_dir <path_to_dataset>
## How to evaluate
python enhancement.py --test_dir <path_to_noisy> --enhanced_dir <path_to_enhanced> --ckpt <path_to_model_checkpoint>

python calc_metrics.py --clean_dir <path_to_clean> --noisy_dir <path_to_noisy> --enhanced_dir <path_to_enhanced>
## Folder Structure
```
.
├── calc_metrics.py
├── enhancement.py
├── README.md
├── requirements_py39.txt
├── echo
│   ├── backbones
│   │   ├── __init__.py
│   │   ├── ncsnpp_deform.py
│   │   ├── ncsnpp_utils
│   │   │   ├── layerspp.py
│   │   │   ├── layers.py
│   │   │   ├── normalization.py
│   │   │   ├── op
│   │   │   │   ├── fused_act.py
│   │   │   │   ├── fused_bias_act.cpp
│   │   │   │   ├── fused_bias_act_kernel.cu
│   │   │   │   ├── __init__.py
│   │   │   │   ├── upfirdn2d.cpp
│   │   │   │   ├── upfirdn2d_kernel.cu
│   │   │   │   └── upfirdn2d.py
│   │   │   ├── up_or_down_sampling.py
│   │   │   └── utils.py
│   │   └── shared.py
│   ├── data_module.py
│   ├── model.py
│   ├── sampling
│   │   ├── correctors.py
│   │   ├── __init__.py
│   │   └── predictors.py
│   ├── sdes.py
│   └── util
│       ├── inference.py
│       ├── other.py
│       ├── registry.py
│       ├── semp.py
│       └── tensors.py
└── train.py
```

