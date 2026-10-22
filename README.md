# Spatial-Group Token Attention and Adaptive Expert Fusion for Dynamic Facial Expression Recognition

Repository accompanying the paper:

> **Spatial-Group Token Attention and Adaptive Expert Fusion for Dynamic Facial Expression Recognition**
> Jiajun Jin, Geng Li, Yanliang Zhang
> School of Physics and Electronic Information, Henan Polytechnic University, Jiaozuo, Henan, China
> 

## Repository Contents

```
DFER-SGTA-MoAE/
├── README.md
├── requirements.txt
├── .gitignore
└── libs/
    ├── __init__.py
    ├── SGTA.py             
    ├── MoAE.py             
    └── model_metrics.py    
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision, numpy, scikit-learn, matplotlib, seaborn

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

The DFEW dataset was obtained from its original provider under the provider's
access conditions and is not redistributed in this repository. See the DFEW
dataset website and the original paper by Jiang et al. (ACM MM 2020). The
AFEW 7.0 dataset, used for cross-database transfer evaluation, is likewise
subject to the original provider's terms.

## Code Availability

The implementation code for SGTA and MoAE and the trained model weights are
currently maintained in a private repository. For peer review, the code and
trained models can be provided to reviewers upon reasonable request. After
publication, the authors will consider releasing the code subject to
institutional approval.

Requests for access should be sent to the first author: Jiajun Jin,
212512020033@home.hpu.edu.cn. The corresponding author is Yanliang Zhang,
ylzhang@hpu.edu.cn.

## Citation

```bibtex
@misc{jin2026sgta_moae,
  title  = {Spatial-Group Token Attention and Adaptive Expert Fusion for Dynamic Facial Expression Recognition},
  author = {Jin, Jiajun and Li, Geng and Zhang, Yanliang},
  note   = {Submitted to the Journal of Electronic Imaging, SPIE, under review},
  year   = {2026}
}
```
