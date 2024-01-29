# D3AD: Dynamic Denoising Diffusion Probabilistic Model for Anomaly Detection
![D3AD](imgs/D3AD.png)
This repo contains the official implementation of [D3AD](https://arxiv.org/abs/2401.04463)
## Setup
**Install dependencies:**
```
pip install -r requirements.txt
```
For [consistency decoder](https://github.com/openai/consistencydecoder):
```
$ pip install git+https://github.com/openai/consistencydecoder.git
```
**Create folder structure** for model checkpoints and results like:
```
checkpoints
|--- VisA
|--- BTAD
|--- MVTec
results
|--- category_name
```
**Data:** <br> 
[MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads) can be used as is. <br>
For [VisA](https://github.com/amazon-science/spot-diff) please structure like:
```
VisA_pytorch
|--- candle
|-----|----- ground_truth
|-----|----- test
|-----|-------|------- good 
|-----|-------|------- bad 
|-----|----- train
|-----|-------|------- good
|--- capsules
|--- ...
```
## Run the model
For **training** of the model run:
```
python main.py
```
For **evaluating** model performance run:
```
python main.py --eval True
```

## Citation
```
@article{tebbe2024d3ad,
  title={D3AD: Dynamic Denoising Diffusion Probabilistic Model for Anomaly Detection},
  author={Tebbe, Justin and Tayyub, Jawad},
  journal={arXiv preprint arXiv:2401.04463},
  year={2024}
}
```