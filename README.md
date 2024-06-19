# Dynamic Addition of Noise in a Diffusion Model for Anomaly Detection
![D3AD](imgs/D3AD.png)
This repo contains the official implementation of [Dynamic Addition of Noise in a Diffusion Model for Anomaly Detection](https://openaccess.thecvf.com/content/CVPR2024W/VAND/papers/Tebbe_Dynamic_Addition_of_Noise_in_a_Diffusion_Model_for_Anomaly_CVPRW_2024_paper.pdf)
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
@InProceedings{Tebbe_2024_CVPR,
    author    = {Tebbe, Justin and Tayyub, Jawad},
    title     = {Dynamic Addition of Noise in a Diffusion Model for Anomaly Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {3940-3949}
}
```
