
# (DOI: 10.11896/jsjkx.210500023)mini-AORCNN: A lightweight residual convolutional neuralnetwork with self-attention block for micro-expressionclassification

ResearchGate:
https://www.researchgate.net/publication/361723066_mini-AORCNNyizhongjiyuBottleneck_Transformerdeqingliangjiweibiaoqingshibiejiagou?_sg%5B0%5D=ge70tuXiu-L-3cfMTI5i7PBS_kueo_30AIKlo7x0WJhSWdPEENxf6BVXkk_G-hpWp4YnhU5dqgRF7PT1u-ric6h0eKccXnvLHbNhLfAs.b2DSmOvyth3I6LyMW-2KDObi9PVrlXuRnjZNa_nRUFtUDHbcqXqVpzoADOEBeDL3WCpxuljuVAnnmZDj3QV4pg

Cite:
ZHANG Jia-hao, LIU Feng, QI Jia-yin. Lightweight Micro-expression Recognition Architecture Based on Bottleneck Transformer[J]. Computer Science, 2022, 49(6A): 370-377. https://doi.org/10.11896/jsjkx.210500023

### How to Run the Code
1.Install packages mentioned in **requirements.txt** 

`pip install -r requirements.txt` 

2.Modify arguments in **train_arg.py**  

3.Get CASME,CASME2 and CASME-2 datasets from the link below,put the cropped pictures under the **dataset** directory. 

The name of the subfolders should be **casme1_cropped**,**casme2_cropped** and **casme^2_cropped** 

CASME - http://fu.psych.ac.cn/CASME/casme.php 

CASME2 - http://fu.psych.ac.cn/CASME/casme2.php 

CASME-2 - http://fu.psych.ac.cn/CASME/cas(me)2.php 

4.Run the code 

`python train.py` 

*The **mini-AORCNN** is named **ARFNet** (Attention-based Residual Flow Net) in source code.

The results can be seen in this chart below.

| Model | UAR | UF1 |Total Params | Total Flops | Total MemR+W |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Off-ApexNet | 0.5832 | 0.5650 | 2.66M | 3.87M | 10.35MB |
| STSTNet | 0.5584 | 0.5399 | 162,051 | 526.98K | 0.78MB |
| Dual-Inception | 0.6167 | 0.5814 | 6.45M | 12.64M | 26.27MB |
| MACNN | 0.6835 | 0.6660 | 70.57M | 793.67M | 297.86MB |
| Micro-Attention | 0.7086 | 0.7021 | 53.38M | 1.0G | 237.97MB |
| mini-AORCNN | 0.7309 | 0.7225 | 39,185 | 13.55M | 3.79MB |
