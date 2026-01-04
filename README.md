# LiteVul
Code for the paper - LiteVul: A Lightweight Vulnerability Detection via Directed PPMI-weighted Token Graphs
## Introduction

In this work, we propose LiteVul, a lightweight framework that constructs compact directed graphs directly from raw source code without relying on external parsing tools, capturing multi-scale lexical co-occurrence relationships and fine-grained structural dependencies. LiteVul utilizes a multi-scale sliding window strategy to capture directed co-occurrence statistics and applies Directed Positive Pointwise Mutual Information (PPMI) to map these frequencies into a probabilistic space, effectively filtering noise and highlighting high-information density connections. Furthermore, we introduce SimpleEdge AwareUniNet, a graph neural network architecture that leverages TransformerConv with an edge-weight attention mechanism to integrate structural weights into the attention computation. By combining this with degree-aware attention pooling, the model adaptively focuses on critical code locations. The proposed method demonstrated superior performance compared to nine representative baseline methods on the Devign, Reveal, and DiverseVul datasets, while achieving substantially improved graph construction efficiency.

## Getting Started 

Create environment and install required packages for LiteVul

### Install packages

- [Python (3.9.2)](https://www.python.org/)
- [Pandas (2.2.0)](https://pandas.pydata.org/)
- [scikit-learn (1.6.1)](https://scikit-learn.org/stable/)
- [PyTorch (2.4.1)](https://pytorch.org/)
- [PyTorch Geometric (2.6.1)](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- [Gensim (4.3.3)](https://radimrehurek.com/gensim/)


All experiments were conducted on servers equipped with two NVIDIA Tesla V100 GPUs (each with 32GB memory). The system conffguration includes NVIDIA driver version 525.105.17 and CUDA version 12.0. The server is equipped with dual Intel Xeon Gold 5215 CPUs (2.50GHz) and 251GB memory. 

## Dataset

We evaluated the performance of our model using three publicly available datasets. The composition of the datasets is as follows, and you can click on the dataset names to download them.

| *Dataset*                                                    | *#Vulnerable* | *#Non-Vulnerable* | *Source*       |
| ------------------------------------------------------------ | ------------- | ----------------- | -------------- |
| [DiverseVul](https://drive.google.com/file/d/12IWKhmLhq7qn5B_iXgn5YerOQtkH-6RG/view?usp=sharing) | 18,945        | 330,492           | Snyk,Bugzilla  |
| [Devign](https://sites.google.com/view/devign)               | 11,888        | 14,149            | Github         |
| [ReVeal](https://github.com/VulDetProject/ReVeal)            | 1664          | 16,505            | Chrome, Debian |

## Usage

##### **Code normalization :**

```
python ./normalization.py -i /your/data/path
```

##### **Constructing Directed PPMI-weighted Token Graphs:**

```
python build_uni.py
```

##### **Split the dataset:**

```
python split.py 
```

##### **Training and Testing:**

```
python train.py 
```


