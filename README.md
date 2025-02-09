[![DOI](https://zenodo.org/badge/892102804.svg)](https://doi.org/10.5281/zenodo.14840797)

# FNDCD
Fake News Detection via Causal Debiasing. As a plugging-in module on existing graph-based fake news detection, this model **FNDCD** adds a structure estimator and a posterior inference to debias the environment-biased samples in the training set. Experiments demonstrate that the FNDCD plugging on simple baselines achieves new state-of-the-art performance over a series of recent baselines on the unseen domain fake news detection (in an out-of-distribution scenario). 

# Model 
![](./figures/model.drawio.png)

# Train 

There are three selectable graph neural network backbones: BiGCN, GIN and GCNii; and two training data sources: Twitter and Weibo. 

To select the backbones and the training sources, follow the commands (e.g. training on Twitter dataset with BiGCN backbone) 

`
python main --gnn_model 'BiGCN' --data_source 'Twitter'
`
# Pre-trained Results
