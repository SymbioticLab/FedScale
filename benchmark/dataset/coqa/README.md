# CoQA Dataset

## Description

CoQA is a large-scale dataset for building Conversational Question Answering systems. The goal of the CoQA challenge is to measure the ability of machines to understand a text passage and answer a series of interconnected questions that appear in a conversation.

## Note

We provide the [data and client mapping and train/test splitting](https://fedscale.eecs.umich.edu/dataset/coqa.tar.gz). The data folder structure is as follows
```
coqa/
├── client_data_mapping
│   └── *.csv for client data mapping and train/val splitting
├── data
│   └── *.txt for client data
```
# References
The original location of this dataset is at
[CoQA Dataset](https://stanfordnlp.github.io/coqa/).

# Acknowledgement

```bibtex
@article{coqa,
  title={CoQA: A Conversational Question Answering Challenge}, 
  author={Siva Reddy and Danqi Chen and Christopher D. Manning},
  year={2019},
  journal={arXiv preprint arXiv:1808.07042},
}
```