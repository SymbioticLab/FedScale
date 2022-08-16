# The Google Landmark Dataset


## Organization
We provide the [data and client mapping and train/val splitting](https://fedscale.eecs.umich.edu/dataset/landmark.tar.gz) for landmark dataset. Note that no details were kept of any of the participants age, gender, or location, and random ids were assigned to each individual. The data folder structure is as follows
```
data/
├── client_data_mapping
│   └── *.csv for client data mapping and train/test split
├── download-dataset.sh
│   └── script for raw data downloading
├── split_data.py
│   └── script for raw data splitting
```

The dataset is splited into training and testing set. Random ids were assigned to each client.

# Acknowledgement

```bibtex
@inproceedings{landmark,
  title     = {Google Landmarks Dataset v2 A Large-Scale Benchmark for Instance-Level Recognition and Retrieval},
  author    = {Tobias Weyand and Andre Araujo and Bingyi Cao and Jack Sim},
  booktitle = {arxiv.org/abs/2004.01804},
  year      = {2020},
}
```