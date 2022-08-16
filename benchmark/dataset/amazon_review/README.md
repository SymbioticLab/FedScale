# Amazon Review Dataset

## Description

This dataset contains product reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 - July 2014.

This dataset includes reviews (ratings, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features), and links (also viewed/also bought graphs).

## Note

We provide the [data and client mapping and train/test splitting](https://fedscale.eecs.umich.edu/dataset/amazon_review.tar.gz). The data folder structure is as follows
```
amazon_review/
├── client_data_mapping
│   └── *.csv for client data mapping
```


# References
The original location of this dataset is at
[Amazon Review Dataset](https://jmcauley.ucsd.edu/data/amazon/).

# Acknowledgement

```bibtex
@inproceedings{amazon-review,
  title={Image-based Recommendations on Styles and Substitutes}, 
  author={Julian McAuley and Christopher Targett and Qinfeng Shi and Anton van den Hengel},
  year={2015},
  booktitle={SIGIR}
}
```