# Common Voice Dataset

## Description

The Common Voice dataset consists of a unique MP3 and corresponding text file. Many of the 9,283 recorded hours in the dataset also include demographic metadata like age, sex, and accent that can help train the accuracy of speech recognition engines.

## Note

We provide the [data and client mapping and train/test splitting](https://fedscale.eecs.umich.edu/dataset/common-voice.tar.gz). The data folder structure is as follows
```
common_voice/
├── client_data_mapping
│   └── *.csv for client data mapping
```

# References
The original location of this dataset is at
[Common Voice Dataset](https://commonvoice.mozilla.org/en/datasets).

# Acknowledgement

```bibtex
@misc{common-voice,
  title = "{{Common Voice Data}}",
  howpublished="\url{https://commonvoice.mozilla.org/en/datasets}"
}
```