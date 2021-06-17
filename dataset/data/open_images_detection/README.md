# Open Images Dataset V5 (For Detection)

## Description

Open Images is a dataset of ~9M images annotated with image-level labels, object bounding boxes, object segmentation masks, visual relationships, and localized narratives:

## Note

The [dataset](https://fedscale.eecs.umich.edu/dataset/openimage_detection.tar.gz) is splited into training and testing set. Note that no details were kept of any of the participants age, gender, or location, and random ids were assigned to each individual. The date folder structure is as follow
```
data/
├── *.txt meta files
├── client_data_mapping
│   └── *.csv for client data mapping and train/test splitting
├── VOCdevkit2007
│   └── VOC2007
│       ├── Annotations
│       │   └── *.xml
│       ├── ImageSets
│       │   └── Main
│       │       └── *.txt
│       └── JPEGImages
│           └── *.jpg Download all OID images to here
```

# References
The original location of this dataset is at
[https://storage.googleapis.com/openimages/web/download_v5.html](https://storage.googleapis.com/openimages/web/download_v5.html).

