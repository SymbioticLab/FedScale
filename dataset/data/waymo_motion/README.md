# Waymo Motion

## Description

Waymo Motion Dataset is composed of 103,354 segments each containing 20
seconds of object tracks at 10Hz and map data for the area covered by the segment. These segments
are further broken into 9 second scenarios (8 seconds of future data and 1 second of history) with 5
second overlap, and we consider each scenario as a client.

## Note

We provide the [data and client mapping and train/test splitting](https://fedscale.eecs.umich.edu/dataset/waymo.tar.gz) s splited into training and testing set. The date folder structure is as follow
```
waymo/
├── client_data_mapping
│   └── *.csv for client data mapping and train/test splitting
```
# References
The original location of this dataset is at
[Waymo Motion](https://waymo.com/intl/en_us/dataset-motion/).

