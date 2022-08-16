# Waymo Motion

## Description

Waymo Motion Dataset is composed of 103,354 segments each containing 20
seconds of object tracks at 10Hz and map data for the area covered by the segment. These segments
are further broken into 9 second scenarios (8 seconds of future data and 1 second of history) with 5
second overlap, and we consider each scenario as a client.

## Note

We provide the [data and client mapping and train/test splitting](https://fedscale.eecs.umich.edu/dataset/waymo.tar.gz) s splited into training and testing set. The data folder structure is as follows
```
waymo/
├── client_data_mapping
│   └── *.csv for client data mapping and train/test splitting
```
# References
The original location of this dataset is at
[Waymo Motion](https://waymo.com/intl/en_us/dataset-motion/).

# Acknowledgement

```bibtex
@article{waymo-motion,
  author    = {Scott Ettinger and
               Shuyang Cheng and
               Benjamin Caine and
               Chenxi Liu and
               Hang Zhao and
               Sabeek Pradhan and
               Yuning Chai and
               Benjamin Sapp and
               Charles Qi and
               Yin Zhou and
               Zoey Yang and
               Aurelien Chouard and
               Pei Sun and
               Jiquan Ngiam and
               Vijay Vasudevan and
               Alexander McCauley and
               Jonathon Shlens and
               Dragomir Anguelov},
  title     = {Large Scale Interactive Motion Forecasting for Autonomous Driving
               : The Waymo Open Motion Dataset},
  journal   = {CoRR},
  volume    = {abs/2104.10133},
  year      = {2021}
}
```