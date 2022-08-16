# Taxi Trajectory Prediction

## Description

This dataset describes a complete year (from 01/07/2013 to 30/06/2014) of the trajectories for all the 442 taxis (clients) running in the city of Porto, in Portugal. In this task, we may build a predictive framework that is able to infer the final destination of taxi rides based on their (initial) partial trajectories. Details of features are available [here](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/overview).

## Note

We provide the [data and client mapping and train/test splitting](https://fedscale.eecs.umich.edu/dataset/taxi_traj.tar.gz) for Taxi Trajectory Prediction. Note that no details were kept of any of the participants age, gender, or location, and random ids were assigned to each individual. The data folder structure is as follows
```
taxi_traj/
├── parser.py  # Script to generate the client-data mapping from the raw file
├──  train/test
│   └── *.csv  # Each csv file consists of the data of each client (i.e., taxi)
```
# References
The original location of this dataset is at
[ECML/PKDD 15: Taxi Trajectory Prediction](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/overview).

# Acknowledgement

```bibtex
@misc{taxi-trajs,
  title = "{{Taxi Trajectory Dataset}}",
  howpublished="\url{https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/overview}"
}
```