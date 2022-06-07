# European Parliament Proceedings Parallel Corpus (Preprocessed)

## Description

The Europarl parallel corpus is extracted from the proceedings of the European Parliament. It includes versions in 21 European languages: Romanic (French, Italian, Spanish, Portuguese, Romanian), Germanic (English, Dutch, German, Danish, Swedish), Slavik (Bulgarian, Czech, Polish, Slovak, Slovene), Finni-Ugric (Finnish, Hungarian, Estonian), Baltic (Latvian, Lithuanian), and Greek. We use French and English here.

## Note

The [dataset](https://fedscale.eecs.umich.edu/dataset/europarl.tar.gz) is splited into training and testing set. Note that no details were kept of any of the participants age, gender, or location, and random ids were assigned to each individual. The date folder structure is as follow
```
data/
├── client_data_mapping
│   └── *.csv for client data mapping and train/test split
├── en
│   └── en.csv English data
├── fr
│   └── fr.csv French data
```

# References
The original location of this dataset is at
[https://www.statmt.org/europarl/](https://www.statmt.org/europarl/).