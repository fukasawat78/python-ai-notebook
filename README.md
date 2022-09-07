# simple-data-science-examples

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)

<!-- code_chunk_output -->

* [simple-data-science-examples]
	* [Requirements](#requirements)
	* [Usage](#how-to-run)
	* [Organization](#organization)
    * [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Requirements
```
$ pip install -r requirements.txt
```

## Usage
```
$ docker-compose up -d
```

## Kaggle
https://www.kaggle.com/iabhishekofficial/mobile-price-classification

## Organization

  ```
  simple-data-science-examples/
    │
    ├── .gitignore
    ├── .dvcignore
    ├── README.md           <- The top-level README for developers using this project.
    ├── config     
    ├── data   
    ├── model
    ├── logs
    ├── notebooks
    ├── src  
    ├── tests
    ├── docker-compose.yml
    ├── Dockerfile
    └── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
                              generated with `pip freeze > requirements.txt`
  ```

## DAGSHUB
```
fds add model data logs
dvc add model

dvc commit -f model.dvc
git add model.dvc logs/metrics.csv logs/params.yml
git commit -m ""
```  


## Acknowledgements
