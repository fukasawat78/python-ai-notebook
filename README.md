# Simple Data Science Example
================================

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-lightning)](https://pypi.org/project/pytorch-lightning/)
[![Conda](https://img.shields.io/conda/v/conda-forge/pytorch-lightning?label=conda&color=success)](https://anaconda.org/conda-forge/pytorch-lightning)

<!-- code_chunk_output -->

* [simple-data-science-examples]
	* [Requirements](#requirements)
	* [Usage](#how-to-run)
	* [Organization](#organization)
    * [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Dependencies
```
$ pip install -r requirements.txt
```

## Docker for Jupyter Lab
```
$ docker-compose up -d
```

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

## Acknowledgements
