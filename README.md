## Insightful analysis of historical sources at scales beyond human capabilities using unsupervised Machine Learning and Explainable AI
## Code and Data Repository


##  Get started

### 1. Virtual Environment

We used `python 3.10` with packages specified in `requirements.txt`.

### 2. Data

The full Sacrobosco Tables Dataset used in our paper can be accessed via [https://zenodo.org/record/5767440](https://zenodo.org/record/5767440) .
All  *.jpg files are expected to be stored in a directory `RAW_DATA_DIR={local_corpus_dir}/raw`. Also update the location in `configs.py` (`SACROBOSCO_DATA_ROOT={local_corpus_dir}`).

To preprocess the full corpus, execute `run_prepare_data.py` and make sure to update `RAW_DATA_DIR` to the correct location. The processed corpus is also provided via DOI `10.5281/zenodo.10933232` (`sacrobosco_tables.zip`).

A demo repository is provided in `data/example_corpus`.

Processed data files used for training are contained in `page_data.zip`. Unzip these files into `data/page_data`. The resulting folders, 
`data/page_data/contrast_patches` and `data/page_data/patches_annotated`,  contain the files loaded in `run_train_table.py`. The folder `data/page_data/eval_annotated` contains the pages for which full digit annotations were collected.


## Description of main scripts

### Data
`run_prepare_data.py`: Preprocessing, i.e. binarization of pages, to prepare the full corpus. Expects the Tables Dataset to be in `RAW_DATA_DIR`.

## Model
`run_train_table.py`: Script used to train our digit detection model. Our trained model weights and jobdict file are given in `data/trained_model` for usage without re-training.

### Inference
`run_table_inference.py`: Script to compute the table page representations, an example to process our corpus is given in `paper_scripts.sh`. 

### Insights
`run_geographical_analysis.py`: Script to reproduce the geographical entropy analysis. Outputs are saved in `results/geographical`.

`run_temporal_analysis.py`: Script to reproduce the temporal entropy analysis. Outputs are saved in `results/temporal`.

`run_evaluation.py`: Script for the evaluation with respect to ground truth digit and cluster annotations.  Outputs are saved in `results/task_evaluation`.
