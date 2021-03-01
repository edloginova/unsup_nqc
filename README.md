## Towards the application of calibrated Transformers to the unsupervised estimation of question difficulty from text

The code contained in this folder can be used to reproduce the results presented in our paper 
"Towards the application of calibrated Transformers to the unsupervised estimation of question 
difficulty from text".

From a high level perspective, these are the steps required to completely reproduce the results:
- Training of the QA models
- Inference on the test set
- Creation of the ensembles
- Evaluation of the predictions (and comparison with the baselines)

-----

### Data

All the data must be located in the `data/` directory.
- the RACE dataset in `data/race/`;
- all the other `.csv` files directly in `data/`.

Output files (containing the evaluation) are saved in one of the following folders:
- `output`;
- `output_figures`.

We provide here the output file of the crowd-sourcing task on MechanicalTurk,  from which we build the `PairwiseRACE_CS`
dataset. This output file can be found at `data/output_mturk.csv`.

We also provide, in the `data/` folder, the output of the QA models, which can be used for the task of pairwise 
difficulty prediction.
Due to the size of the training and eval parts of the dataset, we share only the scores of the test set.

-----

### Training of QA models

#### Output files

All output files have the filename in the following format:
`output_<model>_seed_<seed>_<split>.csv`,
where: 
- `<model>` is the architecture;
- `<seed>` is the random seed which generated those results;
- `<split>` is train, test, or eval.

They have the following columns:
- `idx`,
- `level`, 
- `document_id`,
- `label`,
- `prediction`,
- `score A`,
- `score B`,
- `score C`, 
- `score D`,
- `score variance`

These files have to be inserted in the `data` directory, as they are used by
the scripts that perform QDE.

-----

### Reliability diagram

- `script_image_reliability_diagrams.py`

-----

### Results presented in section PairwiseRACE_HM

The scripts to run are the following:
- `experiments_hm_ece.py`
- `experiments_hm_script_count_correct_answers.py`
- `experiments_hm_script_ensembles_QA_accuracy.py`
- `experiments_hm_test_ensembles.py`
- `experiments_hm_test_single_models.py`


-----

### Results presented in section PairwiseRACE_CS

The scripts to run are the following:
- `experiment_cs_1_script_data_prep.py`
- `experiment_cs_script_bert.py`
- `experiment_cs_script_distilbert.py`
- `experiment_cs_script_xlnet.py`
- `experiment_cs_script_ensembles.py`

-----