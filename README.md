# TabCSDI: Diffusion models for missing value imputation in tabular data

This is the repo for the workshop paper: [Diffusion models for missing value imputation in tabular data | OpenReview](https://openreview.net/forum?id=4q9kFrXC2Ae).

## Setup

```
pip install -r requirements.txt
```

## Running experiments

We provide 3 datasets, including Breast (original), Breast (diagnostic), and Census datasets. For census datasets, three categorical variable handling methods are provided.

Run pure numerical datasets experiments:
- Breast (original) dataset
```
python exe_breast.py
```
- Breast (diagnostic) dataset
```
python exe_breastD.py
```

Run mixed datatypes experiments with census dataset:
- Using feature tokenization for categorical variables
```
python exe_census_ft.py
```
- Using analog bits encoding for categorical variables
```
python exe_census_analog.py
```
- Using one-hot encoding for categorical variables
```
python exe_census_onehot.py
```
## Acknowledgements

The code repo is built upon the [CSDI repo](https://github.com/ermongroup/CSDI).

## Reference
If you find our code useful or use it in your work, please cite the following paper:

```
@inproceedings{tashiro2021csdi,
  title={Diffusion models for missing value imputation in tabular data},
  author={Zheng, Shuhan and Charoenphakdee, Nontawat},
  booktitle={NeurIPS Table Representation Learning (TRL) Workshop},
  year={2022}
}
```

