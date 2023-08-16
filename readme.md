# ADQE:Obtain better deep learning models by evaluating the augmented data quality using information entropy

We built a mathematical tool potentially for choosing the best data augmentation strategies. 

## dataset

Only Image classification data set. Put samples of the same class in one folder, and put all class folders in one directory. Replace the directory under `run.py`. Modify the augmentation strategy by editing `all_strategy` and `cnum`.

```
python data_augment.py
```

## run

Replace the directory under `run.py`

```
python run.py
```

The results can be visualized by `anstotxt.py`