# GuzhengTech99_ICASSP2023

## Paper:
Dichucheng Li, Mingjin Che, Wenwu Meng, Yulun Wu, Yi Yu, Fan Xia and Wei Li ["Frame-Level Multi-Label Playing Technique Detection Using Multi-Scale Network and Self-Attention Mechanism"](https://arxiv.org/pdf/2303.13272.pdf), IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2023)

## File structure

- **./function/**: store all the python files related to training and testing
    - **fit.py**: training process
    - **lib.py**: loss function and metrics computing method
    - **model.py**: model structure
    - **norm_lib.py**: data normalization
    - **load_data.py**: load data
    - **config.py**: configuration option
- **run.py**: start the training process
- **test_frame.py**: start the evaluation process

## Training & Test Process
- Download Guzheng_Tech99 dataset (https://ccmusic-database.github.io/en/database/csmtd.html#Tech99)
- create a new folder "/data/model" to store Model's parameters during training process
- Train the model:
```
python run.py
```
- Test:
```
python test_frame.py
```

## Citation
```
@inproceedings{li2023frame,
  title={Frame-Level Multi-Label Playing Technique Detection Using Multi-Scale Network and Self-Attention Mechanism},
  author={Li, Dichucheng and Che, Mingjin and Meng, Wenwu and Wu, Yulun and Yu, Yi and Xia, Fan and Li, Wei},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP},
  year={2023},
  organization={IEEE}
}
```
