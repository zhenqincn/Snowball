# Snowball

This repository contains the official implementation of our paper [Resisting Backdoor Attacks in Federated Learning via Bidirectional Elections and Individual Perspective](https://ojs.aaai.org/index.php/AAAI/article/view/29385) which has been accepted by **AAAI 2024**. 

> Existing approaches defend against backdoor attacks in federated learning (FL) mainly through a) mitigating the impact of infected models, or b) excluding infected models. The former negatively impacts model accuracy, while the latter usually relies on globally clear boundaries between benign and infected model updates. However, model updates are easy to be mixed and scattered throughout in reality due to the diverse distributions of local data. This work focuses on excluding infected models in FL. Unlike previous perspectives from a global view, we propose Snowball, a novel anti-backdoor FL framework through bidirectional elections from an individual perspective inspired by one principle deduced by us and two principles in FL and deep learning. It is characterized by a) bottom-up election, where each candidate model update votes to several peer ones such that a few model updates are elected as selectees for aggregation; and b) top-down election, where selectees progressively enlarge themselves through picking up from the candidates. We compare Snowball with state-of-the-art defenses to backdoor attacks in FL on five real-world datasets, demonstrating its superior resistance to backdoor attacks and slight impact on the accuracy of the global model.


## Project Structure
```Markdown
.
├── data
│   ├── processed                  // the processed FEMNIST and Sent140 should be saved in this directory.
│   ├── backdoor.py                // tools for building data samples with backdoor triggers
│   ├── data_loader.py             // tools for loading datasets
│   ├── partition.py               // tools for distributing subsets to clients
│   ├── preprocess_femnist.py      // tools for preprocessing data in FEMNIST
│   └── preprocess_sent140.ipynb   // tools for preprocessing data in Sent140
├── fl_core
│   ├── anti_poison                // implementation of baseline approaches and the proposed approaches
│   │   ├── baselines.py           // implementation of partial baselines.
│   │   ├── crfl.py                // implementation of `CRFL`
│   │   ├── flame.py               // implementation of `FLAME`
│   │   ├── rlr.py                 // implementation of `RLR`
│   │   ├── snowball_minus.py      // implementation of an ablation approach of `Snowball`, i.e., Snowball$\boxminus$
│   │   └── snowball.py            // implementation of the proposed approach, `Snowball`
│   ├── client.py                  // implementation of client-side functionalities
│   ├── server.py                  // implementation of server-side functionalities
│   └── trainer.py                 // tools for training and aggregation
├── models                         // definitions of model backbones
│   ├── ...   
│   └── ... 
├── tools                          // some tools
│   ├── ...   
│   └── ...  
└── main.py                        // entry for running experiments.
```


## Environment
Please refer to `requirements.txt` for the experimental environment of our work.


## Data Preparation
`FEMNIST` and `Sent140` are required to be preprocessed before running our experiments.

### FEMNIST
1. download the dataset at `/home/{username}/.dataset` 
2. enter `data` directory by conducting `cd data`
3. run `data/preprocess_femnist.py`

### Sent140
1. enter `data` directory by conducting `cd data`
2. download the `training.csv` and `test.csv` of Sent140 in the `data` directory
3. run `preprocess_sent140.ipynb` to preprocess the .csv files into .json files (Note that the preprocessing script handles the training and testing datasets separately. The current version of the code defaults to processing the testing dataset. When processing the training dataset, the filenames for reading and writing need to be modified.)

## Running Examples
Here we provide two examples for running Snowball.
1. on `MNIST` dataset
```shell
python main.py -a snowball --nodes 200 -k 50 --model_save none  --dataset "mnist" --malicious_ratio 0.2 --iid dir0.5 --model "mnistcnn" --rounds 100 --gpu 0 --vt 0.5
```

2. on `FEMNIST` dataset
```shell
python main.py -a myapp_ddif_ext --nodes 3597 -k 100 --model_save none  --dataset "femnist" --malicious_ratio 0.2 --seed 69 --model "mnistcnn" --rounds 200 --gpu 0 --backdoor_target 61 --vt 0.5 --vae_initial 360 --vae_tuning 40 --v_step 0.05
```

You can run experiments under different settings by adjusting the arguments. 
The meaning of each argument can be found in `main.py`.

## License
This project adopts the Apache-2.0 License. 
If the implementations and/or our paper were useful to you, please consider citing our [work](https://ojs.aaai.org/index.php/AAAI/article/view/29385):
```latex
@inproceedings{qin2024resisting,
  title={Resisting Backdoor Attacks in Federated Learning via Bidirectional Elections and Individual Perspective},
  author={Qin, Zhen and Chen, Feiyi and Zhi, Chen and Yan, Xueqiang and Deng, Shuiguang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={13},
  pages={14677--14685},
  year={2024}
}
```
