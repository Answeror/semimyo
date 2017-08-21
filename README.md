# Semi-Supervised Learning for Surface EMG-based Gesture Recognition

## Prerequisite

* A CUDA compatible GPU
* Ubuntu 14.04 or any other Linux/Unix that can run Docker
* [Docker](http://docker.io/)
* [Nvidia Docker](https://github.com/NVIDIA/nvidia-docker)
* Download the docker image:
  ```
  docker pull answeror/sigr:semi
  ```
  or build it by yourself:
  ```
  docker build -t answeror/sigr:semi -f docker/Dockerfile .
  ```

## Steps to generate table 4

```
# Prepare data
mkdir -p .cache
# Download https://www.idiap.ch/project/ninapro
# Put NinaPro DB1 in .cache/ninapro-db1-raw
# Download http://zju-capg.org/myo/data
# Put CapgMyo DB-a in .cache/dba
# Put CapgMyo DB-b in .cache/dbb
# Put CapgMyo DB-c in .cache/dbc
# Download http://www.csl.uni-bremen.de/cms/forschung/bewegungserkennung
# Put csl-hdemg in .cache/csl

scripts/train_table_4.sh
scripts/app test_semimyo --cmd table_4
```

Training on NinaPro and CapgMyo will take 1 to 2 hours depending on your GPU.
Training on csl-hdemg will take several days.
You can accelerate traning and testing by distribute different folds on different GPUs with the `gpu` parameter.

You can also do train and test for each dataset on different machines or GPUs:

```
scripts/train_db1.sh
scripts/train_dba.sh
scripts/train_dbb.sh
scripts/train_dbc.sh
scripts/train_csl.sh
scripts/app test_semimyo --cmd table_4_db1
scripts/app test_semimyo --cmd table_4_dba
scripts/app test_semimyo --cmd table_4_dbb
scripts/app test_semimyo --cmd table_4_dbc
scripts/app test_semimyo --cmd table_4_csl
```

## Steps to generate table 1

```
# Prepare data
mkdir -p .cache
# Download https://www.idiap.ch/project/ninapro
# Put NinaPro DB1 in .cache/ninapro-db1-raw
# Extract pre-calculated cluster (512 clusters) labels to data folder
unzip data/pose-512 -d .cache/ninapro-db1-raw

scripts/train_table_1.sh
scripts/app test_semimyo --cmd table_1
```

## License

Licensed under an GPL v3.0 license.

## Bibtex

```
@inproceedings{Du_IJCAI_2017,
  author    = {Yu Du, Yongkang Wong, Wenguang Jin, Wentao Wei, Yu Hu, Mohan Kankanhalli, Weidong Geng},
  title     = {Semi-Supervised Learning for Surface EMG-based Gesture Recognition},
  booktitle = {Proceedings of the Twenty-Sixth International Joint Conference on
               Artificial Intelligence, {IJCAI-17}},
  pages     = {1624--1630},
  year      = {2017},
  doi       = {10.24963/ijcai.2017/225},
  url       = {https://doi.org/10.24963/ijcai.2017/225},
}
```

## Misc

Thanks DMLC team for their great [MxNet](https://github.com/dmlc/mxnet)!
