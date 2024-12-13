# Hypergraph-Driven Soft Semantics Flexible Learning Network
Pytorch code for paper "**Hypergraph-Driven Soft Semantics Flexible Learning for Visible-Infrared Person Re-identification**".

### 1. Datasets

- RegDB [1]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html).

- SYSU-MM01 [2]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

  - run:

  - unzip RegDB.zip

  - unzip SYSU-MM01.zip

  - run `python pre_process_sysu.py` to pepare the dataset, the training data will be stored in ".npy" format.

    ```
    python pre_process_sysu.py
    ```

### 2. Training


**Train HSFLNet by**

```
python train.py --dataset 'sysu' --gpu 0
```
- `--dataset`: which dataset "sysu" or "regdb".

- `--gpu`: which gpu to run.

*You may need manually define the data path first.*



### 3. Testing

**Test a model on SYSU-MM01 dataset by**

```
python test.py --dataset 'sysu' --mode 'all' --resume 'model_path'  --gpu 0
```
  - `--dataset`: which dataset "sysu" or "regdb".
  - `--mode`: "all" or "indoor"  (only for sysu dataset).
  - `--resume`: the saved model path.
  - `--gpu`: which gpu to use.



**Test a model on RegDB dataset by**

```
python test.py --dataset 'regdb' --resume 'model_path'  --tvsearch True --gpu 0
```
  - `--tvsearch`:  whether thermal to visible search  True or False (only for regdb dataset).

### 4. Citation

The codes of our backbone are adapted  from [AGW] and [CAJ](https://github.com/mangye16/Cross-Modal-Re-ID-baseline)
