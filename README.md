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

The codes of our backbone are adapted from [AGW](https://github.com/mangye16/Cross-Modal-Re-ID-baseline) [3] and [CAJ](https://github.com/mangye16/Cross-Modal-Re-ID-baseline) [4].

###  5. References.

[1] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.

[2] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.

[3] Ye M, Shen J, Lin G, et al. Deep learning for person re-identification: A survey and outlook[J]. IEEE transactions on pattern analysis and machine intelligence, 2021, 44(6): 2872-2893.

[4] Ye M, Ruan W, Du B, et al. Channel augmented joint learning for visible-infrared recognition[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 13567-13576.