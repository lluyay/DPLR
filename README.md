# Dynamic Pseudo-Label Refinement

Code for the paper 'Semi-Supervised Medical Image Segmentation via Dynamic Pseudo-Label Refinement'.

## Environment Setup

##### Installation


1. Clone the repo

```sh
git clone https://github.com/lluyay/DPLR.git
cd DPLR
```

2. Install some important required packages

```
pip install -r requirements.txt
```

##### Data Preparation


Download the processed data and put the data in `../data/BraTS2019` or `../data/ACDC`, please read and follow the [README](https://github.com/Luoxd1996/SSL4MIS/tree/master/data/).

## Run

### Training on 2D/3D datasets:

```
sh train.sh
```

### Testing on 2D/3D datasets:

```
sh test.sh
```

## Acknowledgements

Our code is origin from SSL4MIS. We express our appreciation to these authors for their impactful contributions. We are hopeful that our novel method will further enrich the domain of Semi-supervised Medical Image Learning research.