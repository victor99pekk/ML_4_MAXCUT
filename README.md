# `Max-Cut`: Learning based approaches

This repository contains the code for my study `maximum cut: learning based approaches` read the paper at this link: [https://victor99pekk.github.io/ML_4_MAXCUT/](https://victor99pekk.github.io/ML_4_MAXCUT/) 

In this study I __(i)__ explored ways of generating training with planted solutions for the (SL) of the `NP-hard` problem maxcut, __(ii)__ trained `neural models` to solve maxcut, __(iii)__ studied the properties of the graphs we can generate with planted solutions, as well as studied the `Geonman & Williamson` lower bound to derive `FS-graphs` which are harder for heuristic algorithms to solve and where learning based approaches might have a benefit.

`run_on_gpu.ipynb` provides an easy tutorial how you can generate data and train the models used in this project. This can be done on either GPUs (free via opening this the notebook in google colba) with cuda, or on a cpu.\
__link__: [run_on_gpu.ipynb](run_on_gpu.ipynb)

### Other links in repo
`(i)`     [data generation](data/gen_maxcut_data.py)\
`(ii)`    [Neural models](neural_network/models/)\
`(iii)`   [training of networks](neural_network/train_network.py)\
`(iv)`   [sources](pdfs/)