# MiranDa

![cover-large](./fig/Picture1.jpg)
[MiranDa: Mimicking the learning process of human doctors to achieve causal inference for medication recommendation](https://arxiv.org/submit/5536286/view)  
I've been a huge fan of the Biohazard games since I was 12 years old, so I named this model after one of the characters from RE8.

### Visulize medication combinations in Hyperbolic space
![Picture3](https://github.com/azusakou/MiranDa/blob/6deaf738c42d7626af2d60aea3b9012483179ba8/fig/Picture3.jpg)


## Datasets

All input data have been accessed through credential verification from public sources, including [MIMIC III database](https://physionet.org/content/mimiciii/1.4/) and [MIMIC IV database](https://physionet.org/content/mimiciv/0.4/).
### preprocessing
Quite complex. Upon acceptance of the paper, I will provide this code. But I have already provided enough details about the preprocessing in the paper, so you can do it by yourself as well.

## Train/inference

we utilized a range of random seed values from 0 to 29, facilitating 30 distinct dataset splits into training, validation, and test sets for MIMIC III database, and 0 for MIMIC IV database.
The models underwent a training period of 50 epochs, with an early stopping mechanism initiated from the fifth epoch, set to withstand a tolerance of three epochs. Decay factor λ, threshold δ,
Confidence γ, Blend factor ϵ was 0.9, 0.2, 0.5, 0.2. The learning rate is 1 × 10−3. A batch size of 512 was uniformly adhered to across the experiment

```bash
python MiranDa.py
```
## Hypermeters

you can use different parameters in cfg.py

| Hypermeters              | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| `dataset_list`           | Train models for different datasets |
| `train_all_models`       | Train different agents |
| `save_memory`            | Delete useless features  |
| `read_data`              | Skip the preprocessing |
| `train_the_last`         | I know it is important, but I really forget, em, if anyone is interested about it, please let me know.|

If you have any questions, please let me know. E-mail: wang@med.tohoku.ac.jp
