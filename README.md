# SplitSEE
SplitSEE: A Splittable Neural Framework for Single-Channel EEG Representation Learning

## Abstract
<p align="center">
<img src="misc/overview.png" width="600" class="center">
</p>

While end-to-end approach to multi-channel electroencephalogram (EEG) learning has shown significant promise, their applicability is often constrained in resource-limited clinical scenarios.
When provided with a single-channel EEG, how can we effectively capture representative features that are robust to multi-channels and scalable across varied clinical tasks, such as seizure prediction?

In this paper, we present SplitSEE, a structurally Splittable framework designed for effective Single-channel EEG representation learning.
The key concept bihind SplitSEE consists of 
- (1) three modules for temporal and frequency domain features and alignment,
- (2) a self-supervised learning framework to learn various feature representations without label supervision, and 
- (3) a pre-training-to-fine-tuning strategy complemented by a splittable architectural design.

SplitSEE has the following properties: 
- (a) Effectiveness: it learns informative features solely from single-channel EEG but has even outperformed multi-channel and multi-data source baselines.
- (b) Robustness: it shows the capacity to adapt across different channels with low performance variance. Superior performance is also achieved with our real clinical dataset.
- (c) Scalability: Our experiments show that with just one fine-tuning epoch, SplitSEE achieves high and stable performance using a few partial model layers.

## Setup

You can install the required dependencies using pip.

```bash
pip install -r requirements.txt
```

If you're using other than CUDA 11.7, you may need to install PyTorch for the proper version of CUDA. See [instructions](https://pytorch.org/get-started/locally/) for more details.


## Configurations
SplitSEE uses configuration files to manage training parameters for different datasets. 
These configuration files are in the ‘config_files’ folder and named after the corresponding dataset folders. 

For example, for the CHBMIT dataset, the configuration file is named ‘CHBMIT_Configs.py’. 

You can edit these configuration files to change the training parameters.
Changes can include adjustments to learning rates, batch sizes, number of epochs, and other model training parameters.

## Data
Put the data in a folder named 'data'. 
Inside this folder, in the subfolder for each dataset, place 'train.pt', 'val.pt', 'test.pt' files. 

The structure of the data files is a dictionary format like this: train.pt = {"samples": data, "labels": labels}, and similarly for val.pt, test.pt."

This repository includes a small CHBMIT dataset for testing.

## Run
You can select one of several modes:
 - Self-supervised training (self_supervised)
 - Fine-tuning the self-supervised model (fine_tune)
 - Testing the fine-tuned model (test)

You can run an experiment with the test dataset.

This repository includes the weights of the model trained with the CHBMIT dataset as the paper. 
It is not necessary to run self-supervised learning and fine-tuning.

```bash
python main.py --mode test --dataset CHBMIT --channel 0 --device cuda
```

## Result

The experiments are saved in the 'experiments_logs' directory by default (you can change that from args too).