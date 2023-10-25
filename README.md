# SplitSEE
SplitSEE: A Splittable Neural Framework for Single-Channel EEG Representation Learning

## Setup

You can install the required dependencies using pip.

```bash
pip install -r requirements.txt
```

If you're using other than CUDA 11.7, you may need to install PyTorch for the proper version of CUDA. See [instructions](https://pytorch.org/get-started/locally/) for more details.

## Run

You can run an experiment with a small test dataset.

```bash
python main.py --mode test --dataset CHBMIT --channel 0 --device cuda
```

## Result

The experiments are saved in "experiments_logs" directory by default (you can change that from args too).