# A tiny and concise PyTorch template for deep learning projects

> modified from https://github.com/chosj95/MIMO-UNet


## Feature

- single file for each module, clear and concise
- no complex functions and dependencies, fast validation of ideas
- DDP support
- argparse for easy parameter setting

## Dependencies

- Python
- Pytorch
- scikit-image
- opencv-python
- Tensorboard

## Directory Structure

- `data/`: dataset and data processing
- `dataset/`: dataset for training and testing
- `docs`: documents
- `models/`: network architecture
- `toolbox/`: some useful tools
- `utils/`: some useful functions used for tasks
- `main.py`: main function for training and testing
- `run.sh`: shell script for running the code
- `train.py`: training function
- `test.py`: test/evaluation function
- `valid.py`: validation function
- `utils.py`: some useful functions used for the template
- `exp/`: exp results and outputs

## Usage

`python main.py -c configs/config.yaml`