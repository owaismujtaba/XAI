# Model Training & Evaluation (Phoneme Decoding)

This directory contains code for training and evaluating the brain-to-phoneme RNN model. The model is based on the architecture described in Card et al. (2024), focusing on translating neural activity directly into phoneme sequences.

## Setup
1. Install the `b2txt25` conda environment as described in the root `README.md`.
2. Download the [Dryad Dataset](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85) and place it in the `data` directory.

## Training
To train the baseline RNN model, run the `train.py` script:
```bash
python src/train.py
```
The model predicts phonemes from neural data using CTC loss and the AdamW optimizer. Hyperparameters are specified in `src/rnn_args.yaml`.

## Evaluation
To evaluate the model's phoneme decoding performance:
```bash
python src/evaluate.py --model_path path/to/checkpoint.pt --data_dir path/to/hdf5_data --eval_type val
```
This script will:
1. Load the trained RNN model.
2. Perform greedy decoding on the validation or test set.
3. Calculate the **Phoneme Error Rate (PER)** for the validation set.
4. Save the predicted phoneme sequences to a CSV file.

## Project Structure
- `src/core/`: Core logic including `model.py`, `dataset.py`, `trainer.py`, and `evaluator.py`.
- `src/utils/`: Helper utilities for augmentations and general processing.
- `src/train.py`: Unified training entry point.
- `src/evaluate.py`: Unified evaluation entry point.
