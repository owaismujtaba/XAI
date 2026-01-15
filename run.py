"""
Unified entry point for the XAI Brain-to-Text project.
Uses config.yaml to determine whether to run training or evaluation.
"""

import os
import sys
import time
import pandas as pd
import torch
import torchaudio.functional as AF
from omegaconf import OmegaConf

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.trainer import BrainToTextTrainer
from src.core.evaluator import BrainToTextEvaluator
from src.utils.helpers import LOGIT_TO_PHONEME


def load_trials(data_dir, sessions, eval_type):
    """
    Loads trial data for evaluation.
    """
    import h5py
    all_trials = []
    for session in sessions:
        file_path = os.path.join(data_dir, session, f"data_{eval_type}.hdf5")
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found.")
            continue
            
        with h5py.File(file_path, "r") as f:
            for key in f.keys():
                trial = f[key]
                t_info = {
                    "session": session,
                    "block_num": trial.attrs["block_num"],
                    "trial_num": trial.attrs["trial_num"],
                    "neural_features": trial["input_features"][:],
                }
                if "seq_class_ids" in trial:
                    true_ids = trial["seq_class_ids"][:].tolist()
                    t_info["true_phonemes"] = " ".join([LOGIT_TO_PHONEME[i] for i in true_ids if i != 0])
                
                all_trials.append(t_info)
    return all_trials


def run_training(config):
    """
    Starts the training pipeline.
    """
    print("--- Starting Training ---")
    os.makedirs(config.paths.output_dir, exist_ok=True)
    os.makedirs(config.paths.checkpoint_dir, exist_ok=True)
    
    trainer = BrainToTextTrainer(config)
    trainer.train()


def run_evaluation(config):
    """
    Starts the evaluation pipeline.
    """
    print("--- Starting Evaluation ---")
    eval_config = config.evaluation
    
    # Load original training configuration if available to ensure architecture matches
    model_dir = os.path.dirname(os.path.dirname(eval_config.model_path))
    model_args_path = os.path.join(model_dir, "checkpoint", "config.yaml")
    
    if os.path.exists(model_args_path):
        print(f"Loading architecture config from {model_args_path}")
        model_args = OmegaConf.load(model_args_path)
    else:
        print("Using current config.yaml for architecture settings")
        model_args = config
        
    device = torch.device(f"cuda:{eval_config.gpu_number}" 
                         if torch.cuda.is_available() and eval_config.gpu_number >= 0 
                         else "cpu")
    
    evaluator = BrainToTextEvaluator(
        model_path=eval_config.model_path,
        device=device,
        model_args=model_args
    )
    
    # Use dataset_dir from paths
    trials = load_trials(config.paths.dataset_dir, model_args.dataset.sessions, eval_config.eval_type)
    print(f"Loaded {len(trials)} trials for evaluation.")
    
    results = evaluator.evaluate_trials(trials, day_idx=0)
    
    if eval_config.eval_type == "val":
        total_ed, total_len = 0, 0
        for res in results:
            if "true_phonemes" in res:
                pred = res["pred_phonemes"].split()
                true = res["true_phonemes"].split()
                ed = AF.edit_distance(pred, true)
                total_ed += ed
                total_len += len(true)
        
        if total_len > 0:
            print(f"\nAggregate Phoneme Error Rate (PER): {100 * total_ed / total_len:.2f}%")

    if eval_config.get("save_csv", True):
        output_path = f"phoneme_results_{eval_config.eval_type}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Phoneme decoding results saved to {output_path}")


def main():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        sys.exit(1)
        
    config = OmegaConf.load(config_path)
    mode = config.experiment.get("mode", "train")
    
    if mode == "train":
        run_training(config)
    elif mode in ["inference", "evaluate", "val", "test"]:
        run_evaluation(config)
    else:
        print(f"Error: Unknown mode '{mode}'. Choose 'train' or 'inference'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
