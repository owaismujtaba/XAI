"""
Inference and evaluation logic for brain-to-text models.
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Tuple
from omegaconf import DictConfig

from src.core.model import GRUDecoder
from src.utils.augmentations import gauss_smooth
from src.utils.helpers import LOGIT_TO_PHONEME


class BrainToTextEvaluator:
    """
    Handles model inference for neural-to-phoneme sequence translation.
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        model_args: DictConfig,
    ):
        """
        Args:
            model_path (str): Path to the trained model checkpoint.
            device (torch.device): Device for inference.
            model_args (DictConfig): Reorganized configuration used during training.
        """
        self.device = device
        self.args = model_args
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """
        Loads the GRUDecoder model from a checkpoint.
        """
        model = GRUDecoder(
            neural_dim=self.args.model.n_input_features,
            n_units=self.args.model.n_units,
            n_days=len(self.args.dataset.sessions),
            n_classes=self.args.dataset.n_classes,
            n_layers=self.args.model.n_layers,
            patch_size=self.args.model.patch_size,
            patch_stride=self.args.model.patch_stride,
        )
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"]
        
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "").replace("_orig_mod.", "")
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        model.eval()
        return model

    def predict_phonemes(self, neural_data: np.ndarray, day_idx: int) -> List[str]:
        """
        Runs neural data through the RNN and decodes to a phoneme sequence.
        """
        x = torch.from_numpy(neural_data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=self.args.experiment.use_amp, dtype=torch.bfloat16):
                x = gauss_smooth(
                    x, self.device, 
                    self.args.dataset.transforms.smooth_kernel_std,
                    self.args.dataset.transforms.smooth_kernel_size,
                    padding="valid"
                )
                logits = self.model(x, torch.tensor([day_idx], device=self.device))
        
        logits = logits.float().cpu().numpy()[0]
        pred_ids = np.argmax(logits, axis=-1)
        
        decoded_ids = []
        for i, val in enumerate(pred_ids):
            if val != 0 and (i == 0 or val != pred_ids[i-1]):
                decoded_ids.append(val)
                
        return [LOGIT_TO_PHONEME[idx] for idx in decoded_ids]

    def evaluate_trials(self, trials: List[Dict[str, Any]], day_idx: int) -> List[Dict[str, Any]]:
        """
        Runs inference on a list of trials.
        """
        results = []
        for trial in tqdm(trials, desc="Decoding phonemes"):
            pred_phonemes = self.predict_phonemes(trial["neural_features"], day_idx)
            
            res = {
                "block": trial["block_num"],
                "trial": trial["trial_num"],
                "pred_phonemes": " ".join(pred_phonemes),
            }
            if "true_phonemes" in trial:
                res["true_phonemes"] = trial["true_phonemes"]
            results.append(res)
            
        return results
