"""
Training logic for brain-to-text models.
"""

import os
import time
import logging
import math
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
import torchaudio.functional as AF
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import DictConfig, OmegaConf

from src.core.dataset import BrainToTextDataset, train_test_split_indices
from src.core.model import GRUDecoder
from src.utils.augmentations import gauss_smooth


class BrainToTextTrainer:
    """
    Handles training and validation of a brain-to-text phoneme decoder.
    """

    def __init__(self, args: DictConfig):
        """
        Args:
            args (DictConfig): Reorganized configuration dictionary.
        """
        self.args = args
        self.device = self._setup_device()
        self.logger = self._setup_logging()

        # Metrics tracking
        self.best_val_per = float("inf")
        self.best_val_loss = float("inf")

        # Initialize components
        self.model = self._init_model()
        self.train_loader, self.val_loader = self._init_dataloaders()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.criterion = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=False)

        # Checkpoint restoration
        if self.args.training.checkpoint.init_from:
            self.load_checkpoint(self.args.training.checkpoint.path)

        # Freeze layers if specified
        self._freeze_layers()

        self.model.to(self.device)

    def _setup_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s: %(message)s")

        if logger.hasHandlers():
            logger.handlers.clear()

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        if self.args.experiment.mode == "train":
            os.makedirs(self.args.paths.output_dir, exist_ok=True)
            log_path = os.path.join(self.args.paths.output_dir, "training_log")
            fh = logging.FileHandler(log_path)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        return logger

    def _init_model(self) -> torch.nn.Module:
        model = GRUDecoder(
            neural_dim=self.args.model.n_input_features,
            n_units=self.args.model.n_units,
            n_days=len(self.args.dataset.sessions),
            n_classes=self.args.dataset.n_classes,
            rnn_dropout=self.args.model.rnn_dropout,
            input_dropout=self.args.model.input_network.input_layer_dropout,
            n_layers=self.args.model.n_layers,
            patch_size=self.args.model.patch_size,
            patch_stride=self.args.model.patch_stride,
        )
        return torch.compile(model)

    def _init_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        ds_args = self.args.dataset
        base_dir = self.args.paths.dataset_dir
        
        train_paths = [os.path.join(base_dir, s, "data_train.hdf5") for s in ds_args.sessions]
        val_paths = [os.path.join(base_dir, s, "data_val.hdf5") for s in ds_args.sessions]

        train_idxs, _ = train_test_split_indices(
            train_paths, test_percentage=0, seed=ds_args.seed
        )
        _, val_idxs = train_test_split_indices(
            val_paths, test_percentage=1, seed=ds_args.seed
        )

        train_ds = BrainToTextDataset(
            trial_indices=train_idxs,
            split="train",
            n_batches=self.args.training.num_batches,
            batch_size=ds_args.batch_size,
            days_per_batch=ds_args.days_per_batch,
            random_seed=ds_args.seed,
            feature_subset=ds_args.get("feature_subset"),
        )
        val_ds = BrainToTextDataset(
            trial_indices=val_idxs,
            split="test",
            n_batches=0,
            batch_size=ds_args.batch_size,
            random_seed=ds_args.seed,
            feature_subset=ds_args.get("feature_subset"),
        )

        train_loader = DataLoader(
            train_ds, batch_size=None, shuffle=ds_args.shuffle, 
            num_workers=ds_args.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=None, shuffle=False, num_workers=0, pin_memory=True
        )

        return train_loader, val_loader

    def _init_optimizer(self) -> torch.optim.Optimizer:
        opt_args = self.args.training.optimizer
        params = self.model.named_parameters()
        bias_params = [p for n, p in params if "bias" in n]
        day_params = [p for n, p in params if "day_" in n]
        other_params = [p for n, p in params if "day_" not in n and "bias" not in n]

        param_groups = [
            {"params": bias_params, "weight_decay": 0},
            {
                "params": day_params,
                "lr": opt_args.lr_max_day,
                "weight_decay": opt_args.weight_decay_day,
            },
            {"params": other_params},
        ]

        return torch.optim.AdamW(
            param_groups,
            lr=opt_args.lr_max,
            betas=(opt_args.beta0, opt_args.beta1),
            eps=opt_args.epsilon,
            weight_decay=opt_args.weight_decay,
            fused=True if "cuda" in str(self.device) else False,
        )

    def _init_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        sch_args = self.args.training.scheduler
        opt_args = self.args.training.optimizer

        def lr_lambda(step, min_ratio, decay, warmup):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            if step < decay:
                progress = float(step - warmup) / float(max(1, decay - warmup))
                cos = 0.5 * (1 + math.cos(math.pi * progress))
                return max(min_ratio, min_ratio + (1 - min_ratio) * cos)
            return min_ratio

        lambdas = [
            lambda s: lr_lambda(s, opt_args.lr_min / opt_args.lr_max, sch_args.total_steps, sch_args.warmup_steps),
            lambda s: lr_lambda(s, opt_args.lr_min_day / opt_args.lr_max_day, sch_args.total_steps_day, sch_args.warmup_steps_day),
            lambda s: lr_lambda(s, opt_args.lr_min / opt_args.lr_max, sch_args.total_steps, sch_args.warmup_steps),
        ]
        return LambdaLR(self.optimizer, lambdas)

    def _freeze_layers(self):
        m_args = self.args.model
        for name, param in self.model.named_parameters():
            if not m_args.rnn_trainable and "gru" in name:
                param.requires_grad = False
            if not m_args.input_network.input_trainable and "day" in name:
                param.requires_grad = False

    def transform_data(self, x, n_steps, mode="train"):
        t_args = self.args.dataset.transforms
        batch_size, time_steps, channels = x.shape

        if mode == "train":
            if t_args.get("static_gain_std", 0) > 0:
                warp = torch.tile(torch.eye(channels, device=self.device).unsqueeze(0), (batch_size, 1, 1))
                warp += torch.randn_like(warp) * t_args.static_gain_std
                x = torch.matmul(x, warp)
            if t_args.get("white_noise_std", 0) > 0:
                x += torch.randn_like(x) * t_args.white_noise_std
            if t_args.get("constant_offset_std", 0) > 0:
                x += torch.randn((batch_size, 1, channels), device=self.device) * t_args.constant_offset_std
            if t_args.get("random_walk_std", 0) > 0:
                x += torch.cumsum(torch.randn_like(x) * t_args.random_walk_std, dim=1)
            if t_args.get("random_cut", 0) > 0:
                cut = np.random.randint(0, t_args.random_cut)
                x = x[:, cut:, :]
                n_steps = n_steps - cut

        if t_args.smooth_data:
            x = gauss_smooth(
                x, self.device, t_args.smooth_kernel_std, t_args.smooth_kernel_size
            )
        return x, n_steps

    def train(self) -> Dict[str, List[float]]:
        self.model.train()
        stats = {"train_loss": [], "val_loss": [], "val_per": []}
        val_steps_no_improve = 0
        
        early_stop_args = self.args.training.early_stopping
        log_args = self.args.logging if hasattr(self.args, 'logging') else self.args.get('logging', {})

        for i, batch in enumerate(self.train_loader):
            self.model.train()
            self.optimizer.zero_grad()

            features = batch["input_features"].to(self.device)
            labels = batch["seq_class_ids"].to(self.device)
            n_steps = batch["n_time_steps"].to(self.device)
            p_lens = batch["phone_seq_lens"].to(self.device)
            d_idxs = batch["day_indices"].to(self.device)

            with torch.autocast(device_type="cuda", enabled=self.args.experiment.use_amp, dtype=torch.bfloat16):
                features, n_steps = self.transform_data(features, n_steps, "train")
                adj_lens = ((n_steps - self.args.model.patch_size) / self.args.model.patch_stride + 1).to(torch.int32)
                
                logits = self.model(features, d_idxs)
                loss = self.criterion(
                    logits.log_softmax(2).permute(1, 0, 2), labels, adj_lens, p_lens
                ).mean()

            loss.backward()
            if self.args.training.grad_norm_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.training.grad_norm_clip)
            
            self.optimizer.step()
            self.scheduler.step()

            stats["train_loss"].append(loss.item())

            if i % self.args.logging.train_log_freq == 0:
                self.logger.info(f"Batch {i}: loss {loss.item():.4f}")

            if i % self.args.logging.val_log_freq == 0 or i == (self.args.training.num_batches - 1):
                res = self.validate()
                stats["val_loss"].append(res["avg_loss"])
                stats["val_per"].append(res["avg_per"])
                
                self.logger.info(f"Val {i}: PER {res['avg_per']:.4f}, loss {res['avg_loss']:.4f}")

                if res["avg_per"] < self.best_val_per:
                    self.best_val_per = res["avg_per"]
                    self.save_checkpoint("best_checkpoint")
                    val_steps_no_improve = 0
                else:
                    val_steps_no_improve += 1

                if early_stop_args.enabled and val_steps_no_improve >= early_stop_args.patience:
                    self.logger.info("Early stopping triggered")
                    break

        return stats

    def validate(self) -> Dict[str, Any]:
        self.model.eval()
        losses = []
        total_ed, total_len = 0, 0

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch["input_features"].to(self.device)
                labels = batch["seq_class_ids"].to(self.device)
                n_steps = batch["n_time_steps"].to(self.device)
                p_lens = batch["phone_seq_lens"].to(self.device)
                d_idxs = batch["day_indices"].to(self.device)

                features, n_steps = self.transform_data(features, n_steps, "val")
                adj_lens = ((n_steps - self.args.model.patch_size) / self.args.model.patch_stride + 1).to(torch.int32)
                
                logits = self.model(features, d_idxs)
                loss = self.criterion(
                    logits.log_softmax(2).permute(1, 0, 2), labels, adj_lens, p_lens
                ).mean()
                losses.append(loss.item())

                for b in range(logits.shape[0]):
                    pred = torch.argmax(logits[b, :adj_lens[b]], dim=-1)
                    pred = torch.unique_consecutive(pred)
                    pred = pred[pred != 0].cpu().numpy()
                    true = labels[b, :p_lens[b]].cpu().numpy()
                    
                    total_ed += AF.edit_distance(pred, true)
                    total_len += len(true)

        return {"avg_loss": np.mean(losses), "avg_per": total_ed / total_len if total_len > 0 else 0}

    def _init_model(self) -> torch.nn.Module:
        model = GRUDecoder(
            neural_dim=self.args.model.n_input_features,
            n_units=self.args.model.n_units,
            n_days=len(self.args.dataset.sessions),
            n_classes=self.args.dataset.n_classes,
            rnn_dropout=self.args.model.rnn_dropout,
            input_dropout=self.args.model.input_network.input_layer_dropout,
            n_layers=self.args.model.n_layers,
            patch_size=self.args.model.patch_size,
            patch_stride=self.args.model.patch_stride,
        )
        model = torch.compile(model)
        
        if self.args.experiment.get("use_multi_gpu", False):
            n_gpus = torch.cuda.device_count()
            self.logger.info(f"Using Multi-GPU: {n_gpus} devices detected")
            model = torch.nn.DataParallel(model)
            
        return model

    def save_checkpoint(self, name: str):
        path = os.path.join(self.args.paths.checkpoint_dir, f"{name}.pt")
        os.makedirs(self.args.paths.checkpoint_dir, exist_ok=True)
        
        # Strip 'module.' prefix if using DataParallel
        state_dict = self.model.state_dict()
        if self.args.experiment.get("use_multi_gpu", False):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
        torch.save({
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_per": self.best_val_per,
        }, path)
        
        with open(os.path.join(self.args.paths.checkpoint_dir, "config.yaml"), "w") as f:
            OmegaConf.save(self.args, f)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        state_dict = ckpt["model_state_dict"]
        
        # If model is DataParallel but checkpoint is single-GPU (or vice versa), fix keys
        if self.args.experiment.get("use_multi_gpu", False):
            new_state_dict = {}
            for k, v in state_dict.items():
                name = f"module.{k}" if not k.startswith("module.") else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_val_per = ckpt["val_per"]
        self.logger.info(f"Loaded checkpoint from {path}")
