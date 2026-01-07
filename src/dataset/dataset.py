
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os

class BrainDataset(Dataset):
    def __init__(self, data_dir, dataset_type='train'):
        """
        Args:
            data_dir (str): Path to the directory containing date folders (e.g. data/raw/hdf5_data_final)
            dataset_type (str): 'train', 'test', or 'val'
        """
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.filepaths = self._get_filepaths()
        self.transform_data = [] # List of tuples (filepath, trial_key)
        self._index_data()

    def _get_filepaths(self):
        # Search recursively for .hdf5 files matching the dataset_type
        search_pattern = os.path.join(self.data_dir, '**', f'data_{self.dataset_type}.hdf5')
        return glob.glob(search_pattern, recursive=True)

    def _index_data(self):
        print(f"Indexing {self.dataset_type} data from {len(self.filepaths)} files...")
        for filepath in self.filepaths:
            try:
                with h5py.File(filepath, 'r') as f:
                    keys = list(f.keys())
                    for key in keys:
                        # Ensure it's a trial group and has required datasets
                        if key.startswith('trial_') and 'input_features' in f[key] and 'seq_class_ids' in f[key]:
                            self.transform_data.append((filepath, key))
            except Exception as e:
                print(f"Error indexing {filepath}: {e}")
        print(f"Found {len(self.transform_data)} trials.")

    def __len__(self):
        return len(self.transform_data)

    def __getitem__(self, idx):
        filepath, trial_key = self.transform_data[idx]
        
        with h5py.File(filepath, 'r') as f:
            group = f[trial_key]
            input_features = group['input_features'][:]
            
            # Use attributes if available, else use shape
            input_length = group.attrs.get('n_time_steps', input_features.shape[0])
            
            if 'seq_class_ids' in group:
                seq_class_ids = group['seq_class_ids'][:]
                target_length = group.attrs.get('seq_len', 0)
                if target_length == 0:
                    non_zero_indices = np.nonzero(seq_class_ids)[0]
                    if len(non_zero_indices) > 0:
                        target_length = non_zero_indices[-1] + 1
            else:
                seq_class_ids = np.array([], dtype=np.int64)
                target_length = 0
            
            # Additional metadata from official logic
            metadata = {
                'session': group.attrs.get('session'),
                'block_num': group.attrs.get('block_num'),
                'trial_num': group.attrs.get('trial_num'),
                'sentence_label': group.attrs.get('sentence_label'),
                'filepath': filepath,
                'trial_key': trial_key
            }
            
        # Convert to torch tensors
        input_features = torch.from_numpy(input_features).float()
        seq_class_ids = torch.from_numpy(seq_class_ids).long()
        
        return input_features, seq_class_ids, input_length, target_length, metadata

def collate_fn(batch):
    # Pad inputs and targets to max length in batch
    input_features, seq_class_ids, input_lengths, target_lengths, metadatas = zip(*batch)
    
    # Pad inputs (T x D) -> (B x T_max x D)
    input_features_padded = torch.nn.utils.rnn.pad_sequence(input_features, batch_first=True, padding_value=0)
    
    # Process targets: Add SOS (41) and EOS (42), remove trailing zeros first
    SOS_TOKEN = 41
    EOS_TOKEN = 42
    
    processed_targets = []
    new_target_lengths = []
    
    for targets in seq_class_ids:
        # Trim zeros
        # targets is a tensor, find last non-zero
        non_zero_indices = torch.nonzero(targets)
        if len(non_zero_indices) > 0:
            end_idx = non_zero_indices[-1].item() + 1
            valid_targets = targets[:end_idx]
        else:
            valid_targets = torch.tensor([], dtype=torch.long)
            
        # Add SOS and EOS
        sos_tensor = torch.tensor([SOS_TOKEN], dtype=torch.long)
        eos_tensor = torch.tensor([EOS_TOKEN], dtype=torch.long)
        
        new_seq = torch.cat([sos_tensor, valid_targets, eos_tensor])
        processed_targets.append(new_seq)
        new_target_lengths.append(len(new_seq))
        
    # Pad targets (B x L_max)
    seq_class_ids_padded = torch.nn.utils.rnn.pad_sequence(processed_targets, batch_first=True, padding_value=0)
    
    return input_features_padded, seq_class_ids_padded, torch.tensor(input_lengths), torch.tensor(new_target_lengths), metadatas
