
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dataset.dataset import BrainDataset, collate_fn
from decode.model import Encoder, Decoder, Attention, Seq2Seq
# We can import train/evaluate from train.py if we want, or just re-implement simple loops here
from decode.train import train, evaluate
from decode.inference import translate_sentence, display_attention

def main():
    # Load Config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    mode = config.get('mode', 'train')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running in {mode} mode on {device}")

    m_cfg = config['model']
    attn = Attention(m_cfg['enc_hid_dim'], m_cfg['dec_hid_dim'])
    enc = Encoder(m_cfg['input_dim'], m_cfg['enc_hid_dim'], m_cfg['dec_hid_dim'], m_cfg['enc_dropout'])
    dec = Decoder(m_cfg['output_dim'], m_cfg['dec_emb_dim'], m_cfg['enc_hid_dim'], m_cfg['dec_hid_dim'], m_cfg['dec_dropout'], attn)
    model = Seq2Seq(enc, dec, device).to(device)

    if mode == 'train':
        t_cfg = config['training']
        data_dir = config['data']['data_dir']
        
        train_dataset = BrainDataset(data_dir, dataset_type='train')
        train_loader = DataLoader(train_dataset, batch_size=t_cfg['batch_size'], shuffle=True, collate_fn=collate_fn)
        
        val_dataset = BrainDataset(data_dir, dataset_type='val')
        val_loader = DataLoader(val_dataset, batch_size=t_cfg['batch_size'], shuffle=False, collate_fn=collate_fn)
        
        optimizer = optim.Adam(model.parameters(), lr=t_cfg.get('learning_rate', 0.001))
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        print(f"Starting training for {t_cfg['n_epochs']} epochs...")
        
        for epoch in range(t_cfg['n_epochs']):
            train_loss = train(model, train_loader, optimizer, criterion, t_cfg['clip'])
            val_loss = evaluate(model, val_loader, criterion)
            
            print(f'Epoch: {epoch+1:02}')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\t Val. Loss: {val_loss:.3f}')
            
            torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pt')

    elif mode == 'inference':
        i_cfg = config['inference']
        data_dir = config['data']['data_dir']
        
        if os.path.exists(i_cfg['checkpoint_path']):
            model.load_state_dict(torch.load(i_cfg['checkpoint_path'], map_location=device))
            print(f"Loaded checkpoint: {i_cfg['checkpoint_path']}")
        else:
            print(f"WARNING: Checkpoint {i_cfg['checkpoint_path']} not found. Using random weights.")

        dataset = BrainDataset(data_dir, dataset_type='train') # Or val/test
        sample_idx = i_cfg['sample_idx']
        src, trg, _, _, _ = dataset[sample_idx]
        
        print(f"Sample Index: {sample_idx}")
        print(f"Input shape: {src.shape}")
        
        translation, attention = translate_sentence(model, src, device, max_len=i_cfg['max_len'])
        print(f"Ground Truth: {trg[:20].tolist()} ...")
        print(f"Predicted: {translation}")
        
        display_attention("Input", translation, attention, i_cfg['output_plot'])

if __name__ == "__main__":
    main()
