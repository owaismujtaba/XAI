
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from dataset.dataset import BrainDataset, collate_fn
from decode.model import Encoder, Decoder, Attention, Seq2Seq
from torch.utils.data import DataLoader

def translate_sentence(model, src, device, max_len = 50):
    model.eval()
    
    # src = [T, D]
    src_len = torch.tensor([src.shape[0]]).to(device)
    src = src.unsqueeze(0).to(device) # [1, T, D]
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src, src_len)
        mask = model.create_mask(src_len, src.shape[1])
        
        # Start with SOS (41) - Assuming 41 is SOS as per dataset
        # Wait, in dataset.py I defined SOS=41.
        SOS_TOKEN = 41
        EOS_TOKEN = 42
        
        trg_indexes = [SOS_TOKEN]
        attentions = torch.zeros(max_len, 1, src.shape[1]).to(device)
        
        for i in range(max_len):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
            
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
            
            attentions[i] = attention
            
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)
            
            if pred_token == EOS_TOKEN:
                break
                
    return trg_indexes, attentions[:len(trg_indexes)-1]

def display_attention(sentence, translation, attention, save_path):
    # sentence: string or list of phonemes (ground truth?)
    # translation: list of predicted phonemes
    # attention: [trg len, src len]
    
    attention = attention.squeeze(1).cpu().detach().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(attention, cmap='viridis', ax=ax)
    
    ax.set_xlabel('Neural Time Steps')
    ax.set_ylabel('Predicted Phonemes')
    
    plt.title('Attention Weights (XAI)')
    plt.savefig(save_path)
    print(f"Attention plot saved to {save_path}")

if __name__ == "__main__":
    # Config (Must match train.py)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 64
    ENC_HID_DIM = 128
    DEC_HID_DIM = 128
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    INPUT_DIM = 512
    OUTPUT_DIM = 43
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Init Model
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, device).to(device)
    
    # Load Weights if available
    model_path = 'model_epoch_0.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded model weights.")
    else:
        print("No model weights found. Using random initialization.")
        
    # Load Data Sample
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw/hdf5_data_final'))
    # Load just one sample
    dataset = BrainDataset(data_dir, dataset_type='train')
    
    sample_idx = 0
    src, trg, _, _ = dataset[sample_idx]
    
    print(f"Input shape: {src.shape}")
    print(f"Target seq: {trg}")
    
    translation, attention = translate_sentence(model, src, device)
    
    print(f"Predicted seq: {translation}")
    
    display_attention("Input", translation, attention, 'attention.png')
