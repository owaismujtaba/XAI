
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import yaml

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from dataset.dataset import BrainDataset, collate_fn
from decode.model import Encoder, Decoder, Attention, Seq2Seq

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    device = model.device
    for i, batch in enumerate(iterator):
        src, trg, src_len, trg_len = batch
        # src: [B, T, D]
        # trg: [B, L]
        
        src = src.to(device)
        trg = trg.to(device)
        src_len = src_len.to(device)
        
        optimizer.zero_grad()
        
        output, _ = model(src, src_len, trg)
        
        # trg = [batch size, trg len]
        # output = [batch size, trg len, output dim]
        
        output_dim = output.shape[-1]
        
        output = output[:,1:].reshape(-1, output_dim)
        trg = trg[:,1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if i % 10 == 0:
            print(f'Step {i} | Loss: {loss.item():.3f}')
            
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    device = model.device
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, trg, src_len, trg_len = batch
            src = src.to(device)
            trg = trg.to(device)
            src_len = src_len.to(device)

            output, _ = model(src, src_len, trg, 0) # turn off teacher forcing
            
            output_dim = output.shape[-1]
            output = output[:,1:].reshape(-1, output_dim)
            trg = trg[:,1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


