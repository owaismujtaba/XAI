
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.rnn = nn.GRU(input_dim, enc_hid_dim, bidirectional = True, batch_first=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        # src = [batch size, src len, input_dim]
        # src_len = [batch size]
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(src, src_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        
        # outputs = [batch size, src len, enc hid dim * 2]
        # hidden = [n layers * num directions, batch size, enc hid dim]
        
        # Initial decoder hidden state
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs, mask):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [batch size, src len, enc_hid_dim * 2]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        # attention = [batch size, src len]
        
        attention = attention.masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention, dim = 1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim, batch_first=True)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]
        # mask = [batch size, src len]
        
        input = input.unsqueeze(1) # [batch size, 1]
        embedded = self.dropout(self.embedding(input)) # [batch size, 1, emb dim]
        
        a = self.attention(hidden, encoder_outputs, mask)
        a = a.unsqueeze(1) # [batch size, 1, src len]
        
        weighted = torch.bmm(a, encoder_outputs) # [batch size, 1, enc hid dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output = [batch size, 1, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        
        assert (output == hidden.transpose(0, 1)).all()
        
        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        # prediction = [batch size, output dim]
        
        return prediction, hidden.squeeze(0), a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def create_mask(self, src_len, max_len):
        # src_len = [batch size]
        batch_size = src_len.shape[0]
        mask = torch.arange(max_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        mask = mask < src_len.unsqueeze(1)
        return mask
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):
        # src = [batch size, src len, input dim]
        # src_len = [batch size]
        # trg = [batch size, trg len]
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        attentions = torch.zeros(batch_size, trg_len, src.shape[1]).to(self.device)
        
        encoder_outputs, hidden = self.encoder(src, src_len)
        
        mask = self.create_mask(src_len, src.shape[1])
        
        # First input to the decoder is the <sos> token
        # Since we don't have explicit SOS in data, we might need to assume 0 is SOS/PAD?
        # Data inspection showed 0s are padding at end. Maybe start with 0?
        # But if 0 is padding, maybe we need a dedicated SOS.
        # For now, let's use the first token of target (teacher forcing) or random for first step?
        # Typically we prepend SOS. 
        # I will handle SOS in training loop (prepend).
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, attention = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[:, t] = output
            attentions[:, t] = attention
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            input = trg[:, t] if teacher_force else top1
            
        return outputs, attentions
