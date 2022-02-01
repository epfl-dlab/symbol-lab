from torch import nn

class Decoder(nn.Module):
    def __init__(self, output_dim: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float):
        
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        layers = nn.ModuleList([self.embedding, self.rnn, self.fc_out, self.dropout])
        
    def forward(self, input_seq, hidden, cell):
        
        '''
            input_seq: torch.Tensor [batch_size]
            hidden: torch.Tensor [n_layers * n_directions, batch_size, hid_dim]
            cell: torch.Tensor [n_layers * n_directions, batch_size, hid_dim]
            
            n_directions in the decoder will both always be 1, therefore:
            hidden = [n_layers, batch_size, hid_dim]
            context = [n_layers, batch_size, hid_dim]
        '''
        
        
        
        input_seq = input_seq.unsqueeze(0)
        # input: [1, batch_size]
        
        
        embedded = self.dropout(self.embedding(input_seq))
        # embedded = [1, batch_size, emb_dim]
                
        output, (new_hidden, new_cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq_len, batch_size, hid_dim * n_directions]
        # hidden = [n_layers * n_directions, batch_size, hid_dim]
        # cell = [n_layers * n_directions, batch_size, hid_dim]
        
        # seq_len and n_directions will always be 1 in the decoder, therefore:
        # output = [1, batch_size, hid_dim]
        # hidden = [n_layers, batch_size, hid_dim]
        # cell = [n_layers, batch_size, hid_dim]
        
        prediction = self.fc_out(output.squeeze(0))
        
        # prediction = [batch_size, output_dim]
        
        return prediction, new_hidden, new_cell



