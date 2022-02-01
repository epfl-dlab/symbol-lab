from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        
        layers = nn.ModuleList([self.embedding, self.rnn, self.dropout])
  
    def forward(self, src):
        
        # src = [src_len, batch_size]
        
        embedded = self.dropout(self.embedding(src))
        # embedded = [src_len, batch_size, emb_dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        # outputs = [src_len, batch_size, hid_dim * n_directions]
        # hidden = [n_layers * n_directions, batch_size, hid_dim]
        # cell = [n_layers * n_directions, batch_size, hid_dim]
        
        # outputs are always from the top hidden layer
        
        return hidden, cell
