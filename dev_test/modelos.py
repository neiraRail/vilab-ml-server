import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

# Modelo Autoencoder con uso de vector latente de encoder como hidden state de decoder.
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.hidden_to_latent = nn.Linear(hidden_size, latent_size)  # Transform hidden state to latent size

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)  # hidden: [1, batch_size, hidden_size]
        latent = self.hidden_to_latent(hidden[-1])  # latent: [batch_size, latent_size]
        return latent
    
class LSTMDecoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(LSTMDecoder, self).__init__()
        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)  # Transform latent to hidden size
        self.lstm = nn.LSTM(output_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Final output layer
        self.output_size = output_size

    def forward(self, latent, sequence_length):
        hidden = self.latent_to_hidden(latent).unsqueeze(0)  # Reshape to [1, batch_size, hidden_size]
        cell = torch.zeros_like(hidden)  # Initialize cell state as zeros
        outputs = []
        input_seq = torch.zeros((latent.size(0), 1, self.output_size)).to(device)  # Starting input for the decoder (adjust as needed)

        for _ in range(sequence_length):
            output, (hidden, cell) = self.lstm(input_seq, (hidden, cell))
            input_seq = self.fc(output)  # Pass output through the fully connected layer
            outputs.append(input_seq)

        return torch.cat(outputs, dim=1)
    
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, sequence_length):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, latent_size)
        self.decoder = LSTMDecoder(latent_size, hidden_size, input_size)
        self.sequence_length = sequence_length

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent, self.sequence_length)
        return output


# Clasificador simple
class Classifier(nn.Module):
  def __init__(self, embedding_size, num_classes):
    super().__init__()
    self.classifier = nn.Sequential(
        nn.Linear(embedding_size, num_classes),
        # nn.LogSoftmax(1)
    )


  def forward(self, x):
    # print(f"shape of hidden: {hidden.shape}")
    return self.classifier(x)# TODO test with multiple LSTM layers
  
# Autoencoder que usa el hidden del encoder repetido como input del decoder y los estados ocultos son inicializados en zeros.
class Encoder(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()

        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn1 = nn.LSTM(
          input_size=input_size,
          hidden_size=hidden_size,
          num_layers=num_layers,
          batch_first=True  # True = (batch_size, seq_len, n_features)
                            # False = (seq_len, batch_size, n_features)
                            #default = false
        )

    def forward(self, x):
        # print(f'ENCODER input dim: {x.shape}')
        # x = x.reshape((batch_size, self.seq_len, self.input_size))
        # print(f'ENCODER reshaped dim: {x.shape}')
        _, (hidden, cell) = self.rnn1(x)
        return hidden[-1]
class Decoder(nn.Module):
    def __init__(self, seq_len, embedding_size, n_features, num_layers):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.n_features = n_features
        self.num_layers = num_layers

        self.rnn1 = nn.LSTM(
          input_size=embedding_size,
          hidden_size=n_features,
          num_layers=num_layers,
          batch_first=True
        )
        # Version 1, va directo a la dimensión de destino
        # self.output_layer = nn.Linear(self.embedding_size, n_features)

    def forward(self, hidden):
        # Repetir el vector oculto 100 veces
        inputs = hidden.unsqueeze(1).repeat(1, self.seq_len, 1)
        cell = torch.zeros(self.num_layers, hidden.shape[0], self.n_features).to(device)
        new_hidden = torch.zeros(self.num_layers, hidden.shape[0], self.n_features).to(device)
        # inputs = torch.zeros(batch_size, self.seq_len, hidden.shape[2]).to(device)
        # outputs = torch.zeros(batch_size, self.seq_len, input_size).to(device)

        output, (_, _) = self.rnn1(inputs, (new_hidden, cell))
        return output
class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_size, num_layers):
        super(RecurrentAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_size = embedding_size

        self.encoder = Encoder(seq_len, n_features, embedding_size, num_layers).to(device)
        self.decoder = Decoder(seq_len, embedding_size, n_features, num_layers).to(device)
    def forward(self, x):
        hidden = self.encoder(x)

        outputs = self.decoder(hidden)
        return outputs