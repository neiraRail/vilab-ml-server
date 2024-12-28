import mlflow
import torch
from modelos import LSTMAutoencoder


seq_len = 100
n_features = 8
embedding_dim = 256
model = LSTMAutoencoder(n_features, 128, embedding_dim, seq_len)
model = model
state_model = torch.load("./autoencoder_lstm_01_best.pth", map_location=torch.device("cpu"))["model_state_dict"]

model.load_state_dict(state_model)

mlflow.pytorch.log_model(model, "autoencoder", registered_model_name="autoencoder_lstm")