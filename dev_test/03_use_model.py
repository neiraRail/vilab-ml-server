import mlflow
import torch
import requests


# Request all measures from api
measures = requests.get("http://localhost:8080/measure/67254752bfe9298568024447").json()


si = measures[5].get("si")
so = measures[5].get("so")

# Request data from api
data = requests.get(f"http://localhost:8080/measure/data/1?si={si}&so={so}").json()

import pandas as pd
df = pd.DataFrame(data[-301:-201])
print(len(df))
df = df.drop(["mx", "my", "mz", "st", "nd", "tm"], axis=1)
ordered_columns = ["ax", "ay", "az", "gx", "gy", "gz", "tp", "dt"]
df = df[ordered_columns]


# # Scale data using scaler
import pickle
scaler = pickle.load(open("scaler_2_classes_high.pkl", "rb"))
df = scaler.transform(df)

# Transform data to tensor
df = torch.tensor(df, dtype=torch.float32)

# Apply autoencoder
autoencoder = mlflow.pytorch.load_model("models:/autoencoder_lstm/1")
autoencoder.eval()
pred = autoencoder.encoder(df.unsqueeze(0))
# print(pred[0])

# Apply classifier
classifier = mlflow.pytorch.load_model("models:/classifier/1")
classifier.eval()
pred = classifier(pred)
_, pred = torch.max(pred, 1)
print(pred)