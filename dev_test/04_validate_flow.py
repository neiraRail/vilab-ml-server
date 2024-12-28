import mlflow
import torch
import requests


# Read csv with validation data
import pandas as pd
df_total = pd.read_csv("validacion4.csv")


# # Scale data using scaler
import pickle
scaler = pickle.load(open("scaler_2_classes_high.pkl", "rb"))

autoencoder = mlflow.pytorch.load_model("models:/autoencoder_lstm/1")
autoencoder.eval()
classifier = mlflow.pytorch.load_model("models:/classifier/1")
classifier.eval()

preds = []
for i in range(40):
    print("Iteracion: ", i)
    start = i * 100
    end = start + 100
    df = df_total[start:end]

    df = df.drop(["mx", "my", "mz", "st", "nd", "tm"], axis=1)
    ordered_columns = ["ax", "ay", "az", "gx", "gy", "gz", "tp", "dt"]
    df = df[ordered_columns]


    df = scaler.transform(df)

    # Transform data to tensor
    df = torch.tensor(df, dtype=torch.float32)

    # Apply autoencoder
    pred = autoencoder.encoder(df.unsqueeze(0))
    # print(pred[0])

    # Apply classifier
    pred = classifier(pred)
    _, pred = torch.max(pred, 1)
    preds.append(pred.item())


print(preds)