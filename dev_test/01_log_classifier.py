import mlflow
import torch
from modelos import Classifier

embedding_dim = 256
num_classes = 2
model = Classifier(embedding_dim, num_classes)

# load pytorch model and save it to model registry using mlflow.
state_dict = torch.load("classifier.pth")["model_state_dict"]
model.load_state_dict(state_dict)


mlflow.pytorch.log_model(model, "classifier", registered_model_name="classifier")

