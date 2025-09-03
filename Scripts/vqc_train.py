# vqc_train.py
# Directory: Scripts/
# Project: Stellar Classification with Gaia DR3 Using Variational Quantum Classifiers

import os
import logging
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import pennylane as qml
from pennylane import numpy as npqml
import numpy as np

# Path setup for cross-platform compatibility
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJ_ROOT, "Datasets", "Processed")
RESULTS_DIR = os.path.join(PROJ_ROOT, "Results")
LOGS_DIR = os.path.join(PROJ_ROOT, "logs")

# -------------------------------
# 1. Setup Logging
# -------------------------------
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "vqc_train.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.info("Starting VQC training script")

# -------------------------------
# 2. Load Processed Dataset
# -------------------------------
processed_csv = os.path.join(DATA_DIR, "gaia_features_20.csv")
if not os.path.exists(processed_csv):
    logging.error(f"{processed_csv} not found. Run gaia_feature_selection.py first.")
    raise FileNotFoundError(f"{processed_csv} not found.")

df = pd.read_csv(processed_csv)
logging.info(f"Loaded dataset shape: {df.shape}")

# Features and labels
X = df.drop(columns=["label"]).values
y = df["label"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to Torch tensors
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

logging.info("Data preprocessing complete")

# -------------------------------
# 3. Define Quantum Circuit
# -------------------------------
n_qubits = 4
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

def angle_embedding(x, wires):
    x = np.array(x)  # Use regular numpy instead of npqml
    # truncate or pad features
    if len(x) < len(wires):
        x = np.pad(x, (0, len(wires) - len(x)))
    elif len(x) > len(wires):
        x = x[:len(wires)]
    for i, wire in enumerate(wires):
        qml.RY(x[i], wires=wire)

@qml.qnode(dev, interface="torch")
def circuit(inputs, weights):
    angle_embedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_layers, n_qubits)}
qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

# -------------------------------
# 4. Define VQC Model (PyTorch)
# -------------------------------
class VQCClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer = qlayer
        self.fc = nn.Linear(n_qubits, 2)

    def forward(self, x):
        x = self.qlayer(x)
        return self.fc(x)

model = VQCClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

logging.info("Model initialized")

# -------------------------------
# 5. Training Loop
# -------------------------------
n_epochs = 50
for epoch in range(n_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = loss_fn(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        logging.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

logging.info("Training complete")

# -------------------------------
# 6. Evaluate Model
# -------------------------------
with torch.no_grad():
    y_pred_t = model(X_test_t)
    y_pred = torch.argmax(y_pred_t, dim=1).numpy()

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

logging.info(f"Test Accuracy: {acc:.4f}")
logging.info(f"Confusion Matrix:\n{cm}")
print(f"\nTest Accuracy: {acc:.4f}")
print("Confusion Matrix:")
print(cm)

# -------------------------------
# 7. Save Results
# -------------------------------
os.makedirs(RESULTS_DIR, exist_ok=True)

# Save predictions
pred_df = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred
})
pred_csv = os.path.join(RESULTS_DIR, "vqc_predictions.csv")
pred_df.to_csv(pred_csv, index=False)
logging.info(f"Predictions saved to {pred_csv}")

# Save model weights
model_path = os.path.join(RESULTS_DIR, "vqc_model.pt")
torch.save(model.state_dict(), model_path)
logging.info(f"Model weights saved to {model_path}")

print("Training complete. Predictions and model saved in Results/")
