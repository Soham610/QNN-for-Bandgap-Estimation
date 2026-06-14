import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic dataset
n_samples = 500
data = {
    "AtomicMass": np.random.uniform(10, 210, n_samples),
    "Electronegativity": np.random.uniform(0.5, 4.0, n_samples),
    "LatticeConstant": np.random.uniform(2.0, 6.0, n_samples),
    "Density": np.random.uniform(1.0, 12.0, n_samples),
    "ValenceElectrons": np.random.randint(1, 9, n_samples),
}

df = pd.DataFrame(data)
noise = np.random.normal(0, 0.15, n_samples)
df["Bandgap"] = (0.8 * df["Electronegativity"] - 0.1 * df["AtomicMass"]/100 +
                 0.3 * df["ValenceElectrons"] + 0.2 * df["LatticeConstant"] + noise)
df["Bandgap"] = df["Bandgap"].clip(0, 3.5)

# Feature engineering
df["ElectronegativitySquared"] = df["Electronegativity"]**2
df["ElectroValenceInteraction"] = df["Electronegativity"] * df["ValenceElectrons"]
df["MassDensityRatio"] = df["AtomicMass"] / df["Density"]

# Prepare data
X = df.drop("Bandgap", axis=1).values
y = df["Bandgap"].values

# Add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_poly = poly.fit_transform(X)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Split with stratification
y_bins = pd.qcut(df["Bandgap"], 5, labels=False, duplicates='drop')
X_train, X_test, y_train, y_test, bins_train, bins_test = train_test_split(
    X_scaled, y, y_bins, test_size=0.2, random_state=42, stratify=y_bins
)

# Convert to torch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32)

# Model definition
class EnhancedHybridModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        
        self.quantum_layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(),
            nn.Dropout(0.2)
        )
        self.quantum_layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.SiLU(),
            nn.Dropout(0.2)
        )
        self.interference = nn.Sequential(
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.Tanh()
        )
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.layer2 = nn.Linear(hidden_size + hidden_size//2, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.layer3 = nn.Linear(hidden_size + hidden_size//4, hidden_size//2)
        self.bn3 = nn.BatchNorm1d(hidden_size//2)
        self.output = nn.Sequential(
            nn.Linear(hidden_size//2, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        q1 = self.quantum_layer1(x)
        q2 = self.quantum_layer2(q1)
        q_interference = self.interference(q2)

        x1 = self.act(self.bn1(self.layer1(x)))
        x1 = self.dropout(x1)

        x2 = torch.cat([x1, q2], dim=1)
        x2 = self.act(self.bn2(self.layer2(x2)))
        x2 = self.dropout(x2)

        x3 = torch.cat([x2, q_interference], dim=1)
        x3 = self.act(self.bn3(self.layer3(x3)))

        out = self.output(x3)
        return out.squeeze()

# Model training
n_models = 5
models = [EnhancedHybridModel(input_size=X_train.shape[1]) for _ in range(n_models)]
optimizers = [torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4) for model in models]
schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5) for opt in optimizers]
loss_fn = nn.MSELoss()

X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_val_torch = torch.tensor(X_val, dtype=torch.float32)
y_val_torch = torch.tensor(y_val, dtype=torch.float32)

batch_size = 32
train_dataset = torch.utils.data.TensorDataset(X_train_torch, y_train_torch)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

best_models = []
for model_idx, (model, optimizer, scheduler) in enumerate(zip(models, optimizers, schedulers)):
    print(f"Training model {model_idx+1}/{n_models}")
    
    epochs = 150
    best_val_loss = float('inf')
    patience = 15
    counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_torch)
            val_loss = loss_fn(val_outputs, y_val_torch)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model_state = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

    model.load_state_dict(best_model_state)
    best_models.append(model)

# Evaluation
all_preds = []
for model in best_models:
    model.eval()
    with torch.no_grad():
        preds = model(X_test_torch).numpy()
        all_preds.append(preds)

y_pred = np.mean(all_preds, axis=0)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nEvaluation:")
print(f"Test MSE: {mse:.4f}")
print(f"Test R²: {r2:.4f}")
print(f"Test MAE: {mae:.4f}")

# Plotting
plt.figure(figsize=(10, 8))
errors = np.abs(y_test - y_pred)
scatter = plt.scatter(y_test, y_pred, c=errors, cmap='viridis_r', 
                      s=80, edgecolor='k', alpha=0.8)

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
         'r--', linewidth=2, label='Perfect Prediction')

from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_pred)
regression_line = slope * np.array([min(y_test), max(y_test)]) + intercept
plt.plot([min(y_test), max(y_test)], regression_line, 'b-', 
         linewidth=2, label=f'Regression Line (R²={r2:.2f})')

plt.fill_between([min(y_test), max(y_test)], 
                 [min(y_test) - mae, max(y_test) - mae],
                 [min(y_test) + mae, max(y_test) + mae], 
                 color='red', alpha=0.1, label=f'MAE Band (±{mae:.2f} eV)')

x_min, x_max = min(y_test) - 0.2, max(y_test) + 0.2
y_min, y_max = min(y_pred) - 0.2, max(y_pred) + 0.2
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

cbar = plt.colorbar(scatter)
cbar.set_label('Absolute Error (eV)', fontsize=12)

plt.text(x_min + 0.05, y_max - 0.15, 
         f"MSE: {mse:.4f}\nR²: {r2:.4f}\nMAE: {mae:.4f}", 
         fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

plt.title("Enhanced Hybrid Model for Bandgap Prediction", fontsize=16)
plt.xlabel("Actual Bandgap (eV)", fontsize=14)
plt.ylabel("Predicted Bandgap (eV)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("enhanced_bandgap_prediction.png", dpi=300)
plt.show()
