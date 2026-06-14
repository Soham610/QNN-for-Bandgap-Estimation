
# 🧠🔬 Enhanced Hybrid Neural Network for Bandgap Prediction

This project presents an **ensemble of enhanced hybrid neural networks**—inspired by quantum interference mechanisms—for accurate **bandgap energy prediction** from elemental and structural properties. The dataset is synthetically generated and augmented with polynomial and interaction features to simulate real-world scenarios.

<p align="center">
  <img src="enhanced_bandgap_prediction.png" width="600"/>
</p>

## 🚀 Overview

Accurate prediction of bandgap energy is crucial in material science for applications like semiconductors and solar cells. This model leverages a hybrid deep learning approach mimicking **quantum-inspired layers**, combined with ensemble learning, to deliver robust regression performance.

---

## 📁 Project Structure

```
📦bandgap-predictor
 ┣ 📜main.py              # Main training and evaluation script
 ┣ 📜README.md            # This file
 ┗ 📊enhanced_bandgap_prediction.png  # Visualization of predictions
```

---

## 🧩 Features Used

- `Atomic Mass`
- `Electronegativity`
- `Lattice Constant`
- `Density`
- `Valence Electrons`
- **Engineered Features**:
  - Electronegativity²
  - Electronegativity × ValenceElectrons
  - AtomicMass / Density
  - 2nd-degree interaction terms via `PolynomialFeatures`

---

## 🛠️ Technologies Used

- Python 🐍
- PyTorch 🔥
- Scikit-learn 🧪
- Matplotlib 📊
- NumPy 🧮

---

## 📊 Model Architecture

The hybrid model includes:

- **Quantum-inspired subnetwork** (nonlinear projections, interference layer)
- **Classical feed-forward network** (multi-layer perceptron)
- **Feature fusion via concatenation**
- Dropout and Batch Normalization for regularization
- Ensemble of 5 models for robust prediction

---

## 📈 Results

- ✅ **Test MSE**: ~0.02 - 0.04  
- ✅ **R² Score**: ~0.92 - 0.96  
- ✅ **MAE**: ~0.10 - 0.15 eV  

Includes MAE band and regression fit on actual vs predicted bandgap values.

---

## 📦 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bandgap-predictor.git
   cd bandgap-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the model:
   ```bash
   python main.py
   ```

> Ensure Python 3.8+ and PyTorch are properly installed.

---

## 📚 Future Work

- Extend to real materials datasets (e.g., Materials Project)
- Incorporate domain knowledge into feature extraction
- Use real quantum layers (e.g., PennyLane, Qiskit integration)
- Hyperparameter optimization and model uncertainty quantification

---

## 🧠 Inspiration

This work is inspired by hybrid quantum neural network research and the need for physics-informed models in materials science.

---

## 📝 License

This project is licensed under the MIT License.
