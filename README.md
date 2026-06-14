Quantum-Inspired Hybrid Neural Network for Bandgap Energy Prediction

Overview

This project presents a Quantum-Inspired Hybrid Neural Network (QIHNN) designed to predict the bandgap energy of materials using elemental and structural properties. The model combines quantum-inspired feature transformations with classical deep learning techniques to capture complex nonlinear relationships that influence material behavior.

Bandgap prediction plays a crucial role in materials science, semiconductor design, photovoltaic systems, and electronic device development. By leveraging advanced feature engineering and neural network architectures, this project aims to improve prediction accuracy while maintaining computational efficiency.

⸻

Problem Statement

Determining the bandgap energy of materials through experimental methods can be costly and time-consuming. Machine learning provides an alternative approach by learning patterns from known material properties and generating accurate predictions for unseen materials.

The objective of this project is to develop a hybrid neural network capable of estimating bandgap energy from material characteristics such as atomic mass, electronegativity, density, lattice constants, and valence electron information.

⸻

Key Features

* Quantum-inspired neural network architecture
* Hybrid quantum-classical feature learning
* Advanced feature engineering and interaction terms
* Ensemble-based prediction framework
* Automated preprocessing and normalization
* Regression analysis with performance evaluation
* Visualization of actual versus predicted values

⸻

Dataset Features

The model utilizes material descriptors including:

* Atomic Mass
* Electronegativity
* Density
* Lattice Constant
* Valence Electrons

Engineered Features

* Electronegativity²
* Electronegativity × Valence Electrons
* Atomic Mass / Density
* Polynomial interaction features
* Higher-order feature combinations

⸻

Model Architecture

The proposed architecture consists of:

Quantum-Inspired Layer

A custom nonlinear transformation layer designed to simulate quantum interference-like feature interactions and enhance representation learning.

Classical Neural Network

A fully connected deep neural network responsible for learning higher-level feature relationships and performing regression.

Feature Fusion

Outputs from both components are combined to create richer feature representations before final prediction.

Regularization Techniques

* Dropout Layers
* Batch Normalization
* Ensemble Learning

These methods help improve generalization and reduce overfitting.

⸻

Technology Stack

* Python
* PyTorch
* NumPy
* Scikit-learn
* Matplotlib

⸻

Performance

The model demonstrates strong predictive capability for bandgap estimation with:

* Mean Squared Error (MSE): ~0.02 – 0.04
* Mean Absolute Error (MAE): ~0.10 – 0.15 eV
* R² Score: ~0.92 – 0.96

Performance is evaluated using standard regression metrics and visualization techniques.

⸻

Project Structure

QNN-for-Bandgap-Estimation/
│
├── QNN_code.py
├── README.md
└── Results/
    └── Prediction_Visualization.png

⸻

Installation

Clone the repository:

git clone https://github.com/yourusername/QNN-for-Bandgap-Estimation.git

Navigate to the project directory:

cd QNN-for-Bandgap-Estimation

Install dependencies:

pip install torch numpy pandas scikit-learn matplotlib

⸻

Usage

Run the training and evaluation script:

python QNN_code.py

The script will:

1. Load and preprocess the dataset
2. Generate engineered features
3. Train the hybrid neural network
4. Evaluate model performance
5. Visualize prediction results

⸻

Applications

* Semiconductor Material Discovery
* Solar Cell Research
* Materials Informatics
* Electronic Device Design
* Computational Materials Science

⸻

Future Enhancements

* Integration with real-world materials databases
* Hyperparameter optimization
* Explainable AI techniques for material analysis
* Quantum computing framework integration (PennyLane, Qiskit)
* Deployment as a web-based prediction tool

⸻

Contributors

Developed as part of a collaborative machine learning project focused on applying AI techniques to materials science and bandgap estimation.

⸻

License

This project is intended for academic and educational purposes.
