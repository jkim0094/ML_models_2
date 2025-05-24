# Document Clustering, Neural Networks, and Self-supervised Learning

This repository contains three machine learning tasks, focusing on document clustering, binary classification with neural networks, and self-supervised learning using autoencoders. All implementations are done in Python using Jupyter Notebooks.

## Section 1: Document Clustering

* Solve a document clustering problem using the Expectation-Maximization (EM) algorithm
* Derive and implement the mathematical formulation of the EM algorithm (E-step and M-step)
* Apply both hard and soft EM clustering on a given text dataset
* Perform Principal Component Analysis (PCA) and visualize clusters in 2D space
* Compare and discuss the differences between hard and soft clustering

### Dataset

* **File**: `Task2A.txt`
* **Description**: A collection of unlabeled text documents used for unsupervised clustering.
* **Format**: Plain text file, one document per line
* **Purpose**: Used to cluster documents into 4 groups using the Expectation-Maximization (EM) algorithm.

## Section 2: Perceptron vs Neural Networks

* Solve a binary classification task using a Perceptron and a 3-layer Neural Network
* Experiment with different learning rates (η) and hidden layer sizes (K)
* Record and compare model performance across configurations
* Visualize decision boundaries and analyze differences between the Perceptron and Neural Network models

### Dataset

* **Files**: `Task2B_train.csv`, `Task2B_test.csv`
* **Description**: A synthetic 2D dataset for binary classification tasks.
* **Format**: CSV files containing two input features and a binary label
* **Purpose**: Used to train and evaluate Perceptron and 3-layer Neural Network models, and to visualize their decision boundaries.

## Section 3: Self-supervised Learning with Autoencoder

* Use both labeled and unlabeled data to train an Autoencoder
* Vary the number of neurons in the hidden layer and analyze reconstruction errors
* Build a 3-layer Neural Network using only original features and evaluate test error
* Create augmented models by combining original features with autoencoder embeddings (self-taught learning)
* Compare performance between baseline and augmented models, and analyze the impact of learned representations

### Dataset

* **Files**: `Task2C_labeled.csv`, `Task2C_unlabeled.csv`, `Task2C_test.csv`
* **Description**:

  * `labeled.csv`: Multiclass labeled training data
  * `unlabeled.csv`: Unlabeled data for training the autoencoder
  * `test.csv`: Test data for evaluating classification performance
* **Format**: CSV files with numerical input features (labels included where applicable)
* **Purpose**: Used to perform self-taught learning by extracting hidden representations via an autoencoder and comparing performance with and without additional features.

## Notes

* All results are presented in Jupyter Notebooks with clear structure, markdown explanations, and visualizations.
* Only permitted libraries such as NumPy, SciPy, scikit-learn, matplotlib, and PyTorch (for Section 3) are used.


