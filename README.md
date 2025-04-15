# Physics-Informed Neural Networks with PyTorch

This repository explores **Physics-Informed Neural Networks (PINNs)** using PyTorch.  
PINNs integrate physical laws, expressed as partial differential equations (PDEs), into the training of neural networks.  
This approach is especially useful when data is scarce but the governing physics is well-understood.

## Overview

This project demonstrates how to implement PINNs to solve PDEs by embedding physical constraints directly into the loss function of a neural network.  
The learned solutions not only fit the data but also respect the underlying physical laws.

📝 For a detailed tutorial, check out the accompanying [Medium article](https://medium.com/@theo.wolf/physics-informed-neural-networks-a-simple-tutorial-with-pytorch-f28a890b874a).

## Repository Structure

```
pinns/
├── notebooks/        # Jupyter notebooks with step-by-step tutorials
├── src/              # Core implementation: models, training loops, etc.
├── LICENSE
└── README.md         # You're here!
```

## Installation

### Requirements

Install the required packages using pip:

```
pip install torch scikit-learn numpy matplotlib seaborn
```

### Clone the Repository

```
git clone https://github.com/TheodoreWolf/pinns.git
cd pinns
```

## Usage

To get started, open one of the Jupyter notebooks in the `src/` directory:

```
jupyter notebook src/
```

These walk through training PINNs on various PDEs with visualizations.

Alternatively, run Python scripts in `src/` to train directly via the command line.

## Resources

- 📘 **Medium Tutorial**: [Physics-Informed Neural Networks: A Simple Tutorial with PyTorch](https://medium.com/@theo.wolf/physics-informed-neural-networks-a-simple-tutorial-with-pytorch-f28a890b874a)
- 📄 **Original Paper**: Raissi, Perdikaris, & Karniadakis (2019)  
  [Physics-informed neural networks (PINNs)](https://maziarraissi.github.io/PINNs/)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Inspired by the foundational work by Raissi et al.  
This repo aims to provide an approachable and practical introduction to PINNs with PyTorch.
