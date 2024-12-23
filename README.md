# stac (Spiking Transformer Augmenting Cognition)

## Overview

This project explores integrating Spiking Neural Networks (SNNs) with transformer architectures for language modeling. Specifically, it implements a novel approach combining an adaptive conductance-based spiking neuron model (AdEx) with a pre-trained GPT-2 transformer. 

## Key Features

* **Spiking Neural Network Integration:** Leverages the AdEx neuron model to introduce spiking dynamics into the language model.
* **Adaptive Conductance:** The AdEx neuron's adaptive conductance mechanism allows for more biologically realistic and potentially efficient computation.
* **Transformer-based Architecture:** Builds upon the powerful GPT-2 transformer model for language understanding and generation.
* **Wikitext-2 Dataset:** Trained and evaluated on the Wikitext-2 dataset for text generation tasks.
* **Weights & Biases Integration:** Uses Weights & Biases for experiment tracking and visualization.
