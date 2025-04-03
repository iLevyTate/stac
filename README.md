# stac (Spiking Transformer Augmenting Cognition)

[![DOI](https://zenodo.org/badge/907152074.svg)](https://doi.org/10.5281/zenodo.14545340)

<div align="center">
  
![STAC](https://github.com/user-attachments/assets/1ea4cc68-0cbe-40bf-805f-94b78080bf15)

[Google Colab notebook](https://colab.research.google.com/drive/1BNmGuqcRaC9hnhxU7DdL9yA-lnFfnCAR?usp=sharing)

</div>


## Overview

This project explores integrating Spiking Neural Networks (SNNs) with transformer architectures for language modeling. Specifically, it implements a novel approach combining an adaptive conductance-based spiking neuron model (AdEx) with a pre-trained GPT-2 transformer. 

## Key Features

* **Spiking Neural Network Integration:** Leverages the AdEx neuron model to introduce spiking dynamics into the language model.
* **Adaptive Conductance:** The AdEx neuron's adaptive conductance mechanism allows for more biologically realistic and potentially efficient computation.
* **Transformer-based Architecture:** Builds upon the powerful GPT-2 transformer model for language understanding and generation.
* **Wikitext-2 Dataset:** Trained and evaluated on the Wikitext-2 dataset for text generation tasks.
* **Weights & Biases Integration:** Uses Weights & Biases for experiment tracking and visualization.
