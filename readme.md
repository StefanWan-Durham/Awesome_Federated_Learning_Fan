# ICLR
- FEDERATED LEARNING WITH MATCHED AVERAGING
  - Federated learning allows edge devices to collaboratively learn a shared model
while keeping the training data on device, decoupling the ability to do model
training from the need to store the data in the cloud. We propose the Federated
matched averaging (FedMA) algorithm designed for federated learning of modern neural network architectures e.g. convolutional neural networks (CNNs) and
LSTMs. FedMA constructs the shared global model in a layer-wise manner by
matching and averaging hidden elements (i.e. channels for convolution layers;
hidden states for LSTM; neurons for fully connected layers) with similar feature
extraction signatures. Our experiments indicate that FedMA not only outperforms
popular state-of-the-art federated learning algorithms on deep CNN and LSTM
architectures trained on real world datasets, but also reduces the overall communication burden.1 
