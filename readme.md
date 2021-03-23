# ICLR
- [FEDERATED LEARNING WITH MATCHED AVERAGING](https://openreview.net/pdf?id=BkluqlSFDS)
  - Federated learning allows edge devices to collaboratively learn a shared model
while keeping the training data on device, decoupling the ability to do model
training from the need to store the data in the cloud. We propose the Federated
matched averaging (FedMA) algorithm designed for federated learning of modern neural network architectures e.g. convolutional neural networks (CNNs) and
LSTMs. FedMA constructs the shared global model in a layer-wise manner by
matching and averaging hidden elements (i.e. channels for convolution layers;
hidden states for LSTM; neurons for fully connected layers) with similar feature
extraction signatures. Our experiments indicate that FedMA not only outperforms
popular state-of-the-art federated learning algorithms on deep CNN and LSTM
architectures trained on real world datasets, but also reduces the overall communication burden.
- [DIFFERENTIALLY PRIVATE META-LEARNING](https://openreview.net/pdf?id=rJgqMRVYvr)
  - Parameter-transfer is a well-known and versatile approach for meta-learning, with
applications including few-shot learning, federated learning, and reinforcement
learning. However, parameter-transfer algorithms often require sharing models that
have been trained on the samples from specific tasks, thus leaving the task-owners
susceptible to breaches of privacy. We conduct the first formal study of privacy
in this setting and formalize the notion of task-global differential privacy as a
practical relaxation of more commonly studied threat models. We then propose a
new differentially private algorithm for gradient-based parameter transfer that not
only satisfies this privacy requirement but also retains provable transfer learning
guarantees in convex settings. Empirically, we apply our analysis to the problems
of federated learning with personalization and few-shot classification, showing that
allowing the relaxation to task-global privacy from the more commonly studied
notion of local privacy leads to dramatically increased performance in recurrent
neural language modeling and image classification.  
- [ON THE CONVERGENCE OF FEDAVG ON NON-IID DATA](https://openreview.net/pdf?id=HJxNAnVtDS)
  - Federated learning enables a large amount of edge computing devices to jointly
learn a model without data sharing. As a leading algorithm in this setting, Federated
Averaging (FedAvg) runs Stochastic Gradient Descent (SGD) in parallel on a small
subset of the total devices and averages the sequences only once in a while. Despite
its simplicity, it lacks theoretical guarantees under realistic settings. In this paper,
we analyze the convergence of FedAvg on non-iid data and establish a convergence
rate of O(1T) for strongly convex and smooth problems, where T is the number of
SGDs. Importantly, our bound demonstrates a trade-off between communicationefficiency and convergence rate. As user devices may be disconnected from
the server, we relax the assumption of full device participation to partial device
participation and study different averaging schemes; low device participation rate
can be achieved without severely slowing down the learning. Our results indicates
that heterogeneity of data slows down the convergence, which matches empirical
observations. Furthermore, we provide a necessary condition for FedAvg on
non-iid data: the learning rate η must decay, even if full-gradient is used; otherwise,
the solution will be Ω(η) away from the optimal. 
