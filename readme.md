# ICLR2020
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
- [Federated Continual Learning with Weighted Inter-client Transfer](https://openreview.net/pdf?id=xWr8qQCJU3m)
  - There has been a surge of interest in continual learning and federated learning, both of which are important in deep neural networks in real-world scenarios. Yet little research has been done regarding the scenario where each client learns on a sequence of tasks from private local data stream. This problem of federated continual learning poses new challenges to continual learning, such as utilizing knowledge from other clients, while preventing interference from irrelevant knowledge. To resolve these issues, we propose a novel federated continual learning framework, Weighted Inter-client Transfer (FedWeIT), which decomposes the network weights into global federated parameters and sparse task-specific parameters, and each client receives selective knowledge from other clients by taking a weighted combination of their task-specific parameters. FedWeIT minimizes interference between incompatible tasks, and also allows positive knowledge transfer across clients during learning. We validate our FedWeIT against existing federated learning and continual learning methods under varying degree of task similarity across clients, and our model significantly outperforms them with large reduction in the communication cost.
- [Federated Residual Learning](https://arxiv.org/pdf/2003.12880.pdf)
  -  We study a new form of federated learning where
the clients train personalized local models and
make predictions jointly with the server-side
shared model. Using this new federated learning
framework, the complexity of the central shared
model can be minimized while still gaining all the
performance benefits that joint training provides.
Our framework is robust to data heterogeneity,
addressing the slow convergence problem traditional federated learning methods face when the
data is non-i.i.d. across clients. We test the theory empirically and find substantial performance
gains over baselines.
# ICLR 2021
- [END-TO-END ON-DEVICE FEDERATED LEARNING: A CASE STUDY](https://openreview.net/pdf?id=VyDYSMx1sFU)
  -   With the development of computation capability in devices, companies are eager
to utilize ML/DL methods to improve their service quality. However, with traditional Machine Learning approaches, companies need to build up a powerful
data center to collect data and perform centralized model training, which turns
out to be expensive and inefficient. Federated Learning has been introduced to
solve this challenge. Because of its characteristics such as model-only exchange
and parallel training, the technique can not only preserve user data privacy but
also accelerate model training speed. In this paper, we introduce an approach to
end-to-end on-device Machine Learning by utilizing Federated Learning. We validate our approach with an important industrial use case, the wheel steering angle
prediction in the field of autonomous driving. Our results show that Federated
Learning can significantly improve the quality of local edge models and reach the
same accuracy level as compared to the traditional centralized Machine Learning approach without its negative effects. Furthermore, Federated Learning can
accelerate model training speed and reduce the communication overhead, which
proves that this approach has great strength when deploying ML/DL components
to real-world embedded systems.
- 
