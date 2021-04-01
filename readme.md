
# 2021-04-01 (Focus on: Parameters Update)

- [Federated Learning with Matched Averaging](https://openreview.net/pdf?id=BkluqlSFDS)
  -  Federated learning allows edge devices to collaboratively learn a shared model while keeping the training data on device, decoupling the ability to do model training from the need to store the data in the cloud. We propose Federated matched averaging (FedMA) algorithm designed for federated learning of modern neural network architectures e.g. convolutional neural networks (CNNs) and LSTMs. FedMA constructs the shared global model in a layer-wise manner by matching and averaging hidden elements (i.e. channels for convolution layers; hidden states for LSTM; neurons for fully connected layers) with similar feature extraction signatures. Our experiments indicate that FedMA not only outperforms popular state-of-the-art federated learning algorithms on deep CNN and LSTM architectures trained on real world datasets, but also reduces the overall communication burden.
  
- [Federated Learning: Strategies for Improving Communication Efficiency](https://arxiv.org/pdf/1610.05492)
  - [Code](https://researchcode.com/code/2158755848/federated-learning-strategies-for-improving-communication-efficiency/)
  - Abstract: Federated Learning is a machine learning setting where the goal is to train a high-quality centralized model while training data remains distributed over a large number of clients each with unreliable and relatively slow network connections. We consider learning algorithms for this setting where on each round, each client independently computes an update to the current model based on its local data, and communicates this update to a central server, where the client-side updates are aggregated to compute a new global model. The typical clients in this setting are mobile phones, and communication efficiency is of the utmost importance.
In this paper, we propose two ways to reduce the uplink communication costs: structured updates, where we directly learn an update from a restricted space parametrized using a smaller number of variables, e.g. either low-rank or a random mask; and sketched updates, where we learn a full model update and then compress it using a combination of quantization, random rotations, and subsampling before sending it to the server. Experiments on both convolutional and recurrent networks show that the proposed methods can reduce the communication cost by two orders of magnitude.

- [Optimal User-Edge Assignment in Hierarchical Federated Learning based on Statistical Properties and Network Topology Constraints](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9337204)
  - Abstract: Learning-based applications have demonstrated practical use cases in ubiquitous environments and amplified interest in exploiting the data stored on users' mobile devices. Distributed optimization algorithms aim to leverage such distributed and diverse data to learn a global phenomena by performing training amongst participating devices and repeatedly aggregating their local models' parameters into a global model. Federated Averaging is a promising solution that allows for extending local training before aggregating the parameters, offering better communication efficiency. However, in the cases where the participants' data are strongly skewed (i.e., local distributions are different), the model accuracy can significantly drop. To face this challenge, we leverage the edge computing paradigm to design a hierarchical learning system that performs Federated Gradient Descent on the user-edge layer and Federated Averaging on the edge-cloud layer. In this hierarchical architecture, the users might be assigned to different edges, leading to different edge-level data distributions. We formalize and optimize this user-edge assignment problem to minimize classes' distribution distance between edge nodes, which enhances the Federated Averaging performance. Our experiments on multiple real datasets show that the proposed optimized assignment is tractable and leads to faster convergence of models towards a better accuracy value.
- [Federated Gradient Averaging for Multi-Site Training with Momentum-Based Optimizers](https://link.springer.com/content/pdf/10.1007%2F978-3-030-60548-3_17.pdf) 
  - Abstract: Multi-site training methods for artificial neural networks are of particular interest to the medical machine learning community primarily due to the difficulty of data sharing between institutions. However, contemporary multi-site techniques such as weight averaging and cyclic weight transfer make theoretical sacrifices to simplify implementation. In this paper, we implement federated gradient averaging (FGA), a variant of federated learning without data transfer that is mathematically equivalent to single site training with centralized data. We evaluate two scenarios: a simulated multi-site dataset for handwritten digit classification with MNIST and a real multi-site dataset with head CT hemorrhage segmentation. We compare federated gradient averaging to single site training, federated weight averaging (FWA), and cyclic weight transfer. In the MNIST task, we show that training with FGA results in a weight set equivalent to centralized single site training. In the hemorrhage segmentation task, we show that FGA achieves on average superior results to both FWA and cyclic weight transfer due to its ability to leverage momentum-based optimization.

- [Adaptive Personalized Federated Learning](https://arxiv.org/pdf/2003.13461)
  - Abstract: Investigation of the degree of personalization in federated learning algorithms has shown that only maximizing the performance of the global model will confine the capacity of the local models to personalize. In this paper, we advocate an adaptive personalized federated learning (APFL) algorithm, where each client will train their local models while contributing to the global model. We derive the generalization bound of mixture of local and global models, and find the optimal mixing parameter. We also propose a communication-efficient optimization method to collaboratively learn the personalized models and analyze its convergence in both smooth strongly convex and nonconvex settings. The extensive experiments demonstrate the effectiveness of our personalization schema, as well as the correctness of established generalization theories.
 
- [Optimal Client Sampling for Federated Learning](https://arxiv.org/pdf/2010.13723.pdf) 
  - Abstract: It is well understood that client-master communication can be a primary bottleneck in Federated Learning. In this work, we address this issue with a novel client subsampling scheme, where we restrict the number of clients allowed to communicate their updates back to the master node. In each communication round, all participated clients compute their updates, but only the ones with "important" updates communicate back to the master. We show that importance can be measured using only the norm of the update and we give a formula for optimal client participation. This formula minimizes the distance between the full update, where all clients participate, and our limited update, where the number of participating clients is restricted. In addition, we provide a simple algorithm that approximates the optimal formula for client participation which only requires secure aggregation and thus does not compromise client privacy. We show both theoretically and empirically that our approach leads to superior performance for Distributed SGD (DSGD) and Federated Averaging (FedAvg) compared to the baseline where participating clients are sampled uniformly. Finally, our approach is orthogonal to and compatible with existing methods for reducing communication overhead, such as local methods and communication compression methods.
# 2020-03-24(Focus on: Federated Learning and Meta-Learning)
## ICLR2020

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
- [Adaptive Federated Optimization ](https://openreview.net/pdf?id=LkFG3lB13U5)
  - Federated learning is a distributed machine learning paradigm in which a large number of clients coordinate with a central server to learn a model without sharing their own training data. Standard federated optimization methods such as Federated Averaging (FedAvg) are often difficult to tune and exhibit unfavorable convergence behavior. In non-federated settings, adaptive optimization methods have had notable success in combating such issues. In this work, we propose federated versions of adaptive optimizers, including Adagrad, Adam, and  Yogi, and analyze their convergence in the presence of heterogeneous data for general non-convex settings. Our results highlight the interplay between client heterogeneity and communication efficiency. We also perform extensive experiments on these methods and show that the use of adaptive optimizers can significantly improve the performance of federated learning. 
- [Federated Composite Optimization](https://arxiv.org/pdf/2011.08474.pdf)
  -  Federated Learning (FL) is a distributed learning paradigm which scales on-device learning collaboratively and privately. Standard FL algorithms such as Federated Averaging (FedAvg) are primarily geared towards smooth unconstrained settings. In this paper, we study the Federated Composite Optimization (FCO) problem, where the objective function in FL includes an additive (possibly) non-smooth component. Such optimization problems are fundamental to machine learning and arise naturally in the context of regularization (e.g., sparsity, low-rank, monotonicity, and constraint). To tackle this problem, we propose different primal/dual averaging approaches and study their communication and computation complexities. Of particular interest is Federated Dual Averaging (FedDualAvg), a federated variant of the dual averaging algorithm. FedDualAvg uses a novel double averaging procedure, which involves gradient averaging step in standard dual averaging and an average of client updates akin to standard federated averaging. Our theoretical analysis and empirical experiments demonstrate that FedDualAvg outperforms baselines for FCO.
## ICLR 2021
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



