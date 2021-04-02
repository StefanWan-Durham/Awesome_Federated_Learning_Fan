
# 2021-04-01 (Keyword: Parameters Update, Video)

- [Efficient and Privacy Preserving Video Transmission in 5G-Enabled IoT Surveillance Networks: Current Challenges and Future Directions](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9299471)
  - keyword: Video Summarizaiton 
  - Vision sensors in Internet of Things (IoT)-connected smart cities play a vital role in the exponential growth of video data, thereby making its analysis and storage comparatively tough and challenging. Those sensors continuously generate data for 24 hours, which requires huge storage resources, dedicated networks for sharing with data centers, and most importantly, it makes browsing, retrieval, and event searching a difficult and time-consuming job. Video summarization (VS) is a promising direction toward a solution to these problems by analyzing the visual contents acquired from a vision sensor and prioritizing them based on events, saliency, person's appearance, and so on. However, the current VS literature still lacks focus on resource-constrained devices that can summarize data over the edge and upload it to data repositories efficiently for instant analysis. Therefore, in this article, we carry out a survey of functional VS methods to understand their pros and cons for resource-constrained devices, with the ambition to provide a compact tutorial to the community of researchers in the field. Further, we present a novel saliency-aware VS framework, incorporating 5G-enabled IoT devices, which keeps only important data, thereby saving storage resources and providing representative data for immediate exploration. Keeping privacy of data as a second priority, we intelligently encrypt the salient frames over resource-constrained devices before transmission over the 5G network. The reported experimental results show that our proposed framework has additional benefits of faster transmission (1.8~13.77 percent frames of a lengthy video are considered for transmission), reduced bandwidth, and real-time processing compared to state-of-the-art methods in the field. 

- [Federated learning with non-iid data](https://arxiv.org/pdf/1806.00582.pdf)
  - keyword: Non-iid, Address the issues of accuracy reduction when train Non-iid data in federated setting. 
  - dataset: CIFAR-10 dataset
 
 - [FedCD: Improving Performance in non-IID Federated Learning](https://arxiv.org/pdf/2006.09637)
   - keyword: We present a novel approach, FedCD, which clones and deletes models to dynamically group devices with similar data. 
 
- [Overcoming Forgetting in Federated Learning on Non-IID Data](https://arxiv.org/pdf/1910.07796)
  - keyword: We add a penalty term to the loss function, compelling all local models to converge to a shared optimum.  

- [A Distributed Video Analytics Architecture Based on Edge-Computing and Federated Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8890415)
  - The current model of sending video streams to the cloud for processing is facing many challenges such as latency and privacy.
  - We introduce a distributed video analytics architecture based on edge-computing and the newly emerging federated learning.
  

- [Federated Learning with Matched Averaging](https://openreview.net/pdf?id=BkluqlSFDS)
  - keyword: LSTM 
  - We propose Federated matched averaging (FedMA) algorithm designed for federated learning of modern neural network architectures e.g. convolutional neural networks (CNNs) and <u>LSTMs</u>.


- [Adaptive Federated Optimization ](https://openreview.net/pdf?id=LkFG3lB13U5)
  - keyword: Non-iid,  adaptive optimization methods
  - In this work, we propose federated versions of adaptive optimizers, including Adagrad, Adam, and  Yogi, and analyze their convergence in the presence of heterogeneous data for general non-convex settings.
 
- [Optimal User-Edge Assignment in Hierarchical Federated Learning based on Statistical Properties and Network Topology Constraints](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9337204)
  - Keyword: Address the model can siginificantly drop due to the Non-iid data.
  - Method: Client performs Federated Gradient Descent; Server performs Federated Averaging 
  

- [Adaptive Personalized Federated Learning](https://arxiv.org/pdf/2003.13461)
  - Abstract: Investigation of the degree of personalization in federated learning algorithms has shown that only maximizing the performance of the global model will confine the capacity of the local models to personalize. In this paper, we advocate an adaptive personalized federated learning (APFL) algorithm, where each client will train their local models while contributing to the global model. We derive the generalization bound of mixture of local and global models, and find the optimal mixing parameter. We also propose a communication-efficient optimization method to collaboratively learn the personalized models and analyze its convergence in both smooth strongly convex and nonconvex settings. The extensive experiments demonstrate the effectiveness of our personalization schema, as well as the correctness of established generalization theories.
 
- [Optimal Client Sampling for Federated Learning](https://arxiv.org/pdf/2010.13723.pdf) 
  - keyword: In each communication round, all participated clients compute their updates, but only the ones with “important” updates communicate back to the master. 
  
- [Improving Federated Learning Personalization via Model Agnostic Meta Learning](https://arxiv.org/pdf/1909.12488)
  - keyword: meta-learning, Non-iid


# 2020-03-24(Keyword: Federated Learning, Meta-Learning)
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



