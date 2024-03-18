# 🧐 Motivation

Most machine learning systems rely on *stationary, labeled, balanced and large-scale* datasets.
**Incremental learning** (IL), also referred to as **lifelong learning** (LL) or **continual learning** (CL), extends the traditional paradigm to work in dynamic and evolving environments.
This requires such systems to acquire and preserve knowledge continually.

Existing CL frameworks like [avalanche](https://github.com/ContinualAI/avalanche)[^1] or [continuum](https://github.com/Continvvm/continuum)[^2] construct data streams by *splitting* large datasets into multiple *experiences*, which has a few disadvantages:

- results in unrealistic scenarios
- offers limited insight into distributions and their evolution
- not extendable to scenarios with less constraints on the stream properties

To answer different research questions in the field of CL, researchers need knowledge and control over:

- class distributions
- novelties and outliers
- complexity and evolution of the background domain
- semantics of the unlabeled parts of a domain
- class dependencies
- class composition (for multi-label modelling)

A more economic alternative to collecting and labelling streams with desired properties is the **generation** of synthetic streams.
Some mentionable efforts in that direction include augmentation based dataset generation like [ImageNet-C](https://github.com/hendrycks/robustness)[^3] or simulation based approaches like the [EndlessCLSim](https://arxiv.org/abs/2106.02585)[^4], where semantically labeled street-view images are generated (and labeled) by a game engine, that procedurally generates the city environment and simulates drift by modifying parameters (like the weather and illumination conditions) over time.

This project builds on these ideas and presents a general framework for generating a stream of labeled samples.

## 🏛️ History

- spring 2023: initial idea occurred during the planning phase of my PhD on Continual Learning
- summer 2023: conceptualization of tree sampler and parameter schedules
- winter 2024: implementation of a first proof-of-concept and several feedback rounds and iterations
- spring 2024: white paper was written

---

## 📄 References

[^1]: V. Lomonaco et al., “Avalanche: an End-to-End Library for Continual Learning,” in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), Nashville, TN, USA: IEEE, Jun. 2021, pp. 3595–3605. doi: 10.1109/CVPRW53098.2021.00399.
[^2]: A. Douillard and T. Lesort, “Continuum: Simple Management of Complex Continual Learning Scenarios.” arXiv, Feb. 11, 2021. doi: 10.48550/arXiv.2102.06253.
[^3]: D. Hendrycks and T. Dietterich, “Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.” arXiv, Mar. 28, 2019. doi: 10.48550/arXiv.1903.12261.
[^4]: T. Hess, M. Mundt, I. Pliushch, and V. Ramesh, “A Procedural World Generation Framework for Systematic Evaluation of Continual Learning.” arXiv, Dec. 13, 2021. doi: 10.48550/arXiv.2106.02585.
