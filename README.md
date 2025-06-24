# Quality of Personalization, Explainability and Robustness of Recommendation Algorithms

<div align="center">
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>
<img src="https://img.shields.io/badge/Python-3.9+-blue.svg" />
<img src="https://img.shields.io/badge/License-MIT-green.svg" />
<img src="https://img.shields.io/badge/Status-Active-brightgreen.svg" />
<img src="https://img.shields.io/badge/MLflow-Tracking-blue.svg" />

**A Master's Thesis Project comparing recommendation algorithms across personalization, privacy, and explainability dimensions.**
</div>

---

## 👥 Authors & Supervision

- **Authors:** 
  - [Julia Podsadna](https://github.com/GambiBambi) 
  - [Bartosz Chwiłkowski](https://github.com/kooogi)
- **Supervisor:** [Mikołaj Morzy](https://github.com/megaduks)
- **University:** [Poznan University of Technology](https://put.poznan.pl/en)
- **Academic Year:** 2024/2025
- **Defense Date:** *To be announced*

---

## 📖 Abstract

This Master's thesis presents an in-depth analysis of three main families of recommendation algorithms (**collaborative filtering**, **content-based filtering**, and **reinforcement learning**) in terms of their:

- 🛡️ **Resilience** to data-related stresses
- 🔒 **Resistance** to anonymization techniques  
- 🔍 **Explainability** and ability to generate meaningful explanations
- ⚖️ **Ethical risks** associated with algorithm implementation

The research is particularly relevant given the recently adopted **EU AI Act** and the introduction of ethical risk taxonomy, as well as the **Omnibus Directive**. The project encompasses multiple recommendation scenarios across different datasets and application domains, implementing comprehensive metrics to assess algorithm behavior across the evaluated criteria.

**Note**: This project builds upon existing implementations from the [Microsoft Recommenders](https://github.com/recommenders-team/recommenders) library (CF & CBF algorithms) and [PGPR](https://github.com/orcax/PGPR) (RL algorithms), with significant modifications and a unified custom dataset loader for comparative analysis.

---

## ❓ Research Questions

This research addresses the following key questions:

1. **Robustness Analysis**: How do different recommendation algorithm families compare in terms of resilience to data anonymization and perturbation techniques?

2. **Privacy-Personalization Trade-off**: What is the relationship between recommendation accuracy, personalization quality, and user privacy preservation for each algorithm family?

3. **Explainability Assessment**: To what extent can each algorithm generate meaningful explanations for its recommendations, and how does this capability affect user trust and system transparency?

4. **Ethical Risk Evaluation**: How can ethical risks associated with each recommender system be identified, measured, and mitigated in accordance with EU AI Act requirements?

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/Master-s-thesis-PPERA/personalization-privacy-and-explainability-of-recommendation-algorithms.git

cd personalization-privacy-and-explainability-of-recommendation-algorithms

make quickstart
```

## 📚 Documentation

- **[Getting Started](docs/getting-started.md)** - Installation, setup, first run
- **[Datasets](docs/datasets.md)** - Download instructions and data details  
- **[Experiments](docs/experiments.md)** - Running and configuring experiments
- **[Architecture](docs/architecture.md)** - Project structure and code organization
- **[Results](docs/results.md)** - Findings and publications
- **[API Reference](docs/api/)** - Code documentation

## 👥 Authors & Citation

**Authors:** [Julia Podsadna](https://github.com/GambiBambi), [Bartosz Chwiłkowski](https://github.com/kooogi)  
**Supervisor:** [Mikołaj Morzy](https://github.com/megaduks)  
**University:** Poznan University of Technology (2024/2025)

[How to cite this work](docs/citation.md)

## 🙏 Acknowledgments

We extend our sincere gratitude to:

- **Prof. Mikołaj Morzy** for his invaluable guidance, expertise, and continuous support throughout this research
- **Poznan University of Technology** for providing the academic environment and resources necessary for this work
- **Microsoft Recommenders Team** for their open-source library that served as a foundation for collaborative filtering implementations
- **PGPR Authors** for making their reinforcement learning framework available for research purposes
- **The Open Source Community** for tools like MLflow, PyTorch, and scikit-learn that made this research possible
- **Kaggle Community** for providing accessible datasets crucial for comprehensive algorithm evaluation
- **EU AI Act Working Groups** for establishing ethical frameworks that guided our risk assessment methodology

Special thanks to our fellow students and faculty members who provided feedback during development and testing phases.

## 🤝 Contributing

Questions? Email us or open an issue. See [Contributing Guidelines](docs/contributing.md).

---
<sub>Built with [Microsoft Recommenders](https://github.com/recommenders-team/recommenders), [PGPR](https://github.com/orcax/PGPR) and [recmetrics](https://github.com/statisticianinstilettos/recmetrics)</sub>