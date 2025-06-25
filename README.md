# Quality of Personalization, Explainability and Robustness of Recommendation Algorithms

<div align="center">

[![CCDS Project Template](https://img.shields.io/badge/CCDS-Project--template-328F97)](https://cookiecutter-data-science.drivendata.org/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/PUT-RecSys-Research/qpera-thesis/pulse)
[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/documentation-view-blue.svg)](https://put-recsys-research.github.io/qpera-thesis/)
[![MLflow Tracking](https://img.shields.io/badge/MLflow-Tracking-blue.svg)](https://mlflow.org/)

**A Master's Thesis Project for evaluating recommendation algorithms on Quality, Personalization, Explainability, and Robustness.**
</div>

## 👥 Authors & Supervision

- **Authors:** 
  - [mgr inż. Julia Podsadna](https://github.com/GambiBambi) 
  - [mgr inż. Bartosz Chwiłkowski](https://github.com/kooogi)
- **Supervisor:** [dr hab. inż. Mikołaj Morzy prof. PP](https://github.com/megaduks)
- **University:** [Poznan University of Technology](https://put.poznan.pl/en)
- **Academic Year:** 2024/2025

---

## 📖 Abstract

This Master's thesis presents an in-depth analysis of three main families of recommendation algorithms (**collaborative filtering**, **content-based filtering**, and **reinforcement learning**). The evaluation focuses on their resilience to data-related stresses, resistance to anonymization, explainability, and the ethical risks associated with their implementation. This research is particularly relevant given the recently adopted **EU AI Act** and the **Omnibus Directive**. The project provides a framework for comprehensive, multi-dimensional evaluation of recommender systems across diverse datasets and domains.

---

## ❓ Research Questions

This research addresses the following key questions:

1.  **Robustness Analysis**: How do different recommendation algorithm families compare in terms of resilience to data anonymization and perturbation techniques?
2.  **Privacy-Personalization Trade-off**: What is the relationship between recommendation accuracy, personalization quality, and user privacy preservation for each algorithm family?
3.  **Explainability Assessment**: To what extent can each algorithm generate meaningful explanations for its recommendations, and how does this capability affect user trust and system transparency?
4.  **Ethical Risk Evaluation**: How can ethical risks associated with each recommender system be identified, measured, and mitigated in accordance with EU AI Act requirements?

---

## 🔬 Key Contributions

This framework provides a novel, unified platform for evaluating recommendation algorithms across multiple, often conflicting, dimensions. Our key contributions include:

-   **Comprehensive Evaluation Framework**: A standardized pipeline to evaluate algorithms across personalization, privacy, explainability, and robustness.
-   **Trade-off Quantification**: Systematic analysis of the performance degradation under privacy constraints and personalization adjustments.
-   **Cross-Domain Analysis**: Evidence-based guidelines for algorithm selection based on performance across different application domains (e-commerce, social media, entertainment).
-   **Reproducible Research**: An open-source implementation with MLflow tracking and containerization to ensure full reproducibility of our findings.

---

## 🚀 Quick Start

**1. Clone the repository:**
```bash
git clone https://github.com/PUT-RecSys-Research/qpera-thesis.git
cd qpera-thesis
```

**2. Configure Kaggle API (required for dataset downloads)**

Download your Kaggle API key (`kaggle.json`) and place it in `~/.kaggle/`. For detailed instructions, run `make kaggle-setup-help`.

**3. Run the main pipeline**
```bash
make quickstart
```
For detailed setup instructions, including API key configuration, please see the [**Getting Started Guide**](docs/getting-started.md).

---

## 📚 Documentation

*   [**Getting Started**](docs/getting-started.md): Installation and first run.
*   [**Running Experiments**](docs/experiments.md): Guide to configuring and running experiments.
*   [**Architecture**](docs/architecture.md): Overview of the project structure and code organization.
*   [**Results & Analysis**](docs/results.md): Summary of key findings and performance metrics.
*   [**API Reference**](docs/api.md): Detailed documentation for the source code.
*   [**Contributing**](docs/contributing.md): How to contribute to the project.

---

## 📄 How to Cite

If you use this framework or our findings in your research, please cite our work.

```bibtex
@software{podsadna_chwilkowski_qpera2025,
  title={QPERA: A Framework for Evaluating Quality of Personalization, Explainability, and Robustness of Recommendation Algorithms},
  author={Podsadna, Julia and Chwi{\l}kowski, Bartosz},
  year={2025},
  url={https://github.com/PUT-RecSys-Research/qpera-thesis},
  version={1.0.0},
  license={MIT},
  note={Open-source framework for comprehensive recommendation algorithm evaluation}
}
```
For more specific citations, please see the [**Citation Guide**](docs/citation.md).

---

## 🙏 Acknowledgments

This research was made possible by the guidance of our supervisor, the resources provided by Poznan University of Technology, and the foundational work of the open-source community. We especially thank the teams behind **Microsoft Recommenders** and **PGPR** for their pioneering implementations. For a complete list of acknowledgments, please see our [**Acknowledgments Page**](docs/acknowledgments.md).

---

## 🤝 Contributing & Support

We welcome contributions and feedback! Please see our [**Contributing Guidelines**](docs/contributing.md) for details on our development workflow and how to submit pull requests.

For bugs, feature requests, or questions, please use [**GitHub Issues**](https://github.com/PUT-RecSys-Research/qpera-thesis/issues) or [**Discussions**](https://github.com/PUT-RecSys-Research/qpera-thesis/discussions).
