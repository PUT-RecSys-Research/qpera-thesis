# Quality of Personalization, Explainability and Robustness of Recommendation Algorithms

<div align="center">

[![CCDS Project Template](https://img.shields.io/badge/CCDS-Project--template-328F97)](https://cookiecutter-data-science.drivendata.org/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/PUT-RecSys-Research/qpera-thesis/pulse)
[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/documentation-view-blue.svg)](https://put-recsys-research.github.io/qpera-thesis/)
[![MLflow Tracking](https://img.shields.io/badge/MLflow-Tracking-blue.svg)](https://mlflow.org/)

**A Master's Thesis Project for evaluating recommendation algorithms on Quality of Personalization, Explainability, and Robustness.**
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

The aim of this study was to investigate how recommendation algorithms — **Content-Based Filtering**, **Collaborative Filtering**, and **Reinforcement Learning** — perform in terms of three key aspects: robustness to the privatization of user history, the ability to personalize recommendations, and the explainability of the generated results. The algorithms were tested using three datasets: **AmazonSales** (high-cost decisions), **MovieLens** (moderate-cost decisions), and **PostRecommendations** (low-cost decisions). Their performance was assessed across defined scenarios using recommendation quality metrics and feedback from survey participants. The results indicated that none of the evaluated algorithms consistently outperformed the others in terms of robustness and personalization. However, with regard to explainability, respondents generally favored Content-Based Filtering. This research is particularly relevant given the recently adopted **EU AI Act** and the **Omnibus Directive**.

---

## ❓ Research Questions

The underlying rapid literature review addressed four research questions:

1.  **RQ1**: What kind of algorithms are used in recommendation systems?
2.  **RQ2**: What kind of metrics can be used to evaluate algorithms' quality of personalization?
3.  **RQ3**: What kind of metrics can be used to evaluate algorithms' explainability?
4.  **RQ4**: What kind of metrics can be used to evaluate algorithms' robustness?

The empirical study evaluated three concrete aspects:

-   **Personalization**: The extent to which the algorithm can provide personalized recommendations despite the aggregation of user history with that of others.
-   **Robustness**: The extent to which the algorithm can deliver accurate recommendations even when the user's history is partially concealed or removed.
-   **Explainability**: The extent to which the recommendation explanation is satisfactory for the user, assessed using the SAGES framework (Simple, Adaptable, Grounded, Extendable, Source-based).

---

## 🔬 Key Contributions

-   **Comprehensive Evaluation Framework**: A standardized pipeline to evaluate disparate recommendation algorithms across personalization, privacy robustness, and explainability using consistent metrics.
-   **Empirical Trade-off Analysis**: Quantitative evidence of the trade-offs between personalization, privacy, and explainability across three algorithm families and three datasets representing different decision-making costs.
-   **Explainability Survey**: A user study with 36 participants evaluating recommendation explanations using the SAGES framework, providing empirical evidence that item-to-item (CBF) explanations are preferred.
-   **Reproducible Research**: An open-source implementation with MLflow tracking to ensure full reproducibility of all findings.

---

## 🚀 Quick Start

**1. Clone the repository:**
```bash
git clone https://github.com/PUT-RecSys-Research/qpera-thesis.git
cd qpera-thesis
```

**2. Configure Kaggle API:**
  This project requires the Kaggle API for downloading datasets.
  - Download your `kaggle.json` API token from your Kaggle account page.
  - For automated setup (places `kaggle.json` from `~/Downloads` to `~/.kaggle/`), run:
  ```bash
  make kaggle-autoconfig
  ```

**3. Run the main pipeline**
```bash
make quickstart
```
For detailed setup instructions, including API key configuration, please see the [**Getting Started Guide**](docs/getting-started.md).

---

## 📖 Full Documentation

**For a comprehensive guide to the project, including setup, architecture, and results, please visit our full documentation site:**

### **[https://put-recsys-research.github.io/qpera-thesis/](https://put-recsys-research.github.io/qpera-thesis/)**

or run it locally:
```bash
make docs
```

---

## 📄 How to Cite

If you use this framework or our findings in your research, please cite our work.

```bibtex
@software{podsadna_chwilkowski_qpera2025,
  title={Quality of Personalization, Explainability and Robustness of Recommendation Algorithms},
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
