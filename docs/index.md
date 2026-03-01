# Quality of Personalization, Explainability and Robustness of Recommendation Algorithms

<p style="text-align: center;">
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/"><img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" /></a>
<a href="https://github.com/PUT-RecSys-Research/qpera-thesis/pulse"><img src="https://img.shields.io/badge/Status-Active-brightgreen.svg" /></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.9+-blue.svg" /></a>
<a href="https://github.com/PUT-RecSys-Research/qpera-thesis/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" /></a>
<a href="https://mlflow.org/"><img src="https://img.shields.io/badge/MLflow-Tracking-blue.svg" /></a>
</p>

**A Master's Thesis Project for evaluating recommendation algorithms on Quality, Personalization, Explainability, and Robustness.**

---

## 📖 Overview

This project investigates how three families of recommendation algorithms — **Content-Based Filtering (CBF)**, **Collaborative Filtering (CF)**, and **Reinforcement Learning (RL)** — perform in terms of three key aspects: **robustness** to the privatization of user history, the ability to **personalize** recommendations, and the **explainability** of the generated results. The algorithms were tested using three datasets representing varying levels of decision-making consequences: AmazonSales (high-cost), MovieLens (moderate-cost), and PostRecommendations (low-cost). This research is particularly relevant given the recently adopted **EU AI Act** and the **Omnibus Directive**.

!!! note "Built on Open Source"
    This project extends and unifies implementations from the [Microsoft Recommenders](https://github.com/recommenders-team/recommenders) library and [PGPR](https://github.com/orcax/PGPR), providing a custom dataset loader and evaluation pipeline for comparative analysis.

---

## 📚 Documentation

Use the cards below to navigate to the section you need.

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting Started**

    ---

    Complete installation guide and first experiment setup.

    [:octicons-arrow-right-24: Get Started](getting-started.md)

-   :material-flask:{ .lg .middle } **Run Experiments**

    ---

    Configure and execute algorithm comparisons.

    [:octicons-arrow-right-24: Run Experiments](experiments.md)

-   :material-chart-line:{ .lg .middle } **Results & Analysis**

    ---

    Research findings, publications, and how to cite this work.

    [:octicons-arrow-right-24: See Results](results.md)

-   :material-cog:{ .lg .middle } **Architecture**

    ---

    Code structure, design patterns, and extension points.

    [:octicons-arrow-right-24: Explore Architecture](architecture.md)

</div>

---

## ❓ Research Questions

The underlying rapid literature review addressed four research questions:

1.  **RQ1**: What kind of algorithms are used in recommendation systems?
2.  **RQ2**: What kind of metrics can be used to evaluate algorithms' quality of personalization?
3.  **RQ3**: What kind of metrics can be used to evaluate algorithms' explainability?
4.  **RQ4**: What kind of metrics can be used to evaluate algorithms' robustness?

The empirical study then evaluated three concrete aspects:

-   **Personalization**: The extent to which the algorithm can provide personalized recommendations despite the aggregation of user history with that of others.
-   **Robustness**: The extent to which the algorithm can deliver accurate recommendations even when the user's history is partially concealed or removed.
-   **Explainability**: The extent to which the recommendation explanation is satisfactory for the user, assessed using the SAGES framework (Simple, Adaptable, Grounded, Extendable, Source-based).

---

## 🤝 Contributing & Support

-   **🐛 Bug Reports**: [GitHub Issues](https://github.com/PUT-RecSys-Research/qpera-thesis/issues)
-   **💬 Questions & Ideas**: [GitHub Discussions](https://github.com/PUT-RecSys-Research/qpera-thesis/discussions)
-   **🛠️ Code Contributions**: See our [Contributing Guidelines](contributing.md)
