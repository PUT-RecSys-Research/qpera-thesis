# PPERA: Personalization, Privacy, and Explainability of Recommendation Algorithms

<div align="center">
<img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
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

This Master's thesis presents an in-depth analysis of **three main families** of recommendation algorithms (**collaborative filtering**, **content-based filtering**, and **reinforcement learning**) in terms of their:

- 🛡️ **Resilience** to data-related stresses
- 🔒 **Resistance** to anonymization techniques  
- 🔍 **Explainability** and ability to generate meaningful explanations
- ⚖️ **Ethical risks** associated with algorithm implementation

The research is particularly relevant given the recently adopted **EU AI Act** and the introduction of ethical risk taxonomy, as well as the **Omnibus Directive**. The project encompasses multiple recommendation scenarios across different datasets and application domains, implementing comprehensive metrics to assess algorithm behavior across the evaluated criteria.

!!! note "Built on Open Source"
    This project builds upon existing implementations from the [Microsoft Recommenders](https://github.com/recommenders-team/recommenders) library (CF & CBF algorithms) and [PGPR](https://github.com/orcax/PGPR) (RL algorithms), with significant modifications and a unified custom dataset loader for comparative analysis.

---

## ❓ Research Questions

This research addresses the following key questions:

1. **Robustness Analysis**: How do different recommendation algorithm families compare in terms of resilience to data anonymization and perturbation techniques?

2. **Privacy-Personalization Trade-off**: What is the relationship between recommendation accuracy, personalization quality, and user privacy preservation for each algorithm family?

3. **Explainability Assessment**: To what extent can each algorithm generate meaningful explanations for its recommendations, and how does this capability affect user trust and system transparency?

4. **Ethical Risk Evaluation**: How can ethical risks associated with each recommender system be identified, measured, and mitigated in accordance with EU AI Act requirements?

---

## 🚀 Quick Start

!!! tip "New to PPERA?"
    Start here for a complete setup and your first experiment!

```bash
# 1. Clone and setup
git clone https://github.com/Master-s-thesis-PPERA/personalization-privacy-and-explainability-of-recommendation-algorithms.git
cd personalization-privacy-and-explainability-of-recommendation-algorithms

# 2. Quick installation
make quickstart

# 3. Run your first experiment
make run-cf-movielens
```

---

## 📚 Documentation Navigation

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting Started**

    ---

    Complete installation guide and first experiment setup

    [:octicons-arrow-right-24: Getting Started](getting-started.md)

-   :material-database:{ .lg .middle } **Datasets**

    ---

    Download instructions and dataset details for MovieLens, Amazon, and more

    [:octicons-arrow-right-24: Datasets](datasets.md)

-   :material-flask:{ .lg .middle } **Run Experiments**

    ---

    Configure and execute algorithm comparisons with privacy and personalization settings

    [:octicons-arrow-right-24: Experiments](experiments.md)

-   :material-chart-line:{ .lg .middle } **Results & Analysis**

    ---

    Research findings, publications, and how to cite this work

    [:octicons-arrow-right-24: Results](results.md)

-   :material-cog:{ .lg .middle } **Architecture**

    ---

    Code structure, design patterns, and extension points

    [:octicons-arrow-right-24: Architecture](architecture.md)

-   :material-code-braces:{ .lg .middle } **API Reference**

    ---

    Detailed code documentation and examples

    [:octicons-arrow-right-24: API Reference](api/)

</div>

---

## 🏗️ Algorithm Overview

The PPERA framework evaluates three main algorithm families:

| Algorithm Family | Key Strengths | Best Use Cases | Privacy Score | Explainability |
|------------------|---------------|----------------|---------------|----------------|
| **Collaborative Filtering** | High accuracy, established methods | Large user bases, rich interaction data | ⭐⭐ | ⭐⭐ |
| **Content-Based Filtering** | Privacy-friendly, transparent | Cold start, feature-rich items | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Reinforcement Learning** | Adaptive, long-term optimization | Dynamic preferences, sequential behavior | ⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 📊 Key Research Contributions

1. **🎯 Unified Evaluation Framework** - First comprehensive comparison across personalization, privacy, explainability, and robustness dimensions

2. **⚖️ Trade-off Analysis** - Quantified inherent tensions between privacy requirements and explainability needs

3. **🎨 Domain-Specific Guidelines** - Algorithm selection recommendations based on application domain characteristics

4. **🔬 Reproducible Research Pipeline** - Complete open-source implementation with detailed documentation

---

## 🔧 Technical Features

- **MLflow Integration** - Automatic experiment tracking and result visualization
- **Modular Design** - Easy to extend with new algorithms and metrics  
- **Privacy-Aware** - Built-in differential privacy and anonymization testing
- **Explainable by Design** - Explanation generation for all algorithm types
- **Production Ready** - Scalable architecture with performance optimizations

---

## 🤝 Getting Help & Contributing

### 📞 Support Channels

- **📖 Documentation Issues**: Check our troubleshooting guide first
- **🐛 Bug Reports**: [GitHub Issues](https://github.com/Master-s-thesis-PPERA/personalization-privacy-and-explainability-of-recommendation-algorithms/issues)
- **💬 Questions**: [GitHub Discussions](https://github.com/Master-s-thesis-PPERA/personalization-privacy-and-explainability-of-recommendation-algorithms/discussions)
- **📧 Direct Contact**: Email the authors for research collaboration

### 🛠️ Development

- **[Contributing Guidelines](contributing.md)** - How to contribute code and documentation
- **[Architecture Guide](architecture.md)** - Understanding the codebase structure
- **[API Documentation](api/)** - Detailed code reference

---

## 🙏 Acknowledgments

We gratefully acknowledge the contributions of the open-source community, our academic supervisors, and the institutions that made this research possible.

**[View Full Acknowledgments](acknowledgments.md)**

---

## 📄 License & Citation

This project is released under the **MIT License**. 

**[How to Cite This Work](citation.md)**

---

<div align="center">
<sub>Built with ❤️ using <a href="https://github.com/recommenders-team/recommenders">Microsoft Recommenders</a>, <a href="https://github.com/orcax/PGPR">PGPR</a>, and <a href="https://github.com/statisticianinstilettos/recmetrics">recmetrics</a></sub>
</div>

