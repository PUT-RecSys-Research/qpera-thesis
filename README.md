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

## 👥 Authors & Supervision

- **Authors:** 
  - [mgr inż. Julia Podsadna](https://github.com/GambiBambi) 
  - [mgr inż. Bartosz Chwiłkowski](https://github.com/kooogi)
- **Supervisor:** [dr hab. inż. Mikołaj Morzy prof. PP](https://github.com/megaduks)
- **University:** [Poznan University of Technology](https://put.poznan.pl/en)
- **Academic Year:** 2024/2025
- **Defense Date:** *To be announced*

## 📖 Abstract

This Master's thesis presents an in-depth analysis of three main families of recommendation algorithms (**collaborative filtering**, **content-based filtering**, and **reinforcement learning**) in terms of their:

- 🛡️ **Resilience** to data-related stresses
- 🔒 **Resistance** to anonymization techniques  
- 🔍 **Explainability** and ability to generate meaningful explanations
- ⚖️ **Ethical risks** associated with algorithm implementation

The research is particularly relevant given the recently adopted **EU AI Act** and the introduction of ethical risk taxonomy, as well as the **Omnibus Directive**. The project encompasses multiple recommendation scenarios across different datasets and application domains, implementing comprehensive metrics to assess algorithm behavior across the evaluated criteria.

### Technical Implementation

The framework integrates and extends existing implementations:
- **Collaborative & Content-Based**: [Microsoft Recommenders](https://github.com/recommenders-team/recommenders)
- **Reinforcement Learning**: [PGPR](https://github.com/orcax/PGPR)
- **Evaluation Metrics**: [recmetrics](https://github.com/statisticianinstilettos/recmetrics)

All implementations include custom modifications for unified evaluation and comparative analysis.

## ❓ Research Questions

This research addresses the following key questions:

1. **Robustness Analysis**: How do different recommendation algorithm families compare in terms of resilience to data anonymization and perturbation techniques?

2. **Privacy-Personalization Trade-off**: What is the relationship between recommendation accuracy, personalization quality, and user privacy preservation for each algorithm family?

3. **Explainability Assessment**: To what extent can each algorithm generate meaningful explanations for its recommendations, and how does this capability affect user trust and system transparency?

4. **Ethical Risk Evaluation**: How can ethical risks associated with each recommender system be identified, measured, and mitigated in accordance with EU AI Act requirements?

## ✨ Key Features

- 🔄 **Multi-Algorithm Implementation**: 
  - Collaborative Filtering & Content-Based Filtering (adapted from [Microsoft Recommenders](https://github.com/recommenders-team/recommenders))
  - Reinforcement Learning (extensively modified from [PGPR](https://github.com/orcax/PGPR))
- 🔗 **Unified Data Pipeline**: Custom dataset loader ensuring consistent data processing across all algorithms
- 🛡️ **Privacy-Preserving Techniques**: Data anonymization and perturbation methods
- 📊 **Comprehensive Evaluation**: Multiple datasets (MovieLens, Amazon Sales, PostRecommendations)
- 🔍 **Explainability Metrics**: Novel metrics for recommendation transparency
- ⚖️ **Ethical Risk Assessment**: AI Act compliance evaluation framework
- 📈 **MLflow Integration**: Experiment tracking and reproducibility
- 🐍 **Modern Python Stack**: PyTorch, scikit-learn, pandas, numpy

## 📋 Prerequisites

- **Python 3.9** (recommended based on Microsoft recommenders)
- **Conda** or **Miniconda** 
- **Git**
- **Kaggle Account** with API access
- **16GB RAM** (recommended for large datasets)
- **Storage**: 100GB free space

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Master-s-thesis-PPERA/Quality-of-Personalization-Explainability-and-Robustness-of-Recommendation-Algorithms.git
cd Quality-of-Personalization-Explainability-and-Robustness-of-Recommendation-Algorithms

# 2. Configure Kaggle API (required for dataset downloads)
# Download your Kaggle API key (kaggle.json) and place it in ~/.kaggle/
# For detailed instructions: make kaggle-setup-help

# 3. Run the main pipeline
make quickstart
```

## ⚡ Performance

### Expected Runtime
| Dataset Size          | Algorithm Type           | Approximate Time                          |
|-----------------------|-------------------------|------------------------------------------|
| AmazonSales (1.4k samples) | Collaborative Filtering | xx.xx seconds for run with no modifications |
| AmazonSales (1.4k samples) | Content-Based Filtering  | xx.xx seconds for run with no modifications |
| AmazonSales (1.4k samples) | Reinforcement Learning   | xx.xx seconds for run with no modifications |
| MovieLens (14k samples)    | Collaborative Filtering | xx.xx seconds for run with no modifications |
| MovieLens (14k samples)    | Content-Based Filtering  | xx.xx seconds for run with no modifications |
| MovieLens (14k samples)    | Reinforcement Learning   | xx.xx seconds for run with no modifications |
| PostRecommendation (14k samples) | Collaborative Filtering | xx.xx seconds for run with no modifications |
| PostRecommendation (14k samples) | Content-Based Filtering  | xx.xx seconds for run with no modifications |
| PostRecommendation (14k samples) | Reinforcement Learning   | xx.xx seconds for run with no modifications |

All experiments can take up to 4 hours on the following configuration:
- **CPU**: 12 cores, 4.3GHz
- **RAM**: 16GB
- **Storage**: 10GB free space
- **OS**: Linux (e.g., Debian 12)

## 📚 Documentation

*   [**Getting Started**](docs/getting-started.md) - Installation, setup, and how to perform a first run.
*   [**Datasets**](docs/datasets.md) - Download instructions and details on the data's structure.
*   [**Experiments**](docs/experiments.md) - A guide to running experiments and configuring parameters.
*   [**Architecture**](docs/architecture.md) - An overview of the project structure and code organization.
*   [**API Reference**](docs/api.md) - Detailed documentation for the source code's modules and functions.
*   [**Results**](docs/results.md) - A summary of key findings and links to publications.
*   [**Related Work**](docs/related-work.md) - The context of this project within the research field.
*   [**Citation**](docs/citation.md) - Instructions on how to properly cite this repository.
*   [**Contributing**](docs/contributing.md) - Guidelines for reporting bugs or contributing to the project.
*   [**Acknowledgments**](docs/acknowledgments.md) - Credits for supervisors, collaborators, and funding.

## 📁 Project Organization

```
.
├── .gitignore          # Git ignore patterns for ML/data files
├── LICENSE             # MIT License file
├── Makefile            # Convenience commands for setup, testing, and execution
├── README.md           # You are here! Main project documentation
├── environment.yml     # Conda environment specification for reproducibility
├── mkdocs.yml          # Configuration for the MkDocs documentation site
├── pyproject.toml      # Python project configuration and dependencies (PEP 621)
│
├── datasets/           # ⚠️ Raw and processed datasets (ignored by Git)
│   ├── AmazonSales/
│   ├── MovieLens/
│   └── PostRecommendation/
│
├── docs/               # 📚 Project documentation source files (for MkDocs)
│   ├── api.md
│   ├── architecture.md
│   ├── citation.md
│   ├── contributing.md
│   ├── datasets.md
│   ├── experiments.md
│   └── results.md
│
├── ppera/              # 🐍 Main source code package
│   ├── __init__.py                     # Makes `ppera` a Python package
│   ├── main.py                         # Main CLI entry point and experiment orchestrator
│   │
│   ├── # --- Data Handling ---
│   ├── datasets_downloader.py          # Utilities for downloading datasets
│   ├── datasets_loader.py              # Unified data loading and preprocessing
│   ├── data_manipulation.py            # Data transformation, augmentation, and privacy
│   │
│   ├── # --- Core Algorithms ---
│   ├── CBF.py                          # Content-Based Filtering implementation
│   ├── CF.py                           # Collaborative Filtering implementation
│   ├── RL.py                           # Reinforcement Learning orchestrator
│   │
│   ├── # --- Reinforcement Learning Components ---
│   ├── rl_preprocess.py                # Data preprocessing for the RL environment
│   ├── rl_knowledge_graph.py           # Knowledge graph construction
│   ├── rl_kg_env.py                    # RL environment combining state, action, and KG
│   ├── rl_transe_model.py              # TransE model implementation for KG embeddings
│   ├── rl_train_transe_model.py        # Script to train the TransE model
│   ├── rl_train_agent.py               # RL agent training logic
│   ├── rl_test_agent.py                # RL agent testing and evaluation logic
│   ├── rl_prediction.py                # Generates recommendations using the trained RL agent
│   ├── rl_decoder.py                   # Decodes agent output into recommendations
│   ├── rl_utils.py                     # Utility functions for the RL components
│   │
│   ├── # --- Utilities & Tooling ---
│   ├── metrics.py                      # Evaluation metrics (e.g., NDCG, HR)
│   ├── log_mlflow.py                   # MLflow integration for experiment tracking
│   ├── frequency_based_rating_gen.py   # Synthetic rating generation
│   └── rating_timestamp_gen.py         # Utilities for generating synthetic timestamps
│
├── reports/            # 📊 Generated analysis, figures, and results
│   ├── clean_loop/                     # Baseline (unmodified) experiment results
│   ├── explainability/                 # Analysis of explanation methods
│   ├── personalization/                # Analysis of personalization quality
│   ├── privacy/                        # Analysis of privacy preservation
│   ├── conversion.ipynb              # Notebook for data format conversions
│   └── generate_figures.ipynb        # Notebook to generate final plots for reports
│
└── references/         # 📄 Academic papers, literature, and external resources
```

## 🙏 Acknowledgments

We extend our sincere gratitude to:

- **Prof. Mikołaj Morzy** for his invaluable guidance, expertise, and continuous support throughout this research
- **Poznan University of Technology** for providing the academic environment and resources necessary for this work
- **Prof. Jerzy Nawrocki** for drive our Rapid Literature Review provide template for the master thesis and insightfull remarks
- **Microsoft Recommenders Team** for their open-source library that served as a foundation for collaborative filtering implementations
- **PGPR Authors** for making their reinforcement learning framework available for research purposes

## 🤝 Support

### 🐛 Found a Bug?
1. Check [existing issues](https://github.com/your-username/Quality-of-Personalization-Explainability-and-Robustness-of-Recommendation-Algorithms/issues)
2. Create a new issue with detailed description
3. Include system information and error logs

### 📧 Contact
- **Research Questions**: [julia.podsadna@example.com](mailto:julia.podsadna@example.com)
- **Technical Issues**: Create a GitHub issue

## 📋 Compliance & Ethics

This research project:
- ✅ **Properly attributes** all source code and methodologies
- ✅ **Transparently documents** modifications and extensions
- ✅ **Complies with** open-source licenses of base repositories
- ✅ **Follows** academic integrity standards
- ✅ **Addresses** EU AI Act ethical considerations

<div align="center">

**🌟 Star this repository if you find it useful!**

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
<p><small>Built with [Microsoft Recommenders](https://github.com/recommenders-team/recommenders), [PGPR](https://github.com/orcax/PGPR) and [recmetrics](https://github.com/statisticianinstilettos/recmetrics)</small></p>
</div>
