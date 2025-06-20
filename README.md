# Quality of Personalization, Explainability and Robustness of Recommendation Algorithms

<div align="center">

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>
<img src="https://img.shields.io/badge/Python-3.9+-blue.svg" />
<img src="https://img.shields.io/badge/License-MIT-green.svg" />
<img src="https://img.shields.io/badge/Status-Active-brightgreen.svg" />
<img src="https://img.shields.io/badge/MLflow-Tracking-blue.svg" />
<img src="https://img.shields.io/badge/PyTorch-Framework-red.svg" />

**A Master's Thesis Project**

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

## 📋 Table of Contents

- [Abstract](#-abstract)
- [Research Questions](#-research-questions)
- [Key Features](#-key-features)
- [Getting Started](#-getting-started)
- [Running Experiments](#-running-experiments)
- [Datasets](#-datasets)
- [Results & Publications](#-results--publications)
- [Project Structure](#-project-structure)
- [Computing Environment](#-computing-environment)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Citation](#-citation)
- [Compliance & Ethics](#-compliance--ethics)

---

## 📖 Abstract

This Master's thesis presents an in-depth analysis of three main families of recommendation algorithms (**collaborative filtering**, **content-based filtering**, and **reinforcement learning**) in terms of their:

- 🛡️ **Resilience** to data-related stresses
- 🔒 **Resistance** to anonymization techniques  
- 🔍 **Explainability** and ability to generate meaningful explanations
- ⚖️ **Ethical risks** associated with algorithm implementation

The research is particularly relevant given the recently adopted **EU AI Act** and the introduction of ethical risk taxonomy, as well as the **Omnibus Directive**. The project encompasses multiple recommendation scenarios across different datasets and application domains, implementing comprehensive metrics to assess algorithm behavior across the evaluated criteria.

**Note**: This project builds upon existing implementations from the [Microsoft Recommenders](https://github.com/recommenders-team/recommenders) library (CF & CBF algorithms) and [PGPR](https://github.com/Master-s-thesis-PPERA/PGPR) (RL algorithms), with significant modifications and a unified custom dataset loader for comparative analysis.

---

## ❓ Research Questions

This research addresses the following key questions:

1. **Robustness Analysis**: How do different recommendation algorithm families compare in terms of resilience to data anonymization and perturbation techniques?

2. **Privacy-Personalization Trade-off**: What is the relationship between recommendation accuracy, personalization quality, and user privacy preservation for each algorithm family?

3. **Explainability Assessment**: To what extent can each algorithm generate meaningful explanations for its recommendations, and how does this capability affect user trust and system transparency?

4. **Ethical Risk Evaluation**: How can ethical risks associated with each recommender system be identified, measured, and mitigated in accordance with EU AI Act requirements?

---

## ✨ Key Features

- 🔄 **Multi-Algorithm Implementation**: 
  - Collaborative Filtering & Content-Based Filtering (adapted from [Microsoft Recommenders](https://github.com/recommenders-team/recommenders))
  - Reinforcement Learning (extensively modified from [PGPR](https://github.com/Master-s-thesis-PPERA/PGPR))
- 🔗 **Unified Data Pipeline**: Custom dataset loader ensuring consistent data processing across all algorithms
- 🛡️ **Privacy-Preserving Techniques**: Data anonymization and perturbation methods
- 📊 **Comprehensive Evaluation**: Multiple datasets (MovieLens, Amazon Sales, PostRecommendations)
- 🔍 **Explainability Metrics**: Novel metrics for recommendation transparency
- ⚖️ **Ethical Risk Assessment**: AI Act compliance evaluation framework
- 📈 **MLflow Integration**: Experiment tracking and reproducibility
- 🐍 **Modern Python Stack**: PyTorch, scikit-learn, pandas, numpy

---

## 🚀 Getting Started

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.9+
- Git

### Quick Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GambiBambi/personalization-privacy-and-explainability-of-recommendation-algorithms.git
   cd personalization-privacy-and-explainability-of-recommendation-algorithms
   ```

2. **One-command setup:**
   ```bash
   make install && conda activate ppera-env && make setup
   ```

3. **Verify installation:**
   ```bash
   make help
   ```

### Manual Installation (Alternative)

<details>
<summary>Click to expand manual installation steps</summary>

1. **Create Conda environment:**
   ```bash
   conda env create -f environment.yml --name ppera-env
   ```

2. **Activate environment:**
   ```bash
   conda activate ppera-env
   ```

3. **Install PyTorch (CPU):**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Install project in editable mode:**
   ```bash
   python -m pip install -e .
   ```

</details>

### Update Dependencies

```bash
make requirements
```

---

## 🧪 Running Experiments

The `Makefile` provides streamlined commands for running experiments:

### Quick Start
```bash
make run-all          # Complete pipeline with auto-cleanup
```

### Interactive Mode
```bash
make run-interactive  # Keep MLflow server running for result inspection
```

### Individual Components
```bash
make run-mlflow      # Start MLflow server only
```

**MLflow UI**: Access experiment results at [http://127.0.0.1:8080](http://127.0.0.1:8080)

### Development Commands
```bash
make lint            # Code quality checks
make format          # Auto-format code
make clean           # Clean temporary files
```

---

## 📊 Datasets

The project evaluates algorithms across three diverse datasets from different domains:

| Dataset | Domain | Processed Size | Description | Source |
|---------|--------|---------------|-------------|---------|
| **[MovieLens 20M](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)** | Entertainment | 14K interactions | Movie ratings and metadata from MovieLens users | Kaggle/GroupLens |
| **[Post Recommendations](https://www.kaggle.com/datasets/vatsalparsaniya/post-pecommendation)** | Social Media | 14K interactions | Social media post engagement and recommendation data | Kaggle |
| **[Amazon Sales](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)** | E-commerce | 1.4K products | Product information and sales data from Amazon | Kaggle |

### Dataset Processing & Modifications

#### 🔧 Automatic Data Preprocessing

The pipeline automatically applies several modifications to ensure dataset compatibility across all algorithms:

**Rating Generation:**
- 📱 **Post Recommendations**: Originally lacks rating information → **Automatic rating generation** based on engagement metrics
- 🛒 **Amazon Sales**: Contains overall product ratings → **User-specific ratings generated** from individual user-product interactions

**Timestamp Handling:**
- 🛒 **Amazon Sales**: Originally lacks timestamp information → **Automatic timestamp assignment** for temporal consistency
- 📱 **Post Recommendations**: Timestamp normalization for consistent temporal analysis

**Format Standardization:**
- All datasets are converted to a unified format: `(user_id, item_id, rating, timestamp)`
- Consistent data types and value ranges across all datasets
- Automatic handling of missing values and data validation

#### ⚡ Computational Limitations & Sampling

Due to computational power limitations and resource constraints:

| Dataset | Original Size | Processed Size | Sampling Strategy |
|---------|--------------|----------------|-------------------|
| **MovieLens** | ~20M interactions | **14K interactions** | Random sampling to ensure all algorithms can complete |
| **Post Recommendations** | ~70K interactions | **14K interactions** | Random sampling for consistency |
| **Amazon Sales** | ~1.4K products | **Full dataset (1.4K)** | Complete dataset used |

**Rationale:**
- 🔄 **Consistency**: 14K sample size ensures all three algorithm families can complete successfully
- 💻 **Resource Constraints**: Computational limitations prevent processing of larger datasets
- 🎯 **Fair Comparison**: Uniform dataset sizes enable meaningful cross-algorithm comparison
- 🔍 **Proof of Concept**: Sample size sufficient for demonstrating methodology and approach
- ⏱️ **Training Time**: Manageable dataset sizes allow for comprehensive experimentation across all algorithms

### Dataset Setup Instructions

#### 📥 Manual Download Required

**Important**: Datasets must be downloaded manually from Kaggle and placed in the existing dataset directories before running experiments.

**Dataset directories are already created at:**
```
datasets/
├── AmazonSales/        (ready for files) 
├── MovieLens/          (ready for files) 
└── PostRecommendations/ (ready for files)
```

#### Download and Setup Process

1. **Download each dataset from Kaggle:**
   - **Amazon Sales**: [Download Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)
   - **MovieLens**: [Download MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)
   - **Post Recommendations**: [Download Post Recommendations Dataset](https://www.kaggle.com/datasets/vatsalparsaniya/post-pecommendation)

2. **Extract and place files in respective directories:**
   ```
   datasets/
   ├── AmazonSales/
   │   └── amazon.csv
   ├── MovieLens/
   │   ├── ratings.csv
   │   ├── movies.csv
   │   ├── tags.csv
   │   └── links.csv
   └── PostRecommendations/
       └── posts.csv
   ```

#### ⚠️ Important Notes
- **Kaggle Account Required**: You'll need a Kaggle account to download these datasets
- **File Formats**: Ensure CSV files are extracted properly from ZIP archives
- **Directory Names**: Files must be placed in the exact directories shown above
- **Automatic Processing**: All dataset modifications happen automatically during pipeline execution
- **Data License**: All datasets are used under their respective licenses for academic research

### Dataset Details

#### 🛒 Amazon Sales Dataset
- **Original Size**: ~1.4K products (1,362 records)
- **Processed Size**: **Full dataset** (no sampling)
- **Features**: Product categories, prices, reviews
- **Modifications**: 
  - ✅ User-specific ratings generated from product interactions
  - ✅ Timestamps automatically assigned for temporal consistency
- **Use Case**: E-commerce recommendation scenarios
- **Privacy Concerns**: Purchase behavior, product preferences

#### 🎬 MovieLens Dataset
- **Original Size**: ~20M interactions (20,000,264 records)
- **Processed Size**: **14K interactions** (sampled)
- **Features**: User ratings, movie genres, tags, timestamps
- **Modifications**: 
  - ✅ Random sampling to 14K interactions
  - ✅ Format standardization (minimal changes needed)
- **Use Case**: Traditional collaborative filtering evaluation
- **Privacy Concerns**: User behavior patterns, rating preferences

#### 📱 Post Recommendations Dataset
- **Original Size**: ~70K interactions (70,616 records)
- **Processed Size**: **14K interactions** (sampled)
- **Features**: Post content, user interactions, engagement metrics
- **Modifications**: 
  - ✅ Ratings generated from engagement metrics
  - ✅ Timestamp normalization
  - ✅ Random sampling to 14K interactions
- **Use Case**: Content-based and social recommendation evaluation
- **Privacy Concerns**: Content preferences, social interaction patterns

### Verification

After downloading and placing datasets, verify the setup:

```bash
# Check if required files are in place
ls datasets/AmazonSales/
ls datasets/MovieLens/
ls datasets/PostRecommendations/

# Expected output should show:
# AmazonSales/: amazon.csv
# MovieLens/: ratings.csv, movies.csv, tags.csv, links.csv  
# PostRecommendations/: posts.csv
```

**After running experiments, processed files will appear:**
```bash
# Check processed datasets (created during pipeline execution)
ls ppera/datasets/AmazonSales/
ls ppera/datasets/MovieLens/
ls ppera/datasets/PostRecommendations/

# Expected output should show:
# Each directory will contain: merge_file.csv (processed dataset)
```

**Note**: The `merge_file.csv` files are automatically created during the data preprocessing phase and contain the unified, processed datasets used by all algorithms.

**Optional**: Test the setup by running the full experimental pipeline:
```bash
# This will run the complete experimental pipeline including:
# - Data preprocessing and merge_file.csv creation
# - Model training and evaluation across all algorithms and datasets
# - Results logging to MLflow
make run-all
```
*Note: The system automatically validates dataset structure, applies necessary modifications, and provides detailed processing logs including sampling statistics and data transformation reports.*

---

### Data Processing Pipeline

Once datasets are properly placed, the custom dataset loader handles:

- 🔄 **Unified Format**: Standardized user-item interaction format with automatic schema conversion
- 🎯 **Smart Sampling**: Intelligent sampling strategies for computational efficiency
- 📊 **Rating Generation**: Automatic rating synthesis for datasets lacking explicit ratings
- ⏰ **Timestamp Handling**: Automatic timestamp assignment and normalization
- 🛡️ **Privacy Transformations**: Configurable anonymization techniques
- 🎯 **Personalization Features**: User-specific data modifications and personalization techniques
- 📊 **Train/Test Splits**: Stratified splitting with reproducible seeds
- 📝 **Data Validation**: Comprehensive data quality checks and error reporting

---

## 📈 Results & Publications

### Thesis Document
- 📄 **[Master's Thesis]** *(Available upon completion - will include detailed experimental results and analysis)*
- 📊 **[Defense Presentation]** *(Available after defense)*

### Research Outputs
- 📊 **Experimental Results**: Comprehensive analysis available in the Master's thesis document
- 📈 **MLflow Logs**: Complete experiment tracking and reproducible results

### Potential Future Publications
- 📝 **Conference Paper**: *Under consideration for submission to recommendation systems venues*
- 📋 **Technical Report**: *May be published based on thesis outcomes*

### Reproducibility
- 🔄 All experiments are fully reproducible using the provided seeds and environment
- 📊 Results and logs are tracked using MLflow
- 📁 Source code and methodology openly available for academic use

---

## 📁 Project Structure

<details>
<summary>Click to expand full project structure</summary>

```
├── LICENSE
├── Makefile           ← Commands like `make run-all`
├── README.md          ← This file
├── environment.yml    ← Conda dependencies
├── pyproject.toml     ← Project configuration
│
├── docs/              ← Documentation (MkDocs)
│   ├── docs/
│   │   ├── index.md
│   │   └── getting-started.md
│   └── README.md
│
├── datasets/          ← Raw datasets
│   ├── AmazonSales/
│   ├── MovieLens/
│   └── PostRecommendations/
│
├── models/            ← Trained models
├── notebooks/         ← Jupyter notebooks
│   └── CF_BPR.ipynb
│
├── reports/           ← Analysis results
│   ├── figures/
│   └── tables/
│
├── references/        ← Academic papers and documentation
│
└── ppera/             ← Source code
    ├── __init__.py
    ├── main.py        ← Main experiment runner
    │
    ├── rl_tmp/        ← RL temporary files (.gitignored)
    │   ├── AmazonSales/
    │   ├── MovieLens/
    │   └── PostRecommendations/
    │
    ├── metrics/       ← Experiment results
    │
    ├── CF.py          ← Collaborative Filtering
    ├── CBF.py         ← Content-Based Filtering  
    ├── RL.py          ← Reinforcement Learning
    │
    ├── rl_*.py        ← RL-specific modules
    ├── datasets_loader.py    ← Data loading utilities
    ├── data_manipulation.py  ← Privacy/personalization
    ├── metrics.py     ← Evaluation metrics
    ├── log_mlflow.py  ← Experiment tracking
    └── ...            ← Other utilities
```

</details>

---

## 💻 Computing Environment

### Hardware Specifications
- **Memory**: 16 GB RAM
- **Processors**: 12 cores
- **Storage**: 100 GB available space
- **Architecture**: x86_64

### Software Environment
- **Operating System**: Debian GNU/Linux (Virtual Machine)
- **Python**: 3.9+
- **Conda**: Miniconda/Anaconda
- **Virtualization**: VMware/VirtualBox (if applicable)

### Performance Considerations
- **Dataset Processing**: 14K samples chosen based on available computational resources
- **Training Time**: Approximate training times per algorithm (clean runs without privacy/personalization modifications):

#### MovieLens Dataset (14K interactions)
- Content-Based Filtering: ~3.1 minutes (187.68 seconds)
- Collaborative Filtering: ~X minutes
- Reinforcement Learning: ~X hours

#### Amazon Sales Dataset (1.4K products)
- Content-Based Filtering: ~21 seconds (20.85 seconds)
- Collaborative Filtering: ~X minutes
- Reinforcement Learning: ~X hours

#### Post Recommendations Dataset (14K interactions)
- Content-Based Filtering: ~2.4 minutes (145.54 seconds)
- Collaborative Filtering: ~X minutes
- Reinforcement Learning: ~X hours

**Note**: Training times may increase significantly when privacy anonymization or personalization modifications are enabled, as these require additional data processing and can affect algorithm convergence.

*Timing data for Collaborative Filtering and Reinforcement Learning will be updated as experiments are completed.*

### Reproducibility Notes
- All experiments run on the same VM configuration
- Random seeds fixed for deterministic results
- Dataset sampling strategy documented in [Datasets section](#-datasets)

---

## 🤝 Contributing

This is an academic research project. For questions or suggestions:

1. 📧 **Email**: 
   - Julia Podsadna: [juliap362@gmail.com](mailto:juliap362@gmail.com)
   - Bartosz Chwiłkowski: [bartosz.chwilkowski00@gmail.com](mailto:bartosz.chwilkowski00@gmail.com)
2. 🐛 **Issues**: Use GitHub Issues for bug reports
3. 💡 **Ideas**: Discussions welcome via GitHub Discussions

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Source Repositories
- **Microsoft Recommenders Team**: For the foundational CF and CBF implementations ([GitHub](https://github.com/recommenders-team/recommenders))
- **PGPR Authors**: For the original reinforcement learning recommendation framework ([GitHub](https://github.com/Master-s-thesis-PPERA/PGPR))

### Academic Support
- **Supervisor**: Prof. Mikołaj Morzy for guidance and support
- **University**: Poznan University of Technology for providing resources

### Technical Community
- **Open Source Libraries**: Contributors of PyTorch, scikit-learn, MLflow, and pandas
- **Dataset Providers**: MovieLens, Amazon, and PostRecommendations dataset maintainers

### Related Work
- Zhao, W. X., et al. "RecBole: Towards a unified, comprehensive and efficient framework for recommendation algorithms." *Proceedings of the 30th ACM International Conference on Information & Knowledge Management.* 2021.
- Xian, Y., et al. "Reinforcement knowledge graph reasoning for explainable recommendation." *Proceedings of the 42nd international ACM SIGIR conference on research and development in information retrieval.* 2019.

#### Microsoft Recommenders Framework
- Argyriou, A., González-Fierro, M., and Zhang, L. "Microsoft Recommenders: Best Practices for Production-Ready Recommendation Systems." *WWW 2020: International World Wide Web Conference Taipei*, 2020. Available: https://dl.acm.org/doi/abs/10.1145/3366424.3382692
- Graham, S., Min, J.K., Wu, T. "Microsoft recommenders: tools to accelerate developing recommender systems." *RecSys '19: Proceedings of the 13th ACM Conference on Recommender Systems*, 2019. Available: https://dl.acm.org/doi/10.1145/3298689.3346967
- Zhang, L., Wu, T., Xie, X., Argyriou, A., González-Fierro, M., and Lian, J. "Building Production-Ready Recommendation System at Scale." *ACM SIGKDD Conference on Knowledge Discovery and Data Mining 2019 (KDD 2019)*, 2019.

#### Educational Resources
- González-Fierro, M. "Recommendation Systems: A Practical Introduction." LinkedIn Learning, 2024. Available: https://www.linkedin.com/learning/recommendation-systems-a-practical-introduction
- Li, D., Lian, J., Zhang, L., Ren, K., Lu, D., Wu, T., Xie, X. "Recommender Systems: Frontiers and Practices." Springer, Beijing, 2024. Available: https://www.amazon.com/Recommender-Systems-Frontiers-Practices-Dongsheng/dp/9819989639/

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{podsadna2025recommendation,
  title={Quality of Personalization, Explainability and Robustness of Recommendation Algorithms},
  author={Podsadna, Julia and Chwiłkowski, Bartosz},
  year={2025},
  school={Poznan University of Technology},
  type={Master's thesis},
  note={Building upon Microsoft Recommenders and PGPR frameworks}
}
```

### Source Repositories Citations

Please also cite the original works:

```bibtex
@misc{recommenders2019,
  title={Microsoft Recommenders},
  author={Microsoft Corporation},
  year={2019},
  howpublished={\url{https://github.com/recommenders-team/recommenders}}
}

@inproceedings{argyriou2020microsoft,
  title={Microsoft Recommenders: Best Practices for Production-Ready Recommendation Systems},
  author={Argyriou, Andreas and González-Fierro, Miguel and Zhang, Liang},
  booktitle={Proceedings of the Web Conference 2020},
  pages={50--51},
  year={2020}
}

@inproceedings{graham2019microsoft,
  title={Microsoft recommenders: tools to accelerate developing recommender systems},
  author={Graham, Scott and Min, Jun Ki and Wu, Tao},
  booktitle={Proceedings of the 13th ACM Conference on Recommender Systems},
  pages={542--543},
  year={2019}
}

@inproceedings{zhang2019building,
  title={Building Production-Ready Recommendation System at Scale},
  author={Zhang, Liang and Wu, Tao and Xie, Xing and Argyriou, Andreas and González-Fierro, Miguel and Lian, Jianxun},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={2788--2789},
  year={2019}
}

@article{xian2019reinforcement,
  title={Reinforcement knowledge graph reasoning for explainable recommendation},
  author={Xian, Yikun and Fu, Zuohui and Muthukrishnan, S and De Melo, Gerard and Zhang, Yongfeng},
  journal={Proceedings of the 42nd international ACM SIGIR conference on research and development in information retrieval},
  year={2019}
}
```

---

## 📋 Compliance & Ethics

This research project:
- ✅ **Properly attributes** all source code and methodologies
- ✅ **Transparently documents** modifications and extensions
- ✅ **Complies with** open-source licenses of base repositories
- ✅ **Follows** academic integrity standards
- ✅ **Addresses** EU AI Act ethical considerations

---

<div align="center">

**🌟 Star this repository if you find it useful!**

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

</div>