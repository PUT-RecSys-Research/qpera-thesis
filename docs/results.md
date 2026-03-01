# Research Findings & Contributions

This document summarizes the key findings, contributions, and conclusions of the QPERA Master's Thesis project.

---

## 1. Abstract

The aim of this study was to investigate how recommendation algorithms — Content-Based Filtering (CBF), Collaborative Filtering (CF), and Reinforcement Learning (RL) — perform in terms of three key aspects: robustness to the privatization of user history, the ability to personalize recommendations, and the explainability of the generated results. The algorithms were tested using three datasets:

- **AmazonSales** — representing recommendations involving high-cost decision-making,
- **MovieLens** — representing recommendations involving moderate-cost decision-making,
- **PostRecommendations** — representing recommendations involving low-cost decision-making.

Their performance was assessed across defined scenarios using recommendation quality metrics and feedback from survey participants. The results indicated that none of the evaluated algorithms consistently outperformed the others in terms of robustness and personalization. However, with regard to explainability, respondents generally favored Content-Based Filtering. Identifying a single algorithm that excels across all examined dimensions was not possible. Each method of recommendation demonstrated distinct strengths and weaknesses, which became evident during the experimental evaluation.

---

## 2. Key Findings

### Performance Summary

| Dimension | CBF | CF | RL |
|:----------|:---:|:--:|:--:|
| **Personalization Robustness** | ⭐⭐⭐⭐⭐ Most robust | ⭐⭐⭐ Dataset-dependent | ⭐ Most fragile |
| **Robustness (Metadata Hiding)** | ⭐⭐⭐ Variable | ⭐⭐⭐⭐⭐ Perfectly immune | ⭐ Erratic |
| **Robustness (Row Removal)** | ⭐⭐⭐ Inconsistent | ⭐⭐⭐⭐ Structured patterns | ⭐ Extremely volatile |
| **Explainability** | ⭐⭐⭐⭐⭐ Highest rated | ⭐⭐⭐ Middle ground | ⭐⭐ Lowest rated |

---

### Baseline Results (No Modifications)

All experiments were run on datasets limited to a maximum of 14,000 rows.

- **CBF** performed poorly in precision and recall across all datasets, but inter-diversity remained very high. For AmazonSales and PostRecommendations, CBF had the lowest or nearly lowest MAE and RMSE values.
- **CF** showed very low precision, recall, and inter-diversity for AmazonSales — the same item was recommended for nearly every user. For MovieLens, correct predictions emerged and CF outperformed other algorithms in multi-recommendation scenarios.
- **RL** performed best in precision and recall for AmazonSales (Precision@10 ≈ 80%, Recall@10 ≈ 80%). RL was the only algorithm to achieve significant item and user coverage for AmazonSales. It also performed very well on MovieLens for MAE and RMSE (MAE < 1, RMSE slightly above 1).

---

### Personalization

The personalization experiment measured algorithmic resilience against profile dilution with globally popular items at four levels (10%, 25%, 50%, 80%).

**Content-Based Filtering** emerged as the most robust algorithm:

- Remarkably stable across MovieLens and PostRecommendations with only minor degradation.
- On AmazonSales, performance was paradoxically *enhanced* by profile dilution — Precision@10 spiked by over 300–500% at higher dilution levels, as popular items with strong content features made matching easier.
- Diversity metrics remained largely stable across all conditions.

**Collaborative Filtering** showed highly conditional performance:

- Completely unaffected on AmazonSales (0% change across all metrics).
- Consistent moderate decline on MovieLens as dilution increased.
- Highly volatile on PostRecommendations with a sawtooth oscillation pattern, indicating the BPR model settled into vastly different solutions depending on dilution level.

**Reinforcement Learning** proved to be the most fragile:

- Near-total performance collapse (approaching -100% delta) across all datasets even at the lowest level of dilution.
- The RL agent fundamentally failed to distinguish genuine user preferences from injected generic signals.
- Diversity metrics were stable but irrelevant given the recommendation quality collapse.

---

### Robustness

Two robustness stress tests were conducted:

#### Metadata Hiding (Replacing values in title/genres columns with NaN)

- **CF** was perfectly immune (0% delta across all metrics), as BPR relies exclusively on the user-item interaction matrix and is blind to content metadata.
- **CBF** showed dataset-dependent behavior: stable on AmazonSales, improved on MovieLens (Recall peaked at ~150% improvement), but severely degraded on PostRecommendations (Precision, MRR, NDCG dropping 60%+).
- **RL** was highly erratic — volatile and non-linear responses across all datasets.

#### Row Removal (Removing entire interaction records)

- **CBF** showed paradoxical behavior: performance *improved* on MovieLens (Precision up 125% at highest removal), suggesting noise reduction, but degraded predictably on PostRecommendations.
- **CF** produced three distinct patterns: perfect stability on AmazonSales, predictable decline on MovieLens, and a dramatic 100% performance spike at "medium" modification on PostRecommendations.
- **RL** exhibited extreme volatility: chaotic boom-and-bust cycles on AmazonSales (Precision surging above 250% then collapsing to -100%), consistent degradation on MovieLens (NDCG collapsed to -100%), and fluctuating underperformance on PostRecommendations.

---

### Explainability

A survey with **36 participants** (19 with IT experience, 17 without) evaluated recommendation explanations using the SAGES framework.

**Overall results:**

- **CBF** was rated highest in satisfaction and simplicity — twice as likely to receive the maximum score. It was the most preferred across nearly all SAGES categories, with explanations based on item-to-item similarity.
- **CF** ranked last in subjective satisfaction ratings. Simplicity was rated similarly to RL but significantly lower than CBF. The only areas where CF was rated highest were "extendable" by IT-experienced respondents and "source-based" by non-IT respondents.
- **RL** received the lowest overall ratings across defined criteria. Despite strong performance on AmazonSales specifically, it did not achieve the highest ranking in any category when averaged. Respondents noted that RL explanations did not adequately outline the recommendation process.

**Per-dataset highlights:**

- **AmazonSales**: RL received the highest satisfaction; CBF was highest in simplicity. CBF received lower satisfaction partly because it recommended the same product the user had already purchased.
- **MovieLens**: CBF was highest in simplicity and adaptability. RL received the lowest ratings in almost all categories.
- **PostRecommendations**: CBF performed best across most categories. RL received slightly better satisfaction than CF but ranked last in all other SAGES criteria.

!!! important "Key Observation"
    Despite all efforts, the assessment of explainability was often viewed through the lens of recommendation quality. When a user was dissatisfied with the recommendation, the explanation was also perceived as unsatisfactory.

---

## 3. Conclusions

This empirical study demonstrates that **no single recommendation algorithm is universally superior**. The effectiveness and resilience of each are highly dependent on the specific experimental scenario, the nature of data modifications, and the underlying statistical properties of the dataset.

A clear stability hierarchy was observed, with RL consistently proving to be the most fragile and unpredictable model, while both CBF and CF exhibited variable performance. The explainability assessment revealed a distinct user preference for explanations grounded in item-to-item similarity, a characteristic inherent to content-based approaches.

### Limitations

- Experiments were conducted on **relatively small data samples** (up to 14,000 rows), which may influence generalizability.
- This was most evident on AmazonSales while using CF, where the dataset's small size (1,351 records) led to observable performance artifacts.
- The algorithms used were specific **open-source implementations** adapted for this research — alternative implementations (e.g., k-Nearest Neighbors for CF) might yield different outcomes.
- Computational constraints limited the scale of analysis to a high-performance personal computer rather than a large-scale server environment.

---

## 4. Contributions

1.  **Comprehensive Evaluation Framework**: A standardized pipeline to evaluate disparate recommendation algorithms across personalization, privacy robustness, and explainability using consistent metrics.
2.  **Empirical Trade-off Analysis**: Quantitative evidence of the trade-offs between personalization, privacy, and explainability across three algorithm families and three datasets.
3.  **Reproducible Research Artifact**: The entire project is packaged for full reproducibility with MLflow tracking, serving as a resource for future research.

---

## 5. Future Work

-   **Larger Datasets**: Utilize more powerful hardware and test on larger datasets or extended subsets to give algorithms that struggled with ranking accuracy due to insufficient data the opportunity to demonstrate their full potential.
-   **Alternative Implementations**: Test alternative algorithm implementations (e.g., k-Nearest Neighbors for CF) to determine if findings are implementation-specific.
-   **Hybrid Models**: Develop hybrid algorithms that combine the strengths of different approaches.
-   **Advanced Explainability**: Integrate Large Language Models (LLMs) to translate path-based explanations into natural language.
-   **Fairness and Bias Audits**: Extend the evaluation framework to include metrics for fairness and bias.

---

## 6. How to Cite This Work

If you use this project in your research, please cite our work. For detailed citation formats, please see the [**Citation Guide**](citation.md).

```bibtex
@mastersthesis{podsadna_chwilkowski2025qpera,
  title     = {Quality of Personalization, Explainability and Robustness of Recommendation Algorithms},
  author    = {Podsadna, Julia and Chwi{\l}kowski, Bartosz},
  year      = {2025},
  school    = {Poznan University of Technology},
  url       = {https://github.com/PUT-RecSys-Research/qpera-thesis}
}
```
