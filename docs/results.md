# Research Findings & Contributions

!!! note "Work in Progress"
    This document outlines the structure for the final research findings. The content presented here is preliminary and will be updated as the project's experiments and analysis are completed.

This document summarizes the key findings, contributions, and future directions of the QPERA Master's Thesis project.

---

## 1. Abstract

<!-- 
  TODO: Authors to write a concise (150-250 word) abstract summarizing the project's context, methods, key findings, and implications.
-->
> *Example: This research presents a comprehensive evaluation of three major recommendation algorithm families—Collaborative Filtering (CF), Content-Based Filtering (CBF), and Reinforcement Learning (RL)—across four critical dimensions: Quality, Personalization, Explainability, and Robustness (QPERA). Using a standardized evaluation pipeline, we analyze the trade-offs inherent in each approach...*

---

## 2. Key Findings

This section summarizes the main results from our experiments. The full, detailed results for every run can be explored in the **MLflow UI**.

### Performance Summary

<!-- 
  TODO: Authors to update this table with a high-level summary of the findings. 
  Use a simple rating system (e.g., ⭐ to ⭐⭐⭐⭐⭐) to compare the algorithms.
-->
| Algorithm | Personalization | Privacy Robustness | Explainability | Accuracy |
|:----------|:---------------:|:------------------:|:--------------:|:--------:|
| **Collaborative Filtering** |       *TBD*       |       *TBD*        |     *TBD*      |  *TBD*   |
| **Content-Based** |       *TBD*       |       *TBD*        |     *TBD*      |  *TBD*   |
| **Reinforcement Learning**  |       *TBD*       |       *TBD*        |     *TBD*      |  *TBD*   |

---

### Key Visualizations

<!-- 
  TODO: Authors to embed the most impactful plots from the `reports/plots` directory.
  Replace the placeholder paths with the actual paths to your generated figures.
-->
<div class="grid cards" markdown>
-   **Privacy vs. Utility Trade-off**

    ---
    ![Placeholder for Privacy Plot](https://via.placeholder.com/400x250.png?text=Privacy+vs.+Utility+Plot)

-   **Personalization Score Comparison**

    ---
    ![Placeholder for Personalization Plot](https://via.placeholder.com/400x250.png?text=Personalization+Plot)

-   **Precision-Recall Curves**

    ---
    ![Placeholder for PR Curve Plot](https://via.placeholder.com/400x250.png?text=Precision-Recall+Curves)
</div>

---

### Analysis by Research Question

<!-- 
  TODO: Authors to provide a narrative summary for each research question, drawing conclusions from the experimental data.
-->

#### 1. How does algorithm performance vary across different datasets?
-   *Your analysis here... (e.g., "On the MovieLens dataset, CF demonstrated the highest accuracy due to...")*
-   *Your analysis here... (e.g., "In contrast, CBF excelled on the Amazon Sales dataset by leveraging...")*

#### 2. What is the trade-off between user privacy and recommendation utility?
-   *Your analysis here... (e.g., "Our experiments showed that CF was surprisingly robust to metadata hiding, with performance degrading by only X%...")*
-   *Your analysis here... (e.g., "The RL model was most sensitive, as its knowledge graph structure was directly impacted by...")*

#### 3. How does the quality of personalization and explainability differ between algorithms?
-   *Your analysis here... (e.g., "In terms of personalization, the RL approach achieved the highest scores, generating more diverse recommendations...")*
-   *Your analysis here... (e.g., "For explainability, CBF provided the most transparent feature-based explanations, while RL offered path-based reasoning...")*

---

## 3. Contributions

<!-- 
  TODO: Authors to list the main contributions of this thesis.
-->
1.  **A Unified Evaluation Framework**: *Your description here... (e.g., "We designed and implemented a standardized pipeline to evaluate disparate recommendation algorithms...")*
2.  **Empirical Trade-off Analysis**: *Your description here... (e.g., "We provide quantitative evidence of the trade-offs between personalization, privacy, and explainability...")*
3.  **Reproducible Research Artifact**: *Your description here... (e.g., "The entire project is packaged for full reproducibility, serving as a valuable resource for future research.")*

---

## 4. Future Work

<!-- 
  TODO: Authors to suggest potential directions for future research based on the findings.
-->
-   **Hybrid Models**: *Your suggestion here... (e.g., "Develop hybrid algorithms that combine the strengths of different approaches...")*
-   **Advanced Explainability**: *Your suggestion here... (e.g., "Integrate Large Language Models (LLMs) to translate path-based explanations into natural language...")*
-   **Fairness and Bias Audits**: *Your suggestion here... (e.g., "Extend the evaluation framework to include metrics for fairness and bias...")*

---

## 5. How to Cite This Work

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