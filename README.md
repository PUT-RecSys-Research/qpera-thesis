# personalization-privacy-and-explainability-of-recommendation-algorithms

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Celem projektu jest dogłębna analiza czterech głównych rodzin algorytmów rekomendacyjnych (collaborative filtering, content-based, learning2rank, reinforcement learning) pod kątem ich odporności na obciążenia występujące w danych, odporności na anonimizację, zdolność algorytmów do generowania wyjaśnień oraz ryzyka etyczne związane z wdrożeniem danego algorytmu. Ostatni punkt wiąże się z przyjętym niedawno przez Unię Europejską AI Act i wprowadzeniem taksonomii ryzyk etycznych, a także z dyrektywą Omnibus. W ramach projektu planuje się wybranie różnych scenariuszy rekomendacji (zbiorów danych i obszarów zastosowania), implementację metryk do oceny odporności algorytmów, oraz przeprowadzenie szeroko zakrojonych eksperymentów badających zachowanie się poszczególnych rodzin algorytmów w przestrzeni ocenianych kryteriów.

W ramach projektu należy porównać cztery rodzaje rekomenderów:
- R1: content-based
- R2: collaborative filtering
- R3: learning2rank
- R4: reinforcement learning

Do porównania należy wykorzystać trzy zbiory danych o zróżnicowanej charakterystyce:
- D1: tradycyjny zbiór reprezentujący rekomendacje produktowe (np. filmy, produkty w sklepie, książki, itp.)
- D2: zbiór danych reprezentujący konieczność rekomendacji ze strumienia (np. wyniki zapytań, strumienie informacji z sieci społecznościowych, itp.)
- D3: zbiór danych reprezentujący rekomendacje "trudne", tj. rekomendacje akcji które są rzadkie i drogie (np. rekomendacje hoteli lub innych tego typu usług)

Porównanie rekomenderów odbędzie się pod kątem trzech aspektów:
- A1: wyjaśnialność (explainability): na ile dany algorytm jest w stanie wygenerować wyjaśnienie wybranych rekomendacji
- A2: personalizacja (personalization): jak duża jest wariancja rekomendacji w zależności od podobieństwa między klientami
- A3: prywatność (privacy): jak czuły jest algorytm na podniesienie poziomu prywatności (np. poprzez randomizację atrybutów klienta lub ograniczeniu historii klienta)

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         personalization_privacy_and_explainability_of_recommendation_algorithms and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── personalization_privacy_and_explainability_of_recommendation_algorithms   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes personalization_privacy_and_explainability_of_recommendation_algorithms a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

