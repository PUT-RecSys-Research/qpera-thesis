# How to Cite This Work

This page provides standardized formats for citing the QPERA project, its associated Master's Thesis, and the foundational libraries it builds upon.

---

## 1. Primary Citation

If you use this project, its findings, or its code in your research, please cite the Master's Thesis.

### BibTeX Format

```bibtex
@mastersthesis{podsadna_chwilkowski2025qpera,
  title     = {Quality of Personalization, Explainability and Robustness of Recommendation Algorithms},
  author    = {Podsadna, Julia and Chwi{\l}kowski, Bartosz},
  year      = {2025},
  school    = {Poznan University of Technology},
  address   = {Poznan, Poland},
  supervisor= {Morzy, Miko{\l}aj},
  type      = {Master's Thesis},
  note      = {Faculty of Computing and Telecommunications},
  url       = {https://github.com/PUT-RecSys-Research/qpera-thesis}
}
```

---

## 2. Software Citation

To cite the software implementation directly, you can use the following format.

### BibTeX Format

```bibtex
@software{qpera_software_2025,
  author    = {Podsadna, Julia and Chwi{\l}kowski, Bartosz},
  title     = {QPERA: A Project for Evaluating Quality, Personalization, Explainability, and Robustness of Recommendation Algorithms},
  year      = {2025},
  publisher = {GitHub},
  version   = {1.0.0},
  url       = {https://github.com/PUT-RecSys-Research/qpera-thesis},
  license   = {MIT}
  // TODO: When a Zenodo release is created, add the DOI here.
  // doi    = {10.5281/zenodo.XXXXXXX}
}
```

---

## 3. Citing Foundational Works

This project builds directly upon pioneering open-source libraries. We strongly encourage you to **also cite their original papers and software** to give proper credit to the foundational work.

### Microsoft Recommenders

Used for the collaborative filtering and content-based filtering implementations.

```bibtex
@inproceedings{recommenders2019,
  author    = {Graham, Scott and Min, Jun Ki and Wu, Tao and Soni, Anish},
  title     = {Microsoft Recommenders: Tools to Accelerate Developing Recommender Systems},
  year      = {2019},
  booktitle = {Proceedings of the 13th ACM Conference on Recommender Systems (RecSys '19)},
  pages     = {542--543},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3298689.3346967}
}
```

### PGPR (Reinforcement Learning)

The basis for our reinforcement learning and explainability implementation.

```bibtex
@inproceedings{xian2019pgpr,
  author    = {Xian, Yikun and Fu, Zuohui and Muthukrishnan, S. and de Melo, Gerard and Zhang, Yongfeng},
  title     = {Reinforcement Knowledge Graph Reasoning for Explainable Recommendation},
  year      = {2019},
  booktitle = {Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '19)},
  pages     = {285--294},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3331184.3331203}
}
```

### Recmetrics

Used for calculating the `personalization` and `intra_list_similarity` metrics.

```bibtex
@software{recmetrics2020,
  author    = {Longo, Claire},
  title     = {Recmetrics: A library of metrics for evaluating recommender systems},
  year      = {2020},
  version   = {0.1.3},
  publisher = {GitHub},
  url       = {https://github.com/statisticianinstilettos/recmetrics}
}
```

---

## 4. License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute the code, but you must include the original copyright notice and license file in any derivative works. Please see the [LICENSE](../LICENSE) file for full details.