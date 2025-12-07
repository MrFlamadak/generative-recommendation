# Generative Recommender System

## Structure
```
root/
├── data/
│   └── ...
│
├── models/
│   └── ...
│
├── src/
│   ├── components/
│   │   ├── quantizer/
│   │   │   ├── __init__.py
│   │   │   └── rq_vae.py
│   │   │
│   │   └── transformer/
│   │       ├── __init__.py
│   │       └── bart.py
│   │
│   ├── data_utils/
│   │   ├── __init__.py
│   │   ├── data_analyzer.py
│   │   └── data_handler.py
│   │
│   ├── eval/
│   │   ├── baseline/
│   │   │   ├── __init__.py
│   │   │   └── collaborative_filtering.py
│   │   │
│   │   ├── __init__.py
│   │   ├── cosine_similarity.py
│   │   ├── evaluation.py
│   │   └── loss_plot.py
│   │
│   ├── train/
│   │   ├── __init__.py
│   │   ├── bart_train.py
│   │   └── rq_vae_train.py
│   │
│   └── pipeline.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```