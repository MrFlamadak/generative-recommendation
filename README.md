# Generative Recommender System

A Generative Recommender System pipeline for generating product recommendations for the [H&M dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations).

## Contains
The repository consists of
 - ```data/```: where the H&M dataset should be
 - ```embeddings/```: where the embeddings should be stored
 - ```semantic_ids/```: where the semantic IDs should be stored
 - ```models/```: where the trained models should be stored
 - ```components/```: source code for main system building blocks (embedder, quantizer, and transformer)
 - ```data_utils/```: scripts for data handling and analysis
 - ```eval/```: includes a baseline model and model evaluation scripts
 - ```train/```: training scripts for components
 - ```pipeline.ipynb```: pipeline for the entire generative recommender system

## Dependencies
The requirements for the project can be acquired by letting an environment run:
```
pip install -r requirements.txt
```



## Structure
```
root/
├── data/
│   ├── embeddings/
│   │   └── ...
│   │
│   ├── semantic_ids/
│   │   └── ...
│   │
│   └── ...
│
├── models/
│   └── ...
│
├── src/
│   ├── components/
│   │   ├── __init__.py
│   │   ├── embedder.py
│   │   ├── quantizer.py
│   │   └── transformer.py
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
│   │   ├── quantizer_train.py
│   │   └── transformer_train.py
│   │
│   └── pipeline.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```