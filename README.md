Facefeel/
│
├── data/                      # Dossiers liés aux données
│   ├── raw/                   # Données brutes (ex: FER2013.csv ou images originales)
│   ├── processed/             # Données prétraitées (train/val/test split)
│
├── notebooks/                # Jupyter notebooks pour exploration initiale
│   └── data_exploration.ipynb
│
├── src/                      # Code source principal
│   ├── data_loader.py        # Chargement et prétraitement des données (PyTorch Dataset)
│   ├── model.py              # Définition du modèle CNN ou fine-tuning
│   ├── train.py              # Entraînement du modèle
│   ├── evaluate.py           # Évaluation et métriques
│   └── predict.py            # Inference sur une image ou webcam
│
├── utils/                    # Fonctions utilitaires
│   ├── transforms.py         # Transformations PyTorch (resize, normalize, etc.)
│   └── visualization.py      # Affichage de prédictions, courbes, confusion matrix
│
├── experiments/              # Sauvegarde des modèles et logs
│   ├── checkpoints/          # Fichiers .pt du modèle entraîné
│   └── logs/                 # Logs d'entraînement
│
├── config/                   # Fichiers de configuration YAML ou JSON
│   └── config.yaml
│
├── main.py                   # Point d’entrée du programme (exécute l'entraînement)
├── requirements.txt          # Dépendances Python (PyTorch, torchvision, etc.)
└── README.md                 # Présentation du projet

