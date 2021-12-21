<!-- Description of the Project -->
## À propos

### Fait par :
Steve Levesque - Weiyue Cai

### Description résumé
Ceci est le readme.md pour les algorithmes de la competition kaggle de IFT3395.

Le projet contient les algorithmes suivants :
- Regression Logistique faite à partir de zéro (Python)
- Réseau de neurones (Keras)
- Ensemble Classifieur (Sklearn)
- Classifieur Vote Majoritaire (Sklearn)

### Introduction



<!-- Repo's Content Tree -->
## Fichiers et Répertoires
<details open>
  <summary><b>L'arbre de dossiers du project</b></summary>
    
  ``` bash
    |- \data
    |  |- \logs
    |  |  |- ...
    |  \_ \predictions
    |  |  |- ...
    |  |- sample_submission.csv
    |  |- test.csv
    |  \_ train.csv
    |
    |- \models
    |  |- \logreg
    |  |  |- activation_func.py
    |  |  |- gradient_descent.py
    |  |  |- main.py
    |  |  |- onehot.py
    |  |  |- optim.py
    |  |  \_ train_model.py
    |  |- \neuralnetwork
    |  |  \_ main.py
    |  |- \stacking
    |  |  \_ main.py
    |  \_ \voting
    |  |  \_ main.py
    |  |
    |- \utils
    |  |- dataset.py
    |  |- metric.py
    |  \_ submission.py
    \_ README.md              # This file
  ```
</details>


<!-- Getting Started -->
## Installation
Voici les programmes nécessaires :
- Python 3.x
- Les modules
- Un IDE de choix



## Comment Exécuter
Il est possible de créer une configuration avec votre IDE favori (PyCharm, VsCode, etc.) et d'exécuter le fichier main.py de chaque catégorie :
- Regression Logistique faite à partir de zéro (reglog/main.py)
- Réseau de neurones (neuralnetwork/main.py)
- Ensemble Classifieur (stacking/main.py)
- Classifieur Vote Majoritaire (voting/main.py)

Chacun des main.py peut être démarré dans le répertoire, il ne faut pas séparer le dossier "utils" contenant les fonctions auxiliaires.

Lors de la première exécution, les dossiers "logs" et "predictions" vont être crées s'il sont inexistants.



<!-- Acknowlegements and Sources -->
## Sources et Liens
Les sources sont citées dans le rapport et l'annexe joint à la remise.
