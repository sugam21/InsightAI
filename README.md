# 💻🔍InsightAI: Bringing insights from images to light
## B.Tech Final Year Project: Image-Based Information Retrieval System
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Redis](https://img.shields.io/badge/redis-%23DD0031.svg?style=for-the-badge&logo=redis&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

Table of content
----
1. [Project Overview](###Project-Overview)
2. [Project Structure]()
3. [Setting up Project]()
4. [Docker]()
5. [Redis]()
6. [CromaDB]()
7. [Local LLM]()
8. [Downloading Data]()
9. [Start Training]()
10. [Stored checkpoints]()
11. [Try the model locally -> using your docker]()

### Project-Overview
This project is an advanced Information Retrieval System designed to handle image-based queries. Trained on a comprehensive laptop dataset, our model takes an image as input and allows you to ask detailed questions about it. The twist? We assume you have no prior knowledge about the image.

Leveraging the power of Retrieval-Augmented Generation (RAG), our system intelligently retrieves information from manuals and answers your questions based on the image data provided. Whether you're looking for specs, features, or troubleshooting info, our model has got you covered.

You can start to replicate this project by cloning this repo.
```bash
git clone https://github.com/sugam21/InsightAI.git
```

### Project Structure

#### ✸ Folder Tree
```
├── checkpoints📂
├── configs📂
│   ├── config.json📄
├── dataloader📂
│   ├── dataloader.py📄
├── evaluation📂
│   ├── test.py📄
├── logger📂
│   ├── logger.py📄
│   ├── logger_config.json📄
├── model📂
│   ├── base_model.py📄
│   ├── loss.py📄
│   ├── metric.py📄
│   ├── model.py📄
├── notebooks📂
│   ├── trial.ipynb
├── trainer📂
│   ├── train.py📄
├── utils📂
│   ├── config_parser.py📄
│   ├── utils.py📄
├── main.py📄
├── requirements.txt📄
└──README.md📄
```
#### ✸ Folder and File specification
| Folder      | Files         | Purpose                                                                                                       |
| ----------- | ------------- | ------------------------------------------------------------------------------------------------------------- |
| checkpoints | checkpoints   | Contains the model checkpoints.                                                                               |
| configs     | config.json   | configuration required to train the model, mostly folder paths, hyperparameters, metrics, loss functions etc. |
| dataloader  | dataloader.py | python files for loading and preprocessing data.                                                              |
| evaluation  | test.py       | python files to test and evaluate the model performance.                                                      |
| logger      | logger.py     | python files to log the result of the model.                                                                  |
| model       | loss.py       | python file containing loss functions.                                                                        |
| model       | metric.py     | python file containing evaluation metrics.                                                                    |
| model       | model.py      | python file containing custom model architecture or pre-trained models.                                       |
| trainer     | train.py      | python files which are used to train the model. This is the file that main.py file will use.                  |
| notebooks   | trail.ipynb   | Includes all the jupyter or .ipynb files that we built while experimentating.                                 |
| utils       | utils.py      | python files for utility or helper functions which are used in more than one places.                          |
| \-          | main.py       | python files which binds everything                                                                           |
