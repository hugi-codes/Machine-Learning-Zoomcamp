## Project Overview
Cardiovascular diseases (CVDs) are the number one cause of death globally, with heart attacks and strokes contributing to most of these fatalities. The Heart Failure Prediction Dataset provides clinical data to predict the likelihood of heart failure, offering an opportunity for early detection and better management of the disease.

This project aims to deploy a machine learning classification model with Flask using Docker, where users can input clinical features and get predictions regarding the presence of heart disease.


## Dataset 
The dataset used in this project contains 11 features related to the clinical conditions of patients that are used to predict heart disease. The dataset has 918 observations and is derived from combining 5 existing heart disease datasets. The dataset can be found on Kaggle under the following [link](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

#### How to download the data
There are several ways to download the data, which are explained on Kaggle (see link above). As an example, one may look into **notebook.ipnyb**  to follow the same approach of loading the data. The approach shown in the notebook requires the ```kagglehub``` library. 

#### Description of the columns
- **Age:** Age of the patient [years]
- **Sex:** Sex of the patient [M: Male, F: Female]
- **ChestPainType:** Chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
- **RestingBP:** Resting blood pressure [mm Hg]
- **Cholesterol:** Serum cholesterol [mm/dl]
- **FastingBS:** Fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
- **RestingECG:** Resting electrocardiogram results [Normal, ST: having ST-T wave abnormality, LVH: showing probable or definite left ventricular hypertrophy]
- **MaxHR:** Maximum heart rate achieved [Numeric value between 60 and 202]
- **ExerciseAngina:** Exercise-induced angina [Y: Yes, N: No]
- **Oldpeak:*: Depression in ST [Numeric value]
- **ST_Slope:** The slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
- **HeartDisease:** Output class [1: heart disease, 0: Normal]


## Files
- **notebook.ipynb:** contains code for EDA, cleaning and model training
- **training.py:** contains code for model training
- **prediction.py:** contains code for loading the model and serving it via Flask
- **Dockerfile:** contains instructions to build the docker image, which can be used to run the Flask app as a Docker container
- **pyproject.toml:** configures the Python project and its dependencies using Poetry, specifying required packages and metadata
- **poetry.lock:** locks the project’s dependencies to specific versions to ensure consistency across environments

## Reproducing the project & Env setup
- How to install dependencies and activate the env
- How to build image, run container