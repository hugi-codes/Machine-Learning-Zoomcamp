## Project Overview
Cardiovascular diseases (CVDs) are the number one cause of death globally, with heart attacks and strokes contributing to most of these fatalities. The Heart Failure Prediction Dataset provides clinical data to predict the likelihood of heart failure, offering an opportunity for early detection and better management of the disease.

This project aims to deploy a machine learning classification model with Flask using Docker, where users can input clinical features and get predictions regarding the presence of heart disease. More info on the project requirements can be found [here](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/projects).



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
- **Oldpeak:** Depression in ST [Numeric value]
- **ST_Slope:** The slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
- **HeartDisease:** Output class [1: heart disease, 0: Normal]


## Files
- **notebook.ipynb:** contains code for EDA, cleaning and model training
- **training.py:** contains code for model training
- **prediction.py:** contains code for loading the model and serving it via Flask
- **score_patient.py**: contains code to make POST request to the Flask app and obtain the model prediction
- **Best_Model_random_forest.pkl**: best model saved as pkl file (generated in train.py)
- **Dict_Vectorizer.pkl**: dict vectorizer for one-hot-encoding saved as pkl file (generated in train.py)
- **MinMax_scaler.pkl**: min max scaler for normalisation saved as pkl file (generated in train.py)
- **Dockerfile:** contains instructions to build the docker image, which can be used to run the Flask app as a Docker container
- **docker_prediction.py** contains the Flask app, loads pkl files from inside container instead from local
- **docker_score_patient.py** contains code to make POST request to the Flask app running as a docker container
- **pyproject.toml:** configures the Python project and its dependencies using Poetry, specifying required packages and metadata
- **poetry.lock:** locks the project’s dependencies to specific versions to ensure consistency across environments


## Setup and Running the Flask App with Docker

### 1. Install Poetry

Poetry is a dependency management tool for Python, and it is used to manage the dependencies specified in the `pyproject.toml` file.

To install Poetry, follow these steps:

- **On macOS/Linux**:  
  Open a terminal and run the following command to install Poetry globally:

  ```bash
  curl -sSL https://install.python-poetry.org | python3 -

### 2. Activating the Virtual Environment

Once Poetry is installed, you can activate the virtual environment where dependencies specified in `pyproject.toml` are installed.

1. Navigate to the project directory, where `pyproject.toml` is located.

2. Run the following command to install the dependencies:

    ```bash
    poetry install
    ```

3. After the dependencies are installed, you can activate the virtual environment by running:

    ```bash
    poetry shell
    ```

This will activate the environment and allow you to run your application with the dependencies specified in `pyproject.toml`.

### 3. Building the Docker Image

After setting up Poetry, you can build the Docker image to run the Flask app inside a container.

1. Navigate to the project directory, where your `Dockerfile` is located.

2. To build the Docker image, run:

    ```bash
    docker build -t flask_app .
    ```

This command will create a Docker image with the name `flask_app`.

### 4. Running the Docker Container

Once the Docker image is built, you can run the Flask app in a Docker container.

1. Run the following command to start the container and expose it on port `5000`:

    ```bash
    docker run -p 5000:5000 flask_app
    ```

This will start the Flask app, and it will be accessible at `http://localhost:5000`.

### 5. Interacting with the Running Flask App

Once the Flask app is running inside the Docker container, you can interact with it. Here’s how you can make a request to the app from another terminal window:

1. Open a new terminal window (don't close the one running the Docker container).

2. Navigate to the project directory where `docker_score_patient.py` is located.

3. Run the following command to execute `docker_score_patient.py`, which will make a POST request to the Flask app running in the container:

    ```bash
    python docker_score_patient.py
    ```

This script will send a POST request to the Flask app running inside the container and should receive a response. You can modify `docker_score_patient.py` to pass the necessary data for your specific use case (e.g., patient data).
