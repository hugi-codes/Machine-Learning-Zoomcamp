### Overview
This folder contains the homework for [Module 5 of the Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/05-deployment) and the necessary scripts / files to solve the questions. The questions and answers can be found in the file [Module_5_Homework_and_Answers.md](https://github.com/hugi-codes/Machine-Learning-Zoomcamp/blob/main/Homework/Module_5_Homework/Module_5_Homework_and_Answers.md)

The topic of this module is the deployment of Machine Learning models. Below is a brief overview of which files in this folder relate to particular exercises of the homework. 

### Note on Exercise 1 and 2
Answers were obtained via command line statements only, so there are no files in this folder that relate to these exercises. More info can be found in [Module_5_Homework_and_Answers.md](https://github.com/hugi-codes/Machine-Learning-Zoomcamp/blob/main/Homework/Module_5_Homework/Module_5_Homework_and_Answers.md). 

### Note on Exercise 3
The goal here is to load an already-trained Machine Learning model and use it to make predictions. The model and the dict vectoriser (the latter is used to created the feature matrix by performing one-hot-encoding) are downloaded to local.

Necessary files for this exercise:
* **model1.bin**: the trained model to be used for inference 
* **dv.bin**: the dict vectoriser which is applied to the model input before inference
* **m5_homework.py**: contains code to load the model and the dict vectoriser from local and to insert a new observation into the trained model to obtain an output (a prediction). 

### Note on Exercise 4
Expanding on exercise 3, the goal here is to serve the model as a web service, using Flask. This means we run the web service (locally) and communicate with it to obtain a prediction from the model: We make a POST request containing the client we want to score, to the web service with which the model is hosted. In **flask_web_service.py** we load the model and the dict vectoriser from local, just like in exercise 3. 

To run the Flask app (the web service), the first step is to run **flask_web_service.py** by executing `python flask_web_service.py` from the command line (need to be in the directory in which the relevant files are stored). In a new terminal window, we can now `run python score_client.py` which contains the code to make the POST request to the Flask app.

Necessary files for this exercise:
* **flask_web_service.py**: contains the code to load the model and the dict vectoriser from local and create the Flask app which will listen to requests on port=5000
* **score_client.py**: contains code to make the POST request and receive the response
* **model1.bin**: the trained model to be used for inference 
* **dv.bin**: the dict vectoriser which is applied to the model input before inference

### Note on Exercise 5
None of the files stored in this repo folder are related to this particular exercise. The goal for this exercise is to download a Docker image from Docker Hub.
The command used to do this was:  `docker pull svizor/zoomcamp-model:3.11.5-slim`

To see the files stored inside the image (to which we have access if we run this image as a container) one can execute `docker run -it svizor/zoomcamp-model:3.11.5-slim /bin/sh` and then ls. The files stored inside the container are model2.bin and dv.bin (inside the app directory).

### Note on Exercise 6
The goal here is to run the Flask app as a Docker container (locally). The first step was to create a Dockerfile. I added a further layer on top of the base image 
svizor/zoomcamp-model:3.11.5-slim which already contains model2.bin and dv.bin (see section “Docker” in [Module_5_Homework_and_Answers.md](https://github.com/hugi-codes/Machine-Learning-Zoomcamp/blob/main/Homework/Module_5_Homework/Module_5_Homework_and_Answers.md)). 

The Dockerfile in this folder builds a production-ready container for a Flask web service, configured to run via Gunicorn on port 9696. It uses Pipenv for dependency management and is built on a lightweight Python image to optimise performance and size.

Because we do not load the model or the dict vectoriser from local anymore (unlike in exercise 4) but from inside the docker container, I created docker_flask_web_service.py, in which the file path for loading the model and the dict vectoriser is adjusted accordingly. In addition, I created docker_score_client.py to score a different client (different POST request to the Flask App than in exercise 4) and adjusted the port to point to the Flask App inside the Docker container (port 9696). 

To obtain the solution for this exercise:
--> need to cd to the directory which contains the necessary files
* build the image: `docker build -t m5-flask-app .`
* run the container: `docker run -p 9696:9696 m5-flask-app`
* make POST request to the Flask app (from new terminal): `python docker_score_client.py`

Necessary files for this exercise:
* **Dockerfile**: contains the Flask app and the installs the necessary required to run it
* **Pipfile.lock**: serves dependency tracking
* **Pipfile**: serves dependency tracking
* **docker_flask_web_service.py**: contains the code of the Flask App 
* **docker_score_client.py**: contains code to score a particular client (via POST request to the Flask app which runs inside the Docker container)