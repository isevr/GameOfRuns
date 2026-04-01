# Basketball Play-by-Play Run Predictor & Optimizer

This project introduces RunNet, a deep learning model trained on basketball play-by-play data. Its primary goal is to predict and optimize "runs" (scoring streaks or momentum shifts) during a game. 

By utilizing a Convolutional Neural Network (CNN) on sequential event data, the application can classify whether a sequence of plays will lead to a run. Additionally, it features a sequence optimizer that can take a partial sequence of plays and predict the missing events most likely to generate a scoring run.

We provide a ready-to-use API for training and utilizing RunNet, as well as the entire code behind the Game of Runs paper.

---

## 🚀 Features

* **Data Preprocessing & Feature Engineering:** Cleans raw play-by-play data, categorizes shot distances, maps player actions (shooters, rebounders, foulers) to home/away teams, and extracts sequences of plays that constitute a "run."
* **Deep Learning Model:** Utilizes a CNN built with TensorFlow/Keras to classify sequences of 10 events (with 11 features each) into "run" or "no run."
* **Sequence Mining:** Analyzes historical play-by-play data to find the most frequent patterns and sequences of events that successfully lead to scoring runs for specific teams.
* **Play Optimization (Gradient Ascent):** Allows users to input a partial sequence of events. The optimizer uses the trained model's gradients to fill in the missing events that maximize the probability of initiating a run.
* **FastAPI Web Interface:** A lightweight backend with endpoints for uploading data, triggering model training, viewing sequence mining results, and submitting optimization forms.

---

## 📁 Project Structure

```text
├── main.py                        # FastAPI application and route definitions
├── model/
│   ├── encoders/                  # Directory where label encoders are saved
│   ├── model_training.py          # CNN model architecture, training loop, and evaluation plot generation
│   ├── predictions.py             # Standalone script for generating predictions from a CSV
│   ├── preprocessing.py           # Wrapper for data loading and preprocessing execution
│   └── preprocessing_functions.py # Core logic for feature engineering, run extraction, and encoding
├── optimizer/
│   └── optimization.py            # Optimization loop to generate missing plays maximizing 'run' probability
├── sequence_mining/
│   └── sequence_mining.py         # Frequent sequence extraction logic based on team/opponent
├── pretrained_models/             # Directory where trained .keras models are saved
├── preprocessed_data/             # Output directory for processed CSVs
├── uploaded_files/                # Temporary storage for user-uploaded baseline data
├── static/                        # Static assets (training plots, etc.)
└── templates/                     # HTML templates for the FastAPI frontend
```

---

## 🛠️ Tech Stack

* **Backend framework:** FastAPI

* **Machine Learning:** TensorFlow, Keras, Scikit-learn

* **Data Manipulation:** Pandas, NumPy

* **Visualization:** Matplotlib

* **Utilities:** Joblib (for saving encoders)

---

## ⚙️ Setup and Installation

1. Clone the repository and navigate to the directory:

```
git clone <repository-url>
cd <project-directory>
```

2. Create a virtual environment:

```
python -m venv gor
source gor/bin/activate 
```

3. Install required dependencies:

```
pip install -r requirements.txt
```

## 🏃‍♂️ Running the Application

1. Start the FastAPI server:
Use Uvicorn to run the application from the root directory:

```
python -m uvicorn main:app
```

2. Access the Web UI:
Open your browser and navigate to ```http://localhost:8000```.

### Core Endpoints / Workflow:

- ```/```: Landing page.

- ```/upload```: Upload your raw play-by-play CSV file. The backend will process that data at a per-team level.

- ```/model_train```: Trains the RunNet model on the preprocessed data, saves the model, and outputs evaluation plots.

- ```/sequence_mining?team={TEAM_ABBR}```: View the most common play sequences that lead to runs for a specific team.

- ```/optimization_form?team={TEAM_ABBR}```: Input partial play sequences and generate the optimal full sequence to trigger a run.