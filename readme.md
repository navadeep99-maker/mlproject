# ML Project: Student Exam Performance Indicator

## Project Overview
This project is a Machine Learning application designed to predict student performance in exams. It utilizes various regression models and provides a user-friendly interface through a Flask web application.
it is mainly designed to understand working pf mlops pipeline

## Project Structure
The project is organized into the following key components:

*   **`application.py`**: The main Flask application file that handles web routes, user input, and orchestrates predictions.
*   **`src/` directory**: Contains the core Python modules for the ML pipeline:
    *   **`src/components/model_trainer.py`**: This module is responsible for training different regression models (Random Forest, Decision Tree, Gradient Boosting, Linear Regression, K-Neighbors Regressor, XGBRegressor, CatBoosting Regressor) and selecting the best performing model.
    *   **`src/pipeline/predict_pipeline.py`**: This module defines the prediction pipeline, including data preprocessing and making predictions using the trained model.
    *   **`src/exception.py`**: Custom exception handling module.
    *   **`src/logger.py`**: Logging utility for tracking application flow and errors.
    *   **`src/utils.py`**: Utility functions, likely for saving/loading objects (models, preprocessors) and evaluating models.
*   **`templates/` directory**: Contains HTML templates for the web application:
    *   **`home.html`**: The main page for user input and displaying prediction results.
    *   **`index.html`**: A simple index page.
*   **`artifacts/` directory**: Stores serialized models and preprocessed data:
    *   `model.pkl`: The trained machine learning model.
    *   `preprocessor.pkl`: The data preprocessor object.
    *   `raw.csv`, `test.csv`, `train.csv`: Data files used for training and testing.
*   **`notebook/` directory**: Contains Jupyter notebooks for exploratory data analysis and initial model development.
    *   `StudentsPerformance.csv`: The raw dataset used in the project.
*   **`requirements.txt`**: Lists all Python dependencies required to run the project.

## Functionality
1.  **Web Interface**: Users can input student details (gender, race/ethnicity, parental level of education, lunch type, test preparation course, reading score, writing score) through a web form.
2.  **Prediction**: The application uses a trained machine learning model to predict the student's math score based on the provided inputs.
3.  **Model Training**: The `model_trainer.py` component evaluates multiple regression models and selects the best one based on R2 score. The best model and preprocessor are then saved as pickle files.
4.  **Data Preprocessing**: The `predict_pipeline.py` handles the preprocessing of input data before it is fed into the model for prediction.

## How to Run (Local Development)
1.  **Clone the repository** (if not already done):
    ```bash
    git clone https://github.com/navadeep99-maker/mlproject.git
    cd mlproject
    ```
2.  **Create a virtual environment** (recommended):
    ```bash
    # If using conda:
    conda create -n venv python=3.8 -y
    conda activate venv
    # If using venv:
    python -m venv venv
    venv\Scripts\activate   # On Windows
    source venv/bin/activate  # On macOS/Linux
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Flask application**:
    ```bash
    python application.py
    ```
5.  **Access the application**: Open your web browser and go to `http://127.0.0.1:5000` (or the address provided in your terminal).

## Completion Status
The project appears to be functionally complete for its stated purpose as a student exam performance indicator. It includes all necessary components for data handling, model training, prediction, and a web-based user interface. The `artifacts` directory contains a pre-trained model and preprocessor, indicating that the training pipeline has been executed at least once.