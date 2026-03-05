import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False)
            }

            params = {
                "Random Forest": {
                    "n_estimators": [8, 16, 32],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5, 10]
                },
                "Decision Tree": {
                    "max_depth": [5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "Gradient Boosting": {
                    "n_estimators": [8, 16, 32],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 10]
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2]
                },
                "XGBRegressor": {
                    "n_estimators": [8, 16, 32],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 10]
                },
                "CatBoosting Regressor": {
                    "iterations": [30, 50, 100],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "depth": [4, 6, 8]
                }
            }

            # Evaluate models
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # Pick best model based on test score
            best_model_name = max(model_report, key=lambda k: model_report[k]["test_score"])
            best_model_score = model_report[best_model_name]["test_score"]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            logging.info(
                f"Best found model on both training and testing dataset is {best_model_name} "
                f"with r2 score: {best_model_score}"
            )

            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            print(f"Best model: {best_model_name} with r2 score: {best_model_score}")

        except Exception as e:
            raise CustomException(e, sys)