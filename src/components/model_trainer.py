import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
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

from src.utils import save_object,evaluate_models
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

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
                "Linear Regressor": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),  # Fixed name to match params
                "XGBRegressor": XGBRegressor(),  # Changed from XGBClassifier
                "CatBoost Regressor": CatBoostRegressor(verbose=False),  # Changed name
                "AdaBoost Regressor": AdaBoostRegressor()  # Changed name
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regressor": {},
                "K-Neighbors Regressor": {
                    'n_neighbors': [5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree']
                },
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoost Regressor": {
                    'depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            model_report = evaluate_models(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test, 
                models=models, 
                param=params
            )

            if not model_report:
                raise CustomException("No models were evaluated successfully", sys)

            # To get the best model score from dict
            best_model_score = max(model_report.values())

            # To get best model name from dict
            best_model_name = [k for k, v in model_report.items() if v == best_model_score][0]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found with score above 0.6', sys)
            
            logging.info(f"Best found model: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)