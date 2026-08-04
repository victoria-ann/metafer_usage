import os,sys
import mlflow
from metafer_usage.exception.exception import NetworkSecurityException
from metafer_usage.logging.logger import logging
from metafer_usage.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact
from metafer_usage.entity.config_entity import ModelTrainerConfig

from metafer_usage.utils.utils import NetworkModel
from metafer_usage.utils.utils import load_object,save_object
from metafer_usage.utils.utils import load_numpy_array_data

import sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from itertools import product

import dagshub
dagshub.init(repo_owner='victoria.palecek', repo_name='metafer_usage', mlflow=True)


class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e



    def train_model(self, X,
        eps=(0.2, 0.4, 0.6, 0.8, 1.0),
        min_samples=(4, 10, 20, 40, 80, 120),
        noise_weight=1.0,
    ):
        try:
            best_score = -np.inf
            best_params = None
            best_model = None
            best_silhouette = None
            best_noise_fraction = None
            results = []

            for eps_value, min_samples_value in product(eps, min_samples):

                model = DBSCAN(
                    eps=eps_value,
                    min_samples=min_samples_value
                )

                labels = model.fit_predict(X)
                mask = labels != -1

                # Need at least 2 clusters after removing noise
                if np.sum(mask) > 1 and len(np.unique(labels[mask])) > 1:

                    silhouette = silhouette_score(X[mask], labels[mask])
                    noise_fraction = np.mean(labels == -1)
                    score = silhouette - noise_weight * noise_fraction

                    results.append({
                        "eps": eps_value,
                        "min_samples": min_samples_value,
                        "silhouette": silhouette,
                        "noise_fraction": noise_fraction,
                        "score": score,
                        "clusters": len(np.unique(labels[mask]))
                    })

                    if score > best_score:
                        best_score = score
                        best_params = {
                            "eps": eps_value,
                            "min_samples": min_samples_value
                        }
                        best_model = model
                        best_silhouette = silhouette
                        best_noise_fraction = noise_fraction

            results = (
                pd.DataFrame(results)
                .sort_values("score", ascending=False)
                .reset_index(drop=True)
            )

            print (
                best_score,
                best_params,
                best_model,
                best_silhouette,
                best_noise_fraction,
                results
            )

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)

            Network_Model = NetworkModel(preprocessor=preprocessor,model=best_model)
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=Network_Model)

            save_object("final_model/model.pkl",best_model)

            # Create model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact= (best_silhouette,best_noise_fraction,best_score)
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys) from e

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_ath = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading transformed training and testing dataset
            train_array = load_numpy_array_data(train_file_ath)

            model_trainer_artifact = self.train_model(train_array)
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e