import sys,os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from metafer_usage.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from metafer_usage.entity.config_entity import DataTransformationConfig
from metafer_usage.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from metafer_usage.logging.logger import logging
from metafer_usage.exception.exception import NetworkSecurityException
from metafer_usage.utils.utils import save_numpy_array_data, save_object
from sklearn.base import BaseEstimator, TransformerMixin


# Create a custom sklearn transformer for the time column. Convert into minutes with consideration for when shift changes are.
class TimeToOperationalMinutes(BaseEstimator, TransformerMixin):
    def __init__(self, start_hour=3):
        self.start_hour = start_hour

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        time = pd.to_datetime(X.iloc[:, 0])

        operational_minutes = (
            time.dt.hour * 60 
            + time.dt.minute 
            - self.start_hour * 60
        ) % (24 * 60)

        return operational_minutes.to_numpy().reshape(-1, 1)
    
class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e

    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def get_data_transformer_object(self)->Pipeline:
        """This function initializes the KNN imputer object with the parameters defined in the training_pipline.py file and returns a Pipeline
        object with the KNNImputer object as the first step
        
        Args:
            cls: DataTransformation
        Returns:
            A Pipeline Object
        """
        logging.info("Entered the get_imputer_object method of DataTransformation class")
        try:
            # Create a column transformer to preprocess the data for clustering
            num_features = ['scans']
            cat_features = ['weekday']
            time_feature = ['time_bin']

            numberic_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()                
            time_transformer = Pipeline(
                steps=[
                ("convert_time", TimeToOperationalMinutes()),
                ("scale_time", StandardScaler())])
            preprocessor = ColumnTransformer(
                [("OneHotEncoder", oh_transformer, cat_features),
                ("Time", time_transformer, time_feature),
                ("StandardScaler", numberic_transformer, num_features)])
            return preprocessor
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e

    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered the initiate_imputation method of DataTransformation class")
        try:
            logging.info("Starting data imputation")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)

            ## Training dataframe
            # Remove the target feature from the independent features and replace -1 with 0 in the target feature
            transform_feature_train_df = train_df.drop(columns=['case_number','capture_date','metafer','date','time','month','weekday_index','shift'],axis=1)

            # Fit Transform Independent features for KNN imputer to replace the missing values and transform the independent features
            # Create a preprocessor object using the get_data_object method and fit it to the training independent features
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(transform_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(transform_feature_train_df)

            # Create np arrays for train and test data by combining the transformed independent features and target features
            train_arr = np.c_[transformed_input_train_feature]

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,train_arr)
            save_object(self.data_transformation_config.transformed_object_file_path,preprocessor_object)

            save_object("final_model/preprocessor.pkl",preprocessor_object)
            
            # Preparing artifacts

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path)

            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e