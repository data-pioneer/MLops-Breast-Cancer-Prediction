import sys

from sklearn.pipeline import Pipeline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from cancer_prediction.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from cancer_prediction.entity.config_entity import DataTransformationConfig
from cancer_prediction.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from cancer_prediction.exception import CancerPredictionException
from cancer_prediction.logger import logging
from cancer_prediction.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CancerPredictionException(e, sys)
        

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CancerPredictionException(e, sys)

    
    def get_data_transformer_object(self) -> Pipeline:
        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns a data transformer object for the data
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )

        try:
            logging.info("Got numerical cols from schema config")

            label_transformer = LabelEncoder()
            oh_transformer = OneHotEncoder()
           

            logging.info("Initialized  OneHotEncoder, LabelEncoder")

            oh_columns = self._schema_config['oh_columns']
            label_columns = self._schema_config['label_columns']
                
            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("LabelEncoder", label_transformer, label_columns)
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )
            return preprocessor

        except Exception as e:
            raise CancerPredictionException(e, sys) from e

    def initiate_data_transformation(self, ) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        
        Output      :   data transformer steps are performed and preprocessor object is created  
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
                logging.info("Starting data transformation")
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)



                #  Find Product of Pollutants
                train_df["N02"]=train_df["N02"]*train_df["Ozone"]*train_df["PM25"]
                test_df["N02"]=test_df["N02"]*test_df["Ozone"]*test_df["PM25"]


                # Drop some features
                for col in train_df.drop(["patient_zip3","N02"],axis=1).columns:
                     train_df["check"]=train_df.groupby(["patient_zip3","N02"])[col].transform("nunique")
                     if train_df["check"].max()==1:
                      # print("dropped ",col)
                       train_df=train_df.drop(col,axis=1)
                       test_df=test_df.drop(col,axis=1)
                train_df=train_df.drop("check",axis=1)


                logging.info("Got train features and test features of Training dataset")

               # define target variable and categorical features
                target = "DiagPeriodL90D"
                cat_cols = list(test_df.columns[test_df.dtypes=="object"])
                cols = list(test_df.drop(["patient_id"],axis=1).columns)
                test_df[target] = np.nan



                logging.info("Added company_age column to the Test dataset")

                # concatenate train and test set
                train_df = pd.concat([train_df,test_df[train_df.columns]],axis=0)
                
                train_df["clust"]=(train_df.metastatic_cancer_diagnosis_code.str.len()==4).astype("int")
                train_df["is_female"] = train_df.breast_cancer_diagnosis_desc.str.contains("female").astype("int")

              

                train_df = test_df.drop(columns=["DiagPeriodL90D"], axis=1)
                print(train_df.columns)
                input_feature_train_arr = preprocessor.fit_transform(train_df)

                logging.info(
                    "Used the preprocessor object to fit transform the train features"
                )

                input_feature_test_arr = preprocessor.transform(test_df)

                logging.info("Used the preprocessor object to transform the test features")
            

                train_arr = np.c_[
                    input_feature_train_arr, np.array(input_feature_train_arr)
                ]

                test_arr = np.c_[
                    input_feature_test_arr, np.array(input_feature_test_arr)
                ]

                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

                logging.info("Saved the preprocessor object")

                logging.info(
                    "Exited initiate_data_transformation method of Data_Transformation class"
                )

               


                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
                return data_transformation_artifact
        except Exception as e:
            raise CancerPredictionException(e, sys) from e