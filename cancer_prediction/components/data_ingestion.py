import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from cancer_prediction.entity.config_entity import DataIngestionConfig
from cancer_prediction.entity.artifact_entity import DataIngestionArtifact
from cancer_prediction.exception import CancerPredictionException
from cancer_prediction.logger import logging
from cancer_prediction.data_access.cancer_prediction_data import Cancer_Prediction_Data

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CancerPredictionException(e,sys)

    def export_data_into_feature_store(self)->DataIngestionArtifact:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from mongodb to csv file
        
        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info(f"Exporting data from mongodb")
            CancerPredictionData= Cancer_Prediction_Data()
            trainDataframe = CancerPredictionData.export_collection_as_dataframe(collection_name=
                                                                   self.data_ingestion_config.training_collection_name)
            logging.info(f"Shape of dataframe: {trainDataframe.shape}")


            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)


            trainDataframe.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)


            testDataframe = CancerPredictionData.export_collection_as_dataframe(collection_name=
                                                                   self.data_ingestion_config.testing_collection_name)
              
            testDataframe.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)

            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
            test_file_path=self.data_ingestion_config.testing_file_path)
            
         
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
           

        except Exception as e:
            raise CancerPredictionException(e,sys)

   

    def initiate_data_ingestion(self) ->DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline 
        
        Output      :   train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            data_ingestion_artifact = self.export_data_into_feature_store()

            logging.info("Got the data from mongodb")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            #data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
            #test_file_path=self.data_ingestion_config.testing_file_path)
            
            #logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise CancerPredictionException(e, sys) from e
        