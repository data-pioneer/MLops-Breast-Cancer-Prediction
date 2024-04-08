import sys
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from cancer_prediction.exception import CancerPredictionException
from cancer_prediction.logger import logging
from cancer_prediction.utils.main_utils import load_numpy_array_data, read_yaml_file, load_object, save_object
from cancer_prediction.entity.config_entity import ModelTrainerConfig
from cancer_prediction.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from cancer_prediction.entity.estimator import cancerPredictionModel

from sklearn.linear_model import LogisticRegression
import catboost
import xgboost
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: Configuration for data transformation
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def getMetaModel(self, train: np.array, test: np.array):
        try:
            # print(self.data_transformation_artifact.transformed_train_file_path)
            # df = pd.read_csv(self.data_transformation_artifact.transformed_train_file_path,encoding='utf-8')
            # tdf = pd.read_csv(self.data_transformation_artifact.transformed_test_file_path, encoding='utf-8')
            target = "DiagPeriodL90D"
            df = train
            tdf = test
            tdf=df[df[target].isna()]
            df=df[df[target].notna()]
            len(df),len(tdf)

            modela = CatBoostClassifier(iterations=500, silent=True, learning_rate=0.05, depth=10, eval_metric='AUC', random_seed=42)
            modelb = CatBoostClassifier(iterations=500, silent=True, learning_rate=0.05, depth=10, eval_metric='AUC', random_seed=42)
            model2a = XGBClassifier(
                learning_rate=0.1,
                max_depth=6,
                n_estimators=100,
                subsample=0.9
            )


            # finalize features for training
            drop_cols=["patient_id",target,"patient_zip3","patient_state"]
            cols=list(set(df.columns)-set(drop_cols))

            # Define the number of folds
            num_folds = 10
            from sklearn.model_selection import StratifiedKFold
            kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

            # use stacking method, define meta model
            from sklearn.linear_model import LogisticRegression
            meta_model = LogisticRegression()
            meta_features=['pred1b','pred2','pred1']


            # Initialize empty dataframe to store predictions from all folds
            predictions_from_folds=pd.DataFrame()
            # Iterate over folds
            for fold, (train_index, val_index) in enumerate(kf.split(df, df[target])):
                dfx, efx = df.iloc[train_index], df.iloc[val_index]
                # train and make predictions on train set
                efx["pred1"] = modela.fit(dfx[cols].values, dfx[target]).predict_proba(efx[cols].values)[:,1]
                efx["pred2"] = modelb.fit(dfx[cols].values, dfx[target]).predict_proba(efx[cols].values)[:,1]
                efx["pred1b"] = model2a.fit(dfx[cols].values, dfx[target]).predict_proba(efx[cols].values)[:,1]  
                # make predictions on test set
                tdf["pred1"] = modela.predict_proba(tdf[cols].values)[:,1]
                tdf["pred2"] = modelb.predict_proba(tdf[cols].values)[:,1]
                tdf["pred1b"] = model2a.predict_proba(tdf[cols].values)[:,1]
                # train meta-models and make final predictions 
                tdf["pred"] = meta_model.fit(efx[meta_features], efx[target]).predict_proba(tdf[meta_features])[:, 1]
                predictions_from_folds=pd.concat([predictions_from_folds,tdf],axis=0)
        except Exception as e:
                raise CancerPredictionException(e, sys) from e
        



    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function uses neuro_mf to get the best model object and report of the best model
        
        Output      :   Returns metric artifact object and best model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Using neuro_mf to get best model object and report")
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
            
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

            best_model_detail = model_factory.get_best_model(
                X=x_train,y=y_train,base_accuracy=self.model_trainer_config.expected_accuracy
            )
            model_obj = best_model_detail.best_model

            y_pred = model_obj.predict(x_test)
            
            accuracy = accuracy_score(y_test, y_pred) 
            f1 = f1_score(y_test, y_pred)  
            precision = precision_score(y_test, y_pred)  
            recall = recall_score(y_test, y_pred)
            metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)
            
            return best_model_detail, metric_artifact
        
        except Exception as e:
            raise CancerPredictionException(e, sys) from e
        

    def initiate_model_trainer(self, ) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:

           
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            self.getMetaModel(train=train_arr, test=test_arr)
            # best_model_detail ,metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            
            # preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)


            # if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
            #     logging.info("No best model found with score more than base score")
            #     raise Exception("No best model found with score more than base score")

            # usvisa_model = cancerPredictionModel(preprocessing_object=preprocessing_obj,
            #                            trained_model_object=best_model_detail.best_model)
            # logging.info("Created usvisa model object with preprocessor and model")
            # logging.info("Created best model file path.")
            # save_object(self.model_trainer_config.trained_model_file_path, usvisa_model)

            # model_trainer_artifact = ModelTrainerArtifact(
            #     trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            #     metric_artifact=metric_artifact,
            # )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise CancerPredictionException(e, sys) from e