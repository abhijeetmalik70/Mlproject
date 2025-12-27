import sys



from sklearn.ensemble import (RandomForestRegressor,
                              AdaBoostRegressor,
                              GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.components.data_transform import DataTransformation
from src.components.data_injestion import DataInjestion


#lets get all the models 
class ModelTrainConfig:
    model_train_config = DataTransformation()

class ModelTrain:
            model_train_object = ModelTrainConfig()
            

            
            #training data 
            def model_train(self,X_train, y_train, X_test, y_test):
                        try:
                                
                                models = {"random_forest" : RandomForestRegressor(),
                                "ada" : AdaBoostRegressor(),
                                "gradient" : GradientBoostingRegressor(),
                                    "linear_regression" : LinearRegression(),
                                    "knn" : KNeighborsRegressor(),
                                    "tree" : DecisionTreeRegressor()
                                   
                             
                                        }



                                

                                #fitting each model on the giving training data and calculting the performance for each model on train and test data
                                prediction_dict = {}
                                for name , model in models.items():
                                    model.fit(X_train,y_train)
                                    y_train_predict = model.predict(X_train)
                                    y_test_predict = model.predict(X_test)
                                    r2_score_train = r2_score(y_train,y_train_predict)
                                    r2_score_test = r2_score(y_test,y_test_predict)
                                    prediction_dict[name] = [r2_score_train,r2_score_test]
                                logging.info("we found out the predictions for all the models")

                                #lets get the best model which has predicted the best out of all the models
                                sorted_dict = dict(sorted(prediction_dict.items(), key=lambda item: item[1][1],reverse=True))
                                best_model = next(iter(sorted_dict))
                                best_prediction = next(iter(sorted_dict.values()))
                                print(f"best_mode = {best_model} and best_predict {best_prediction}")
                                logging.info("printed the best model and its prediction on test data")

                                return (best_model,best_prediction)

                        except Exception as e:
                            raise CustomException(e,sys)





if __name__ == "__main__":
    # Call your function here to see results
    try:
            obj_data_injestion = DataInjestion()
            train_data,test_data = obj_data_injestion.initiate_data_injestion()

            obj_data_transformation = DataTransformation()
            X_train,y_train,X_test,y_test,_ = obj_data_transformation.initialise_data_transformer(train_data,test_data)

            object_model_train = ModelTrain()
            object_model_train.model_train( X_train,y_train,X_test,y_test)
    except Exception as e :
           raise CustomException(e,sys)
           
