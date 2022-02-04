import uuid
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, ElasticNet, ElasticNetCV
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, roc_curve
import sys

# Exceptions
from exception_layer.generic_exception.generic_exception import GenericException as RandomForestClassificationException
from exception_layer.generic_exception.generic_exception import GenericException as XGBoostClassificationException
from exception_layer.generic_exception.generic_exception import GenericException as ModelFinderException

# Graphs
from plotly_dash.accuracy_graph.accuracy_graph import AccurayGraph
from project_library_layer.initializer.initializer import Initializer


class ModelFinder:

    def __init__(self, project_id, file_object, logger_object):

        try:
            self.project_id = project_id
            self.file_object = file_object
            self.logger_object = logger_object
            self.clf = RandomForestClassifier()
            self.knn = KNeighborsClassifier()
            self.xgb = XGBClassifier(objective='binary:logistic')
            self.sv_classifier = SVC()
            self.gnb = GaussianNB()
            self.linearReg = LinearRegression()
            self.RandomForestReg = RandomForestRegressor()
            self.DecisionTreeReg = DecisionTreeRegressor()
            self.sv_regressor = SVR()
            self.sgd_regression = SGDRegressor()
            self.initailizer = Initializer()
            self.model_name = []
            self.model = []
            self.score = []

        except Exception as e:
            # Using GenericException for inserting the Exception in the database
            model_finder = ModelFinderException(
                "Failed during object instantiation in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.__init__.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def save_accuracy_data(self, model_name, score, execution_model_comparision_id):
        try:
            # Create an instance of AccurayGraph class
            accuracy_graph_data = AccurayGraph(project_id=self.project_id,
                                               model_accuracy_dict={'model_name': model_name,
                                                                    'score': score,
                                                                    'execution_model_comparision': execution_model_comparision_id,
                                                                    'training_execution_id': self.logger_object.execution_id})
            # saving the accuracy data in the mongodb database
            accuracy_graph_data.save_accuracy()

        except Exception as e:
            model_finder = ModelFinderException(
                "save model accuracy [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_model_mushroom.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

######################################################################################################################
################################################### BEST MODEL ON SCORE ##############################################
######################################################################################################################

    def get_best_model_on_score(self, model_name: list, model: list, score: list):

        try:
            # Creating a record with model names, models, and model scores
            record = {'model_name': model_name, 'model': model, 'score': score}
            # Creating dataframe out of it
            df = pd.DataFrame(record)
            # Use the model names as index for the dataframe
            df.index = df.model_name

            # Extract the model_name with the maximum score and assign it to model_name variable
            model_name = df.max()['model_name']
            # Extract the model with maximum score by using the model_name variable
            model = df.loc[model_name]['model']

            # Return the best model_name, and model
            return model_name, model

        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]".format(self.__module__, ModelFinder.__name__,
                                                                  self.get_best_model_on_score.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

######################################################################################################################
############################################### REGRESSION ###########################################################
######################################################################################################################

    def get_best_params_for_ridge_regression(self, train_x, train_y):

        try:
            # Logging
            self.logger_object.log("Entered the get_best_params_for_ridge_regression method in Model Finder Class")

            # Getting alpha values for regression
            alphas = np.random.uniform(low=0, high=10, size=(50,))

            # RidgeCV model creation
            ridge_cv = RidgeCV(alphas=alphas, cv=5, normalize=True)

            # RidgeCV Model training
            ridge_cv.fit(train_x, train_y)

            # Finding the best alpha value for the RidgeCV model
            alpha = ridge_cv.alpha_

            # Creating the ridge regression model with the best alpha value
            ridge_model = Ridge(alpha=alpha)

            # Training the ridge regression model
            ridge_model.fit(train_x, train_y)

            # Logging
            self.logger_object.log(
                'Ridge Regressor best params <alpha value: ' + str(ridge_cv.alpha_) + '>. Exited the '
                'get_best_params_for_ridge_regression method of the Model_Finder class')

            # Returning the best ridge model
            return ridge_model

        except Exception as e:
            model_finder = ModelFinderException(
                "Model Selection Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_ridge_regression.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_support_vector_regressor(self, train_x, train_y):

        try:
            # Logging
            self.logger_object.log("Entered the get_best_params_for_support_vector_regressor method in ModelFinder class")

            # Possible parameter grid
            param_grid = {'C': [0.1, 1, 10, 50, 100, 500], 'gamma': [1, 0.5, 0.1, 0.01, 0.001]}

            # Creating the GridSearchCV model with SVR model for finding the best parameters of the SVR model
            grid = GridSearchCV(SVR(), param_grid, verbose=3, cv=5)

            # Training the Grid model
            grid.fit(train_x, train_y)

            # Finding the best parameters from the Grid model
            C = grid.best_params_['C']
            gamma = grid.best_params_['gamma']

            # Creating the SVR model with the best parameters
            svr_reg = SVR(C=C, gamma=gamma)

            # Training the SVR model
            svr_reg.fit(train_x, train_y)

            # Logging
            self.logger_object.log('Support Vector Regressor best params: ' + str(grid.best_params_) + '. Exited the '
                                   'get_best_params_for_support_vector_regressor method of the Model_Finder class')

            # Returning the best SVR model
            return svr_reg

        except Exception as e:
            model_finder = ModelFinderException(
                "Model Selection Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_support_vector_regressor.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_Random_Forest_Regressor(self, train_x, train_y):

        try:
            # Logging
            self.logger_object.log('Entered the get_best_params_for_Random_Forest_Regressor method of the Model_Finder class')

            # Possible parameter grid
            param_grid_random_forest_tree = {
                "n_estimators": [10, 20, 30],
                "max_features": ["auto", "sqrt", "log2"],
                "min_samples_split": [2, 4, 8],
                "bootstrap": [True, False]}

            # Creating the GridSearchCV model with the base model for finding the best parameters
            grid = GridSearchCV(RandomForestRegressor(), param_grid_random_forest_tree, verbose=3, cv=5)

            # Training the Grid Model
            grid.fit(train_x, train_y)

            # Extracting the best parameters from the Grid Model
            n_estimators = grid.best_params_['n_estimators']
            max_features = grid.best_params_['max_features']
            min_samples_split = grid.best_params_['min_samples_split']
            bootstrap = grid.best_params_['bootstrap']

            # Creating the new model with the best parameters
            random_forest_reg = RandomForestRegressor(n_estimators=n_estimators,
                                                      max_features=max_features,
                                                      min_samples_split=min_samples_split,
                                                      bootstrap=bootstrap)

            # Training the new model
            random_forest_reg.fit(train_x, train_y)

            # Logging
            self.logger_object.log('RandomForestReg best params: ' + str(
                grid.best_params_) + '. Exited the RandomForestReg method of the Model_Finder class')

            # Returning the best model
            return random_forest_reg

        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_Random_Forest_Regressor.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_linearReg(self, train_x, train_y):

        try:
            # Logging
            self.logger_object.log('Entered the get_best_params_for_linearReg method of the Model_Finder class')

            # Possible parameter grid
            param_grid_linear_reg = {
                'fit_intercept': [True, False], 'normalize': [True, False], 'copy_X': [True, False]}

            # Creating the GridSearchCV model with the base model for finding the best parameters
            grid = GridSearchCV(LinearRegression(), param_grid_linear_reg, verbose=3, cv=5)

            # Training the Grid Model
            grid.fit(train_x, train_y)

            # Extracting the best parameters from the Grid Model
            fit_intercept = grid.best_params_['fit_intercept']
            normalize = grid.best_params_['normalize']
            copy_x = grid.best_params_['copy_X']

            # Creating the new model with the best parameters
            lin_reg = LinearRegression(fit_intercept=fit_intercept, normalize=normalize,
                                       copy_X=copy_x)
            # Training the new model
            lin_reg.fit(train_x, train_y)

            # Logging
            self.logger_object.log('LinearRegression best params: ' + str(
                grid.best_params_) + '. Exited the get_best_params_for_linearReg method of the Model_Finder class')

            # Returning the best model
            return lin_reg

        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_linearReg.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e


################################################################################
########################### GET BEST REGRESSION MODEL ##########################
################################################################################

    def get_best_model_for_reg(self, train_x, train_y, test_x, test_y, cluster_no=None):

        try:
            # Logging
            self.logger_object.log('Entered the get_best_model_for_reg method of the Model_Finder class')

            # Generating the title
            title = "Cluster {} ".format(cluster_no) if cluster_no is not None else ''

            ################### Linear Regression Model #######################

            self.model_name.append("Linear_Regression")
            # Creating the LinearRegression model with the best parameters
            linear_reg = self.get_best_params_for_linearReg(train_x, train_y)
            # Predicting the test results
            prediction_linear_reg = linear_reg.predict(test_x)

            # Calculating the performance
            linear_reg_error = r2_score(test_y, prediction_linear_reg)

            # Adding Linear Regression model to the models
            self.model.append(linear_reg)
            # Adding Linear Regression model score to the model scores list
            self.score.append(linear_reg_error)
            # Logging
            self.logger_object.log("Linear Regression r2 score {}".format(linear_reg_error))

            ################### Decision Tree Model #######################

            self.model_name.append('Decision_Tree')
            # Creating the Decision Tree model with the best parameters
            decision_tree_reg = self.get_best_params_for_decision_tree_regressor(train_x, train_y)

            # Predicting the test results
            prediction_decision_tree_reg = decision_tree_reg.predict(test_x)

            # Calculating the performance
            decision_tree_reg_error = r2_score(test_y, prediction_decision_tree_reg)

            # Adding Decision Tree model to the models list
            self.model.append(decision_tree_reg)
            # Adding Decision Tree model score to the model scores list
            self.score.append(decision_tree_reg_error)
            # Logging
            self.logger_object.log("Decision tree regression r2 score {}".format(decision_tree_reg_error))

            ################### XG Boost model #######################

            self.model_name.append('XG_BOOST')
            # Creating the XGBoost model with the best parameters
            xgboost = self.get_best_params_for_xgboost_regressor(train_x, train_y)

            # Predicting the test results
            prediction_xgboost = xgboost.predict(test_x)

            # Calculating the performance
            prediction_xgboost_error = r2_score(test_y, prediction_xgboost)

            # Adding XGBoost model to the models list
            self.model.append(xgboost)
            # Adding XGBoost model score to the model scores list
            self.score.append(prediction_xgboost_error)
            # Logging
            self.logger_object.log("XGBoost regression r2 score {}".format(prediction_xgboost_error))

            ################### Random Forest #######################

            self.model_name.append("Random_Forest")
            # Creating the Random Forest model with the best parameters
            random_forest_reg = self.get_best_params_for_Random_Forest_Regressor(train_x, train_y)

            # Predicting the test results
            prediction_random_forest_reg = random_forest_reg.predict(test_x)

            # Calculating the performance
            prediction_random_forest_error = r2_score(test_y, prediction_random_forest_reg)

            # Adding the Random Forest model to the models list
            self.model.append(random_forest_reg)

            # Adding Random Forest model score to the model scores list
            self.score.append(prediction_random_forest_error)

            # Logging
            self.logger_object.log("Random Forest regression r2 score {}".format(prediction_random_forest_error))

            ################### Support Vector Regressor #######################

            self.model_name.append("SVR")
            # Creating the Support Vector Regressor model with the best parameters
            sv_reg = self.get_best_params_for_support_vector_regressor(train_x, train_y)

            # Predicting the test results
            prediction_sv_reg = sv_reg.predict(test_x)

            # Calculating the performance
            prediction_sv_reg_error = r2_score(test_y, prediction_sv_reg)

            # Adding the SVR model to the models list
            self.model.append(sv_reg)

            # Adding the SVR model's score to the model scores list
            self.score.append(prediction_sv_reg_error)

            # Logging
            self.logger_object.log("Support vector regression r2 score {}".format(prediction_sv_reg_error))

            # Visualization begin based on above model

            prediction_value = [prediction_linear_reg,
                                prediction_decision_tree_reg,
                                prediction_xgboost,
                                prediction_random_forest_reg,
                                prediction_sv_reg]

            # Saving the plots
            for data in zip(self.model_name, prediction_value):
                AccurayGraph().save_scatter_plot(x_axis_data=test_y, y_axis_data=data[1],
                                                 project_id=self.project_id,
                                                 execution_id=self.logger_object.execution_id,
                                                 file_object=self.file_object,
                                                 x_label="True Target values", y_label="Predicted Target value",
                                                 title=title + "Predicted vs True " + data[0])

                AccurayGraph().save_distribution_plot(data=np.abs(test_y - data[1]),
                                                      label="Residual distribution plot",
                                                      project_id=self.project_id,
                                                      execution_id=self.logger_object.execution_id,
                                                      file_object=self.file_object,
                                                      x_label="Error ",
                                                      y_label="frequency or occurance",
                                                      title=title + "{} residual distribution plot".format(data[0]))

            # Mean absolute errors of each model's predictions
            mean_abs_error = []
            for data in prediction_value:
                mean_abs_error.append(np.mean(np.abs(test_y - data)))

            # Saving the accuracy of all the models as a single bar graph for comparison
            AccurayGraph().save_accuracy_bar_graph(
                model_name_list=self.model_name,
                accuracy_score_list=mean_abs_error,
                project_id=self.project_id,
                execution_id=self.logger_object.execution_id,
                file_object=self.file_object,
                x_label="Model List",
                y_label="MAE comparison between {}".format(self.model_name),
                title=title + "Mean Absolute error ")

            # Saving accuracy data based on model on mongo db
            # Creating a unique execution id each time
            execution_model_comparison_id = str(uuid.uuid4())

            for data in zip(self.model_name, self.score):
                # Saving accuracy data in MongoDB database
                self.save_accuracy_data(model_name=data[0], score=data[1],
                                        execution_model_comparision_id=execution_model_comparison_id)
            # Returning the best model based on the score
            return self.get_best_model_on_score(model_name=self.model_name, model=self.model, score=self.score)

        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_model_for_reg.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

# Climate Visiblity

    def get_best_model_climate_visibility(self, train_x, train_y, test_x, test_y, cluster_no=None):

        try:
            # Logging
            self.logger_object.log('Entered the get_best_model_climate_visibility method of the Model_Finder class')

            # Generating the title
            title = "Cluster {} ".format(cluster_no) if cluster_no is not None else ''

            ################### Ridge Regression model #######################

            self.model_name.append('RIDGE_REG')
            # Creating the Ridge Regression model with the best parameters
            ridge_regression = self.get_best_params_for_ridge_regression(train_x, train_y)

            # Predicting the test results
            prediction_ridge_regression = ridge_regression.predict(test_x)

            # Calculating the performance
            prediction_ridge_error = r2_score(test_y, prediction_ridge_regression)

            # Adding Ridge Regression model to the models
            self.model.append(ridge_regression)

            # Adding Ridge Regression model score to the model scores list
            self.score.append(prediction_ridge_error)
            # Logging
            self.logger_object.log("RIDGE_REG regression r2 score {}".format(prediction_ridge_error))

            ################### Decision Tree model #######################

            self.model_name.append('Decision_Tree')

            # Creating the Decision Tree model with the best parameters
            decision_tree_reg = self.get_best_params_for_decision_tree_regressor(train_x, train_y)

            # Predicting the test results
            prediction_decision_tree_reg = decision_tree_reg.predict(test_x)

            # Calculating the performance
            decision_tree_reg_error = r2_score(test_y, prediction_decision_tree_reg)

            # Adding Decision Tree model to the models list
            self.model.append(decision_tree_reg)
            # Adding Decision Tree model score to the model scores list
            self.score.append(decision_tree_reg_error)
            # Logging
            self.logger_object.log("Decision tree regression r2 score {}".format(decision_tree_reg_error))

            ################### XG Boost model #######################

            self.model_name.append('XG_BOOST')
            # Creating the XGBoost model with the best parameters
            xgboost = self.get_best_params_for_xgboost_regressor(train_x, train_y)

            # Predicting the test results
            prediction_xgboost = xgboost.predict(test_x)

            # Calculating the performance
            prediction_xgboost_error = r2_score(test_y, prediction_xgboost)
            # Adding XGBoost model to the models list
            self.model.append(xgboost)
            # Adding XGBoost model score to the model scores list
            self.score.append(prediction_xgboost_error)
            # Logging
            self.logger_object.log("XGBoost regression r2 score {}".format(prediction_xgboost_error))

            ################### Random Forest #######################

            self.model_name.append("Random_Forest")
            # Creating the Random Forest model with the best parameters
            random_forest_reg = self.get_best_params_for_Random_Forest_Regressor(train_x, train_y)

            # Predicting the test results
            prediction_random_forest_reg = random_forest_reg.predict(test_x)

            # Calculating the performance
            prediction_random_forest_error = r2_score(test_y, prediction_random_forest_reg)

            # Adding the Random Forest model to the models list
            self.model.append(random_forest_reg)

            # Adding Random Forest model score to the model scores list
            self.score.append(prediction_random_forest_error)

            # Logging
            self.logger_object.log("Random Forest regression r2 score {}".format(prediction_ridge_error))

            ################### Support Vector Regressor #######################

            self.model_name.append("SVR")

            # Creating the Support Vector Regressor model with the best parameters
            sv_reg = self.get_best_params_for_support_vector_regressor(train_x, train_y)

            # Predicting the test results
            prediction_sv_reg = sv_reg.predict(test_x)

            # Calculating the performance
            prediction_sv_reg_error = r2_score(test_y, prediction_sv_reg)

            # Adding the SVR model to the models list
            self.model.append(sv_reg)

            # Adding the SVR model's score to the model scores list
            self.score.append(prediction_sv_reg_error)

            # Logging
            self.logger_object.log("Support vector regression r2 score {}".format(prediction_ridge_error))


            # Visualization begin based on above model

            prediction_value = [prediction_decision_tree_reg,
                                prediction_xgboost,
                                prediction_ridge_regression,
                                prediction_random_forest_reg,
                                prediction_sv_reg]

            # Saving the plots
            for data in zip(self.model_name, prediction_value):

                AccurayGraph().save_scatter_plot(x_axis_data=test_y, y_axis_data=data[1],
                                                 project_id=self.project_id,
                                                 execution_id=self.logger_object.execution_id,
                                                 file_object=self.file_object,
                                                 x_label="True Target values", y_label="Predicted Target value",
                                                 title=title + "Predicted vs True " + data[0])

                AccurayGraph().save_distribution_plot(data=np.abs(test_y - data[1]),
                                                      label="Residual distribution plot",
                                                      project_id=self.project_id,
                                                      execution_id=self.logger_object.execution_id,
                                                      file_object=self.file_object,
                                                      x_label="Error ",
                                                      y_label="frequency or occurrence",
                                                      title=title + "{} residual distribution plot".format(data[0])
                                                      )
            # Mean absolute errors of each model's predictions
            mean_abs_error = []
            for data in prediction_value:
                mean_abs_error.append(np.mean(np.abs(test_y - data)))

            # Saving the accuracy of all the models as a single bar graph for comparison
            AccurayGraph().save_accuracy_bar_graph(
                model_name_list=self.model_name,
                accuracy_score_list=mean_abs_error,
                project_id=self.project_id,
                execution_id=self.logger_object.execution_id,
                file_object=self.file_object,
                x_label="Model List",
                y_label="MAE comparison between {}".format(self.model_name),
                title=title + "Mean Absolute error "
            )

            # Saving accuracy data based on model on mongo db
            # Creating a unique execution id each time
            execution_model_comparison_id = str(uuid.uuid4())
            for data in zip(self.model_name, self.score):
                # Saving accuracy data in MongoDB database
                self.save_accuracy_data(model_name=data[0], score=data[1],
                                        execution_model_comparision_id=execution_model_comparison_id)
            # Returning the best model based on the score
            return self.get_best_model_on_score(model_name=self.model_name, model=self.model, score=self.score)

        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_model_climate_visibility.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

######################################################################################################################
############################################### CLASSIFICATION #######################################################
######################################################################################################################

    def get_best_params_for_random_forest(self, train_x, train_y):

        try:
            # Logging
            self.logger_object.log('Entered the get_best_params_for_random_forest method of the Model_Finder class')

            # Possible parameter grid
            param_grid = {"n_estimators": [10, 130], "criterion": ['gini', 'entropy'],
                          "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}

            # Creating the GridSearchCV model with Random Forest model for finding the best parameters of the Random forest model
            grid = GridSearchCV(estimator=self.clf, param_grid=param_grid, cv=5, verbose=3)

            # Training the Grid model
            grid.fit(train_x, train_y)

            # Finding the best parameters from the Grid model
            criterion = grid.best_params_['criterion']
            max_depth = grid.best_params_['max_depth']
            max_features = grid.best_params_['max_features']
            n_estimators = grid.best_params_['n_estimators']

            # Creating the Random Forest Classifier model with the best parameters
            self.clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                              max_depth=max_depth, max_features=max_features)

            # Training the new Random Forest with the best parameters found
            self.clf.fit(train_x, train_y)

            # Logging
            self.logger_object.log('Random Forest best params: ' + str(grid.best_params_) + '. Exited the '
                                   'get_best_params_for_random_forest method of the Model_Finder class')

            # Returning the best Random Forest Classifier Model
            return self.clf

        except Exception as e:
            random_clf_exception = RandomForestClassificationException(
                "Random Forest Parameter tuning  failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_random_forest.__name__))
            raise Exception(random_clf_exception.error_message_detail(str(e), sys)) from e

    def get_best_params_for_xgboost(self, train_x, train_y):

        try:
            # Logging
            self.logger_object.log('Entered the get_best_params_for_xgboost method of the Model_Finder class')

            # Possible parameter grid
            param_grid_xgboost = {
                'learning_rate': [0.5, 0.001],
                'max_depth': [20],
                'n_estimators': [10, 200]
            }

            # Creating the GridSearchCV model with XGBoost Classifier model for finding the best parameters of the XGBoost Classifier model
            grid = GridSearchCV(XGBClassifier(objective='binary:logistic'), param_grid_xgboost, verbose=3, cv=5)

            # Training the Grid model
            grid.fit(train_x, train_y)

            # Finding the best parameters from the Grid model
            learning_rate = grid.best_params_['learning_rate']
            max_depth = grid.best_params_['max_depth']
            n_estimators = grid.best_params_['n_estimators']

            # Creating the XGBoost Classifier model with the best parameters
            xgb = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)

            # Training the new XGBoost model with the best parameters found
            xgb.fit(train_x, train_y)

            # Logging
            self.logger_object.log('XGBoost best params: ' + str(grid.best_params_) + '. Exited the '
                                                                                      'get_best_params_for_xgboost method of the Model_Finder class')
            # Returning the best XGBoost Classifier Model
            return xgb

        except Exception as e:
            xg_boost_clf_exception = XGBoostClassificationException(
                "XGBoost Parameter tuning  failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_xgboost.__name__))
            raise Exception(xg_boost_clf_exception.error_message_detail(str(e), sys)) from e

    def get_best_params_for_naive_bayes(self, train_x, train_y):

        try:
            # Logging
            self.logger_object.log('Entered the get_best_params_for_naive_bayes method of the Model_Finder class')

            # Possible parameter grid
            param_grid = {"var_smoothing": [1e-9, 0.1, 0.001, 0.5, 0.05, 0.01, 1e-8, 1e-7, 1e-6, 1e-10, 1e-11]}

            # Creating the GridSearchCV model with Naive_bayes model for finding the best parameters
            grid = GridSearchCV(estimator=self.gnb, param_grid=param_grid, cv=5, verbose=3)

            # Training the Grid model
            grid.fit(train_x, train_y)

            # Finding the best parameters from the Grid model
            var_smoothing = grid.best_params_['var_smoothing']

            # Creating the Naive Bayes model with the best parameters
            gnb = GaussianNB(var_smoothing=var_smoothing)
            # Model Training
            gnb.fit(train_x, train_y)
            # Logging
            self.logger_object.log('Naive Bayes best params: ' + str(
                grid.best_params_) + '. Exited the get_best_params_for_naive_bayes method of the Model_Finder class')
            # Returning the best Naive_bayes Model
            return gnb

        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]".format(self.__module__, ModelFinder.__name__,
                                                                  self.get_best_params_for_naive_bayes.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_svc(self, train_x, train_y):

        try:
            # Logging
            self.logger_object.log('Entered the get_best_params_for_svc method of the Model_Finder class')

            # Possible parameter grid
            param_grid = {"kernel": ['rbf', 'sigmoid'],
                          "C": [0.1, 0.5, 1.0],
                          "random_state": [0, 100, 200, 300]}

            # Creating an object of the Grid Search class
            grid = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=5, verbose=3)

            # Training the Grid Model
            grid.fit(train_x, train_y)

            # Extracting the best parameters from the Grid Model
            kernel = grid.best_params_['kernel']
            C = grid.best_params_['C']
            random_state = grid.best_params_['random_state']

            # Creating the new model with the best parameters
            sv_classifier = SVC(kernel=kernel, C=C, random_state=random_state, probability=True)

            # Training the new model
            sv_classifier.fit(train_x, train_y)

            # Logging
            self.logger_object.log('SVM best params: ' + str(
                grid.best_params_) + '. Exited the get_best_params_for_svm method of the Model_Finder class')

            # Returning the best model
            return sv_classifier

        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_svc.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_KNN(self, train_x, train_y):

        try:
            # Logging
            self.logger_object.log('Entered the get_best_params_for_KNN method of the Model_Finder class')

            # Possible parameter grid
            param_grid_knn = {
                'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [10, 17, 24, 28, 30, 35],
                'n_neighbors': [4, 5, 8, 10, 11],
                'p': [1, 2]
            }

            # Creating an object of the Grid Search class
            grid = GridSearchCV(KNeighborsClassifier(), param_grid_knn, verbose=3,
                                cv=5)

            # Training the Grid model
            grid.fit(train_x, train_y)

            # Finding the best parameters from the Grid model
            algorithm = grid.best_params_['algorithm']
            leaf_size = grid.best_params_['leaf_size']
            n_neighbors = grid.best_params_['n_neighbors']
            p = grid.best_params_['p']

            # Creating a new model with the best parameters
            knn = KNeighborsClassifier(algorithm=algorithm, leaf_size=leaf_size,
                                       n_neighbors=n_neighbors, p=p, n_jobs=-1)

            # Training the new model
            knn.fit(train_x, train_y)

            # Logging
            self.logger_object.log('KNN best params: ' + str(
                grid.best_params_) + '. Exited the KNN method of the Model_Finder class')

            # Returning the best model
            return knn

        except Exception as e:
            model_finder = ModelFinderException(
                "Model Selection Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_KNN.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

# Credit Card Defaults

    def get_best_params_for_naive_bayes_credit_defaulter(self, train_x, train_y):

        try:
            # Logging
            self.logger_object.log('Entered the get_best_params_for_naive_bayes_credit_defaulter method of the Model_Finder class')

            # Possible parameter grid
            param_grid = {"var_smoothing": [1e-9, 0.1, 0.001, 0.5, 0.05, 0.01, 1e-8, 1e-7, 1e-6, 1e-10, 1e-11]}

            # Creating an object of the Grid Search class
            grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid, cv=3, verbose=3)

            # Training the Grid model
            grid.fit(train_x, train_y)

            # Finding the best parameters from the Grid model
            var_smoothing = grid.best_params_['var_smoothing']

            # Creating a new model with the best parameters
            gnb = GaussianNB(var_smoothing=var_smoothing)

            # Training the new model
            gnb.fit(train_x, train_y)

            # Logging
            self.logger_object.log('Naive Bayes best params: ' + str(
                grid.best_params_) + '. Exited the get_best_params_for_naive_bayes method of the Model_Finder class')

            # Returning the best model
            return gnb

        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_naive_bayes_credit_defaulter.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_xgboost_credit_defaulter(self, train_x, train_y):

        try:
            # Logging
            self.logger_object.log('Entered the get_best_params_for_xgboost_credit_defaulter method of the Model_Finder class')

            # Possible parameter grid
            param_grid_xgboost = {

                "n_estimators": [50, 100, 130],
                "max_depth": range(3, 11, 1),
                "random_state": [0, 50, 100]
            }

            # Creating an object of the Grid Search class
            grid = GridSearchCV(XGBClassifier(objective='binary:logistic'), param_grid_xgboost, verbose=3,
                                cv=2, n_jobs=-1)

            # Training the Grid model
            grid.fit(train_x, train_y)

            # Finding the best parameters from the Grid model
            random_state = grid.best_params_['random_state']
            max_depth = grid.best_params_['max_depth']
            n_estimators = grid.best_params_['n_estimators']

            # Creating a new model with the best parameters
            xgb = XGBClassifier(random_state=random_state, max_depth=max_depth,
                                n_estimators=n_estimators, n_jobs=-1)

            # Training the new model
            xgb.fit(train_x, train_y)

            # Logging
            self.logger_object.log('XGBoost best params: ' + str(
                grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')

            # Returning the best model
            return xgb

        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_xgboost_credit_defaulter.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

# Forest Cover

    def get_best_params_for_random_forest_forest_cover_clf(self, train_x, train_y):

        try:
            # Logging
            self.logger_object.log('Entered the get_best_params_for_random_forest_forest_cover_clf method of the Model_Finder class')

            # Possible parameter grid
            param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                          "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}

            # Creating an object of the Grid Search class
            grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, verbose=3, n_jobs=-1)

            # Training the Grid model
            grid.fit(train_x, train_y)

            # Finding the best parameters from the Grid model
            criterion = grid.best_params_['criterion']
            max_depth = grid.best_params_['max_depth']
            max_features = grid.best_params_['max_features']
            n_estimators = grid.best_params_['n_estimators']

            # Creating a new model with the best parameters
            clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                         max_depth=max_depth, max_features=max_features)

            # Training the new model
            clf.fit(train_x, train_y)

            # Logging
            self.logger_object.log('Random Forest best params: ' + str(
                grid.best_params_) + '. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            # Returning the best model
            return clf

        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_random_forest_forest_cover_clf.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_xgboost_forest_cover_clf(self, train_x, train_y):

        try:
            # Logging
            self.logger_object.log('Entered the get_best_params_for_xgboost_forest_cover_clf method of the Model_Finder class')

            # Possible parameter grid
            param_grid_xgboost = {

                'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [10, 50, 100, 200]

            }

            # Creating an object of the Grid Search class
            grid = GridSearchCV(XGBClassifier(objective='multi:softprob'), param_grid_xgboost, verbose=3, cv=5,
                                n_jobs=-1)

            # Training the Grid model
            grid.fit(train_x, train_y)

            # Finding the best parameters from the Grid model
            learning_rate = grid.best_params_['learning_rate']
            max_depth = grid.best_params_['max_depth']
            n_estimators = grid.best_params_['n_estimators']

            # Creating a new model with the best parameters
            xgb = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)

            # Training the new model
            xgb.fit(train_x, train_y)

            # Logging
            self.logger_object.log('XGBoost best params: ' + str(
                grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')

            # Returning the best model
            return xgb

        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_xgboost_forest_cover_clf.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

# Phishing Classifier

    def get_best_params_for_svm_phising_classifier(self, train_x, train_y):

        try:
            # Logging
            self.logger_object.log('Entered the get_best_params_for_svm_phising_classifier method of the Model_Finder class')

            # Possible parameter grid
            param_grid = {"kernel": ['rbf', 'sigmoid'],
                          "C": [0.1, 0.5, 1.0],
                          "random_state": [0, 100, 200, 300]}

            # Creating an object of the Grid Search class
            grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5, verbose=3)

            # Training the Grid model
            grid.fit(train_x, train_y)

            # Finding the best parameters from the Grid model
            kernel = grid.best_params_['kernel']
            c = grid.best_params_['C']
            random_state = grid.best_params_['random_state']

            # Creating a new model with the best parameters
            sv_classifier = SVC(kernel=kernel, C=c, random_state=random_state, probability=True)

            # Training the new model
            sv_classifier.fit(train_x, train_y)

            # Logging
            self.logger_object.log('SVM best params: ' + str(
                grid.best_params_) + '. Exited the get_best_params_for_svm method of the Model_Finder class')

            # Returning the best model
            return sv_classifier

        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_svm_phising_classifier.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

    def get_best_params_for_xgboost_phising_classifier(self, train_x, train_y):

        try:
            # Logging
            self.logger_object.log('Entered the get_best_params_for_xgboost_phising_classifier method of the Model_Finder class')

            # Possible parameter grid
            param_grid_xgboost = {
                "n_estimators": [100, 130], "criterion": ['gini', 'entropy'],
                "max_depth": range(8, 10, 1)
            }

            # Creating an object of the Grid Search class
            grid = GridSearchCV(XGBClassifier(objective='binary:logistic'), param_grid_xgboost, verbose=3,
                                cv=5)

            # Training the Grid model
            grid.fit(train_x, train_y)

            # Finding the best parameters from the Grid model
            criterion = grid.best_params_['criterion']
            max_depth = grid.best_params_['max_depth']
            n_estimators = grid.best_params_['n_estimators']

            # Creating a new model with the best parameters
            xgb = XGBClassifier(criterion=criterion, max_depth=max_depth, n_estimators=n_estimators,
                                n_jobs=-1)

            # Training the new model
            xgb.fit(train_x, train_y)

            # Logging
            self.logger_object.log('XGBoost best params: ' + str(
                grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')

            # Returning the best model
            return xgb

        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_params_for_xgboost_phising_classifier.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

################################################################################
######################### GET BEST CLASSIFICATION MODEL ########################
################################################################################

    def get_best_model(self, train_x, train_y, test_x, test_y, cluster_no=None):
        # Finding the Best model for the given cluster_no

        try:
            # Logging
            self.logger_object.log('Entered the get_best_model method of the Model_Finder class')

            # Generating the title
            if cluster_no is not None:
                # If we have different clusters
                title_generator = " Cluster " + cluster_no + " model {}"
            else:
                title_generator = "Model {}"

            ################### XG Boost model #######################

            self.model_name.append('XG_BOOST')

            # Generating title for XGBoost model
            title = title_generator.format('XG_BOOST')

            # Creating the XGBoost model with the best parameters
            xgboost = self.get_best_params_for_xgboost(train_x, train_y)

            # Predicting the test results
            prediction_xgboost = xgboost.predict(test_x)

            # Calculating performance
            if len(test_y.unique()) == 1:  # If there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                xgboost_score = accuracy_score(test_y, prediction_xgboost)
                self.logger_object.log('Accuracy for XGBoost:' + str(xgboost_score))
            else:
                xgboost_score = roc_auc_score(test_y, prediction_xgboost)        # AUC for XGBoost
                self.logger_object.log('AUC for XGBoost:' + str(xgboost_score))  # Log AUC
                # ROC Curve
                y_score = xgboost.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                # Saving ROC Curve
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            # Adding XGBoost model to the models
            self.model.append(xgboost)
            # Adding XGBoost model score to the model scores list
            self.score.append(xgboost_score)

            ################### Naive bayes #######################

            self.model_name.append('NAIVE_BAYES')
            # Generating title for XGBoost model
            title = title_generator.format('NAIVE_BAYES')
            # Creating Naive Bayes model with the best parameters
            naive_bayes = self.get_best_params_for_naive_bayes(train_x, train_y)
            # Predicting the test results
            prediction_naive_bayes = naive_bayes.predict(test_x)

            # Calculating performance
            if len(test_y.unique()) == 1:  # If there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                naive_bayes_score = accuracy_score(test_y, prediction_naive_bayes)
                self.logger_object.log('Accuracy for naive bayes score' + str(naive_bayes_score))
            else:
                naive_bayes_score = roc_auc_score(test_y, prediction_naive_bayes)
                self.logger_object.log('AUC for naive bayes score:' + str(naive_bayes_score))
                # ROC curve
                y_score = naive_bayes.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[0])
                # Saving ROC Curve
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            # Adding the Naive Bayes model to the Models list
            self.model.append(naive_bayes)

            # Adding Naive Bayes model score to the model scores list
            self.score.append(naive_bayes_score)

            ################### Random Forest #######################

            self.model_name.append('Random_Forest')
            # Generating title
            title = title_generator.format('Random_Forest')
            # Creating model with the best parameters
            random_forest = self.get_best_params_for_random_forest(train_x, train_y)
            # Predicting the test results
            prediction_random_forest = random_forest.predict(test_x)

            # Adding the model to the model list
            self.model.append(random_forest)

            # Calculating performance
            if len(test_y.unique()) == 1: # If there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                random_forest_score = accuracy_score(test_y, prediction_random_forest)
                self.logger_object.log('Accuracy for Random Forest' + str(random_forest_score))
            else:
                random_forest_score = roc_auc_score(test_y, prediction_random_forest)
                self.logger_object.log('AUC for Random Forest' + str(random_forest_score))
                # ROC curve
                y_score = random_forest.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                # Saving ROC Curve
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)
            # Adding model's score to the model scores list
            self.score.append(random_forest_score)

            ################### KNN #######################

            self.model_name.append('KNN')
            # Generating title
            title = title_generator.format('KNN')
            # Creating model with the best parameters
            knn_clf = self.get_best_params_for_KNN(train_x, train_y)
            # Predicting the test results
            prediction_knn = knn_clf.predict(test_x)

            # Adding the model to the model list
            self.model.append(knn_clf)

            # Calculating performance
            if len(test_y.unique()) == 1:
                knn_score = accuracy_score(test_y, prediction_knn)
                self.logger_object.log('Accuracy for KNN clf' + str(knn_score))
            else:
                knn_score = roc_auc_score(test_y, prediction_knn)
                self.logger_object.log('AUC for KNN' + str(knn_score))
                # ROC curve
                y_score = knn_clf.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                # Saving ROC Curve
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)
            # Adding model's score to the model scores list
            self.score.append(knn_score)

            ################### Support Vector Classifier #######################

            # Only if all the output values in the test results are not same, we can use support vector classifier
            if len(test_y.unique()) != 1:
                self.model_name.append("SVC")
                # Generating title
                title = title_generator.format("SVC")

                # Creating model with the best parameters
                svc_clf = self.get_best_params_for_svc(train_x, train_y)

                # Creating model with the best parameters
                prediction_svc = svc_clf.predict(test_x)

                # Adding the model to the model list
                self.model.append(svc_clf)

                # Calculating performance
                if len(test_y.unique()) == 1:
                    svc_score = accuracy_score(test_y, prediction_svc)
                    self.logger_object.log('Accuracy for svc clf' + str(svc_score))
                else:
                    svc_score = roc_auc_score(test_y, prediction_svc)
                    self.logger_object.log('AUC for svc' + str(svc_score))
                    # ROC curve
                    y_score = svc_clf.predict_proba(test_x)[:, 1]
                    fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                    # Saving ROC Curve
                    AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                             execution_id=self.logger_object.execution_id,
                                                                             file_object=self.file_object,
                                                                             title=title)
                # Adding model's score to the model scores list
                self.score.append(svc_score)

            # Saving the accuracy of all the models as a single bar graph for comparison
            AccurayGraph().save_accuracy_bar_graph(
                model_name_list=self.model_name,
                accuracy_score_list=self.score,
                project_id=self.project_id,
                execution_id=self.logger_object.execution_id,
                file_object=self.file_object,
                x_label="Model List",
                y_label="Accuracy score comparison {}".format(self.model_name),
                title="Accuracy Score "
            )

            # Saving accuracy data based on model on mongo db
            # Creating a unique execution id each time
            execution_model_comparison_id = str(uuid.uuid4())

            for data in zip(self.model_name, self.score):
                # Saving accuracy data in MongoDB database
                self.save_accuracy_data(model_name=data[0], score=data[1],
                                        execution_model_comparision_id=execution_model_comparison_id)

            # Returning the best model based on the score
            return self.get_best_model_on_score(model_name=self.model_name, model=self.model, score=self.score)

        except Exception as e:
            model_finder = ModelFinderException(
                "Model Selection Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_model.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

# Credit Card Defaulters

    def get_best_model_credit_deaulter(self, train_x, train_y, test_x, test_y, cluster_no):

        try:
            # Logging
            self.logger_object.log('Entered the get_best_model_credit_deaulter method of the Model_Finder class')

            # Title Generator
            title_generator = " Cluster " + cluster_no + " model {}"

            ################### XG Boost model #######################

            self.model_name.append('XG_BOOST')
            # Generating title for XGBoost model
            title = title_generator.format('XG_BOOST')

            # Creating the XGBoost model with the best parameters
            xgboost = self.get_best_params_for_xgboost_credit_defaulter(train_x, train_y)

            # Predicting the test results
            prediction_xgboost = xgboost.predict(test_x)

            # Calculating performance
            if len(test_y.unique()) == 1:   # If there is only one label in y, then roc_auc_score will return an error. So, we will use accuracy in that case
                xgboost_score = accuracy_score(test_y, prediction_xgboost)
                self.logger_object.log('Accuracy for XGBoost:' + str(xgboost_score))  # Log AUC
            else:
                xgboost_score = roc_auc_score(test_y, prediction_xgboost)        # AUC for XGBoost
                self.logger_object.log('AUC for XGBoost:' + str(xgboost_score))  # Log AUC
                # ROC Curve
                y_score = xgboost.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                # Saving ROC Curve
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)
            # Adding XGBoost model to the models
            self.model.append(xgboost)
            # Adding XGBoost model score to the model scores list
            self.score.append(xgboost_score)

            ################### Naive bayes #######################

            self.model_name.append('NAIVE_BAYES')
            # Generating title for XGBoost model
            title = title_generator.format('NAIVE_BAYES')
            # Creating Naive Bayes model with the best parameters
            naive_bayes = self.get_best_params_for_naive_bayes_credit_defaulter(train_x, train_y)
            # Predicting the test results
            prediction_naive_bayes = naive_bayes.predict(test_x)

            # Calculating performance
            if len(test_y.unique()) == 1:  # If there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                naive_bayes_score = accuracy_score(test_y, prediction_naive_bayes)
                self.logger_object.log('Accuracy for naive bayes score' + str(naive_bayes_score))
            else:
                naive_bayes_score = roc_auc_score(test_y, prediction_naive_bayes)
                self.logger_object.log('AUC for naive bayes score:' + str(naive_bayes_score))
                # ROC curve
                y_score = naive_bayes.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[0])
                # Saving ROC Curve
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)

            # Adding the Naive Bayes model to the Models list
            self.model.append(naive_bayes)

            # Adding Naive Bayes model score to the model scores list
            self.score.append(naive_bayes_score)

            ################### Random Forest #######################

            self.model_name.append('Random_Forest')
            # Generating title
            title = title_generator.format('Random_Forest')
            # Creating model with the best parameters
            random_forest = self.get_best_params_for_random_forest(train_x, train_y)
            # Predicting the test results
            prediction_random_forest = random_forest.predict(test_x)

            # Adding the model to the model list
            self.model.append(random_forest)

            # Calculating performance
            if len(test_y.unique()) == 1:    # If there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                random_forest_score = accuracy_score(test_y, prediction_random_forest)
                self.logger_object.log('Accuracy for Random Forest' + str(random_forest_score))
            else:
                random_forest_score = roc_auc_score(test_y, prediction_random_forest)  # AUC for Random Forest
                self.logger_object.log('AUC for Random Forest' + str(random_forest_score))
                # ROC curve
                y_score = random_forest.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                # Saving ROC Curve
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)
            # Adding model's score to the model scores list
            self.score.append(random_forest_score)

            ################### KNN #######################
            self.model_name.append('KNN')
            # Generating title
            title = title_generator.format('KNN')
            # Creating model with the best parameters
            knn_clf = self.get_best_params_for_KNN(train_x, train_y)
            # Predicting the test results
            prediction_knn = knn_clf.predict(test_x)

            # Adding the model to the model list
            self.model.append(knn_clf)

            # Calculating performance
            if len(test_y.unique()) == 1:
                knn_score = accuracy_score(test_y, prediction_knn)
                self.logger_object.log('Accuracy for KNN clf' + str(knn_score))
            else:
                knn_score = roc_auc_score(test_y, prediction_knn)
                self.logger_object.log('AUC for KNN' + str(knn_score))
                # ROC curve
                y_score = knn_clf.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                # Saving ROC Curve
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)
            # Adding model's score to the model scores list
            self.score.append(knn_score)

            ################### Support Vector Classifier #######################

            # Only if all the output values in the test results are not same, we can use support vector classifier
            if len(test_y.unique()) != 1:
                self.model_name.append("SVC")
                # Generating title
                title = title_generator.format("SVC")

                svc_clf = self.get_best_params_for_svm_phising_classifier(train_x, train_y)

                # Creating model with the best parameters
                prediction_svc = svc_clf.predict(test_x)

                # Adding the model to the model list
                self.model.append(svc_clf)

                # Calculating performance
                if len(test_y.unique()) == 1:
                    svc_score = accuracy_score(test_y, prediction_svc)
                    self.logger_object.log('Accuracy for svc clf' + str(svc_score))
                else:
                    svc_score = roc_auc_score(test_y, prediction_svc)  # AUC for Random Forest
                    self.logger_object.log('AUC for svc' + str(svc_score))
                    # ROC curve
                    y_score = svc_clf.predict_proba(test_x)[:, 1]
                    fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                    # Saving ROC Curve
                    AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                             execution_id=self.logger_object.execution_id,
                                                                             file_object=self.file_object,
                                                                             title=title)
                # Adding model's score to the model scores list
                self.score.append(svc_score)

            # Saving the accuracy of all the models as a single bar graph for comparison
            AccurayGraph().save_accuracy_bar_graph(
                model_name_list=self.model_name,
                accuracy_score_list=self.score,
                project_id=self.project_id,
                execution_id=self.logger_object.execution_id,
                file_object=self.file_object,
                x_label="Model List",
                y_label="Accuracy score comparison {}".format(self.model_name),
                title="Cluster " + str(cluster_no) + "Accuracy Score "
            )

            # Saving accuracy data based on model on mongo db
            # Creating a unique execution id each time
            execution_model_comparison_id = str(uuid.uuid4())

            for data in zip(self.model_name, self.score):
                # Saving accuracy data in MongoDB database
                self.save_accuracy_data(model_name=data[0], score=data[1],
                                        execution_model_comparision_id=execution_model_comparison_id)

            # Returning the best model based on the score
            return self.get_best_model_on_score(model_name=self.model_name, model=self.model, score=self.score)


        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_model_credit_deaulter.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

# Forest Cover Classification

    def get_best_model_forest_cover(self, train_x, train_y, test_x, test_y, cluster_no=None):

        try:
            # Logging
            self.logger_object.log('Entered the get_best_model_forest_cover method of the Model_Finder class')

            # Generating the title
            if cluster_no is not None:
                # If we have different clusters
                title_generator = " Cluster " + cluster_no + " model {}"
            else:
                title_generator = "Model {}"

            ################### XG Boost model #######################

            self.model_name.append('XG_BOOST')

            # Generating title for XGBoost model
            title = title_generator.format('XG_BOOST')

            # Creating the XGBoost model with the best parameters
            xgboost = self.get_best_params_for_xgboost(train_x, train_y)

            # Predicting the test results
            prediction_xgboost = xgboost.predict(test_x)

            # Calculating performance
            if len(test_y.unique()) == 1:
                xgboost_score = accuracy_score(test_y, prediction_xgboost)
                self.logger_object.log('Accuracy for XGBoost:' + str(xgboost_score))  # Log AUC
            else:
                xgboost_score = roc_auc_score(test_y, y_scores, multi_class='ovr')  # AUC for XGBoost
                self.logger_object.log('AUC for XGBoost:' + str(xgboost_score))  # Log AUC
                # ROC Curve
                y_scores = xgboost.predict_proba(test_x)
                # Saving ROC Curve
                AccurayGraph().save_plot_multiclass_roc_curve(test_y, y_scores, xgboost,
                                                              project_id=self.project_id,
                                                              execution_id=self.logger_object.execution_id,
                                                              file_object=self.file_object,
                                                              title="XGBoost ROC curve"
                                                              )

            # Adding XGBoost model to the models
            self.model.append(xgboost)
            # Adding XGBoost model score to the model scores list
            self.score.append(xgboost_score)

            ################### Naive bayes #######################

            self.model_name.append('NAIVE_BAYES')
            # Generating title for XGBoost model
            title = title_generator.format('NAIVE_BAYES')
            # Creating Naive Bayes model with the best parameters
            naive_bayes = self.get_best_params_for_naive_bayes(train_x, train_y)
            # Predicting the test results
            prediction_naive_bayes = naive_bayes.predict(test_x)

            # Calculating performance
            if len(test_y.unique()) == 1:
                naive_bayes_score = accuracy_score(test_y, prediction_naive_bayes)
                self.logger_object.log('Accuracy for naive bayes score' + str(naive_bayes_score))
            else:
                naive_bayes_score = roc_auc_score(test_y, y_scores, multi_class='ovr')
                self.logger_object.log('AUC for naive bayes score:' + str(naive_bayes_score))
                # ROC curve
                y_scores = naive_bayes.predict_proba(test_x)
                # Saving ROC Curve
                AccurayGraph().save_plot_multiclass_roc_curve(test_y, y_scores, naive_bayes,
                                                              project_id=self.project_id,
                                                              execution_id=self.logger_object.execution_id,
                                                              file_object=self.file_object,
                                                              title=title + self.model_name[-1])

            # Adding the Naive Bayes model to the Models list
            self.model.append(naive_bayes)

            # Adding Naive Bayes model score to the model scores list
            self.score.append(naive_bayes_score)

            ################### Random Forest #######################

            self.model_name.append('Random_Forest')
            # Generating title
            title = title_generator.format('Random_Forest')
            # Creating model with the best parameters
            random_forest = self.get_best_params_for_random_forest(train_x, train_y)
            # Predicting the test results
            prediction_random_forest = random_forest.predict(test_x)

            # Adding the model to the model list
            self.model.append(random_forest)

            # Calculating performance
            if len(test_y.unique()) == 1:
                random_forest_score = accuracy_score(test_y, prediction_random_forest)
                self.logger_object.log('Accuracy for Random Forest' + str(random_forest_score))
            else:
                random_forest_score = roc_auc_score(test_y, y_scores, multi_class='ovr')
                self.logger_object.log('AUC for Random Forest' + str(random_forest_score))
                # ROC curve
                y_scores = random_forest.predict_proba(test_x)
                # Saving ROC Curve
                AccurayGraph().save_plot_multiclass_roc_curve(test_y, y_scores, random_forest,
                                                              project_id=self.project_id,
                                                              execution_id=self.logger_object.execution_id,
                                                              file_object=self.file_object,
                                                              title=title + self.model_name[-1])

            # Adding model's score to the model scores list
            self.score.append(random_forest_score)

            ################### KNN #######################

            self.model_name.append('KNN')

            # Generating title
            title = title_generator.format('KNN')

            # Creating model with the best parameters
            knn_clf = self.get_best_params_for_KNN(train_x, train_y)

            # Predicting the test results
            prediction_knn = knn_clf.predict(test_x)

            # Adding the model to the model list
            self.model.append(knn_clf)

            # Calculating performance
            if len(test_y.unique()) == 1:
                knn_score = accuracy_score(test_y, prediction_knn)
                self.logger_object.log('Accuracy for KNN clf' + str(knn_score))
            else:
                knn_score = roc_auc_score(test_y, y_scores, multi_class='ovr')  # AUC for Random Forest
                self.logger_object.log('AUC for KNN' + str(knn_score))
                # ROC curve
                y_scores = knn_clf.predict_proba(test_x)
                # Saving ROC Curve
                AccurayGraph().save_plot_multiclass_roc_curve(test_y, y_scores, knn_clf,
                                                              project_id=self.project_id,
                                                              execution_id=self.logger_object.execution_id,
                                                              file_object=self.file_object,
                                                              title=title + self.model_name[-1])

            # Adding model's score to the model scores list
            self.score.append(knn_score)

            ################### Support Vector Classifier #######################

            # Only if all the output values in the test results are not same, we can use support vector classifier
            if len(test_y.unique()) != 1:
                self.model_name.append("SVC")
                # Generating title
                title = title_generator.format("SVC")

                # Creating model with the best parameters
                svc_clf = self.get_best_params_for_svc(train_x, train_y)

                # Creating model with the best parameters
                prediction_svc = svc_clf.predict(test_x)

                # Adding the model to the model list
                self.model.append(svc_clf)

                # Calculating performance
                if len(test_y.unique()) == 1:
                    svc_score = accuracy_score(test_y, prediction_svc)
                    self.logger_object.log('Accuracy for svc clf' + str(svc_score))
                else:
                    svc_score = roc_auc_score(test_y, y_scores, multi_class='ovr')  # AUC for Random Forest
                    self.logger_object.log('AUC for svc' + str(svc_score))
                    # ROC curve
                    y_scores = svc_clf.predict_proba(test_x)
                    # Saving ROC Curve
                    AccurayGraph().save_plot_multiclass_roc_curve(test_y, y_scores, svc_clf,
                                                                  project_id=self.project_id,
                                                                  execution_id=self.logger_object.execution_id,
                                                                  file_object=self.file_object,
                                                                  title=title + self.model_name[-1])

                # Adding model's score to the model scores list
                self.score.append(svc_score)

            # Saving the accuracy of all the models as a single bar graph for comparison
            AccurayGraph().save_accuracy_bar_graph(
                model_name_list=self.model_name,
                accuracy_score_list=self.score,
                project_id=self.project_id,
                execution_id=self.logger_object.execution_id,
                file_object=self.file_object,
                x_label="Model List",
                y_label="Accuracy score comparison {}".format(self.model_name),
                title="Accuracy Score "
            )

            # Saving accuracy data based on model on mongo db
            # Creating a unique execution id each time
            execution_model_comparison_id = str(uuid.uuid4())

            for data in zip(self.model_name, self.score):
                # Saving accuracy data in MongoDB database
                self.save_accuracy_data(model_name=data[0], score=data[1],
                                        execution_model_comparision_id=execution_model_comparison_id)

            # Returning the best model based on the score
            return self.get_best_model_on_score(model_name=self.model_name, model=self.model, score=self.score)

        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_model_forest_cover.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

# Phishing Classifier

    def get_best_model_phising_classifier(self, train_x, train_y, test_x, test_y, cluster_no):

        try:
            # Logging
            self.logger_object.log('Entered the get_best_model_phising_classifier method of the Model_Finder class')

            # Generating the title
            title_generator = " Cluster " + cluster_no + " model {}"

            ################### XG Boost model #######################

            self.model_name.append('XG_BOOST')

            # Generating title for XGBoost model
            title = title_generator.format('XG_BOOST')

            # Creating the XGBoost model with the best parameters
            xgboost = self.get_best_params_for_xgboost_phising_classifier(train_x, train_y)
            # Predicting the test results
            prediction_xgboost = xgboost.predict(test_x)  # Predictions using the XGBoost Model

            # Calculating performance
            if len(test_y.unique()) == 1:  # If there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                xgboost_score = accuracy_score(test_y, prediction_xgboost)
                self.logger_object.log('Accuracy for XGBoost:' + str(xgboost_score))  # Log AUC
            else:
                xgboost_score = roc_auc_score(test_y, prediction_xgboost)  # AUC for XGBoost
                self.logger_object.log('AUC for XGBoost:' + str(xgboost_score))  # Log AUC
                # ROC Curve
                y_score = xgboost.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                # Saving ROC Curve
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)
            # Adding XGBoost model to the models
            self.model.append(xgboost)
            # Adding XGBoost model score to the model scores list
            self.score.append(xgboost_score)

            ################### Naive bayes #######################

            self.model_name.append('NAIVE_BAYES')
            # Generating title for XGBoost model
            title = title_generator.format('NAIVE_BAYES')
            # Creating Naive Bayes model with the best parameters
            naive_bayes = self.get_best_params_for_naive_bayes(train_x, train_y)
            # Predicting the test results
            prediction_naive_bayes = naive_bayes.predict(test_x)

            # Calculating performance
            if len(test_y.unique()) == 1:
                naive_bayes_score = accuracy_score(test_y, prediction_naive_bayes)
                self.logger_object.log('Accuracy for naive bayes score' + str(naive_bayes_score))
            else:
                naive_bayes_score = roc_auc_score(test_y, prediction_naive_bayes)
                self.logger_object.log('AUC for naive bayes score:' + str(naive_bayes_score))
                # ROC curve
                y_score = naive_bayes.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[0])
                # Saving ROC Curve
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)
            # Adding the Naive Bayes model to the Models list
            self.model.append(naive_bayes)

            # Adding Naive Bayes model score to the model scores list
            self.score.append(naive_bayes_score)


            ################### Random Forest #######################

            self.model_name.append('Random_Forest')
            # Generating title
            title = title_generator.format('Random_Forest')
            # Creating model with the best parameters
            random_forest = self.get_best_params_for_random_forest(train_x, train_y)
            # Predicting the test results
            prediction_random_forest = random_forest.predict(test_x)

            # Adding the model to the model list
            self.model.append(random_forest)

            # Calculating performance
            if len(test_y.unique()) == 1:
                random_forest_score = accuracy_score(test_y, prediction_random_forest)
                self.logger_object.log('Accuracy for Random Forest' + str(random_forest_score))
            else:
                random_forest_score = roc_auc_score(test_y, prediction_random_forest)
                self.logger_object.log('AUC for Random Forest' + str(random_forest_score))
                # ROC curve
                y_score = random_forest.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                # Saving ROC Curve
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)
            # Adding model's score to the model scores list
            self.score.append(random_forest_score)

            ################### KNN #######################

            self.model_name.append('KNN')
            # Generating title
            title = title_generator.format('KNN')
            # Creating model with the best parameters
            knn_clf = self.get_best_params_for_KNN(train_x, train_y)
            # Predicting the test results
            prediction_knn = knn_clf.predict(test_x)

            # Adding the model to the model list
            self.model.append(knn_clf)

            # Calculating performance
            if len(test_y.unique()) == 1:
                knn_score = accuracy_score(test_y, prediction_knn)
                self.logger_object.log('Accuracy for KNN clf' + str(knn_score))
            else:
                knn_score = roc_auc_score(test_y, prediction_knn)
                self.logger_object.log('AUC for KNN' + str(knn_score))
                # ROC curve
                y_score = knn_clf.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                # Saving ROC Curve
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)
            # Adding model's score to the model scores list
            self.score.append(knn_score)

            ################### Support Vector Classifier #######################

            # Only if all the output values in the test results are not same, we can use support vector classifier
            if len(test_y.unique()) != 1:
                self.model_name.append("SVC")
                # Generating title
                title = title_generator.format("SVC")

                # Creating model with the best parameters
                svc_clf = self.get_best_params_for_svm_phising_classifier(train_x, train_y)

                # Creating model with the best parameters
                prediction_svc = svc_clf.predict(test_x)

                # Adding the model to the model list
                self.model.append(svc_clf)

                # Calculating performance
                if len(test_y.unique()) == 1:
                    svc_score = accuracy_score(test_y, prediction_svc)
                    self.logger_object.log('Accuracy for svc clf' + str(svc_score))
                else:
                    svc_score = roc_auc_score(test_y, prediction_svc)  # AUC for Random Forest
                    self.logger_object.log('AUC for svc' + str(svc_score))
                    # ROC curve
                    y_score = svc_clf.predict_proba(test_x)[:, 1]
                    fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                    # Saving ROC Curve
                    AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                             execution_id=self.logger_object.execution_id,
                                                                             file_object=self.file_object,
                                                                             title=title)
                # Adding model's score to the model scores list
                self.score.append(svc_score)

            # Saving the accuracy of all the models as a single bar graph for comparison
            AccurayGraph().save_accuracy_bar_graph(
                model_name_list=self.model_name,
                accuracy_score_list=self.score,
                project_id=self.project_id,
                execution_id=self.logger_object.execution_id,
                file_object=self.file_object,
                x_label="Model List",
                y_label="Accuracy score comparison {}".format(self.model_name),
                title="Cluster " + str(cluster_no) + "Accuracy Score "
            )

            # Saving accuracy data based on model on mongo db
            # Creating a unique execution id each time
            execution_model_comparison_id = str(uuid.uuid4())

            for data in zip(self.model_name, self.score):
                # Saving accuracy data in MongoDB database
                self.save_accuracy_data(model_name=data[0], score=data[1],
                                        execution_model_comparision_id=execution_model_comparison_id)

            # Returning the best model based on the score
            return self.get_best_model_on_score(model_name=self.model_name, model=self.model, score=self.score)

        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_model_phising_classifier.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e

# Scania Truck

    def get_best_model_scania_truck(self, train_x, train_y, test_x, test_y, cluster_no=None):


        try:
            # Logging
            self.logger_object.log('Entered the get_best_model_scania_truck method of the Model_Finder class')

            # Generating the title
            if cluster_no is not None:
                # If we have different clusters
                title_generator = " Cluster " + cluster_no + " model {}"
            else:
                title_generator = "Model {}"

            ################### XG Boost model #######################

            self.model_name.append('XG_BOOST')

            # Generating title for XGBoost model
            title = title_generator.format('XG_BOOST')

            # Creating the XGBoost model with the best parameters
            xgboost = self.get_best_params_for_xgboost(train_x, train_y)

            # Predicting the test results
            prediction_xgboost = xgboost.predict(test_x)

            # Calculating performance
            if len(test_y.unique()) == 1:
                xgboost_score = accuracy_score(test_y, prediction_xgboost)
                self.logger_object.log('Accuracy for XGBoost:' + str(xgboost_score))  # Log AUC
            else:
                xgboost_score = roc_auc_score(test_y, prediction_xgboost)  # AUC for XGBoost
                self.logger_object.log('AUC for XGBoost:' + str(xgboost_score))  # Log AUC
                # ROC Curve
                y_score = xgboost.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                # Saving ROC Curve
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)
            # Adding XGBoost model to the models
            self.model.append(xgboost)
            # Adding XGBoost model score to the model scores list
            self.score.append(xgboost_score)

            ################### Random Forest #######################

            self.model_name.append('Random_Forest')
            # Generating title
            title = title_generator.format('Random_Forest')
            # Creating model with the best parameters
            random_forest = self.get_best_params_for_random_forest(train_x, train_y)
            # Predicting the test results
            prediction_random_forest = random_forest.predict(test_x)

            # Adding the model to the model list
            self.model.append(random_forest)

            # Calculating performance
            if len(test_y.unique()) == 1:
                random_forest_score = accuracy_score(test_y, prediction_random_forest)
                self.logger_object.log('Accuracy for Random Forest' + str(random_forest_score))
            else:
                random_forest_score = roc_auc_score(test_y, prediction_random_forest)  # AUC for Random Forest
                self.logger_object.log('AUC for Random Forest' + str(random_forest_score))
                # ROC curve
                y_score = random_forest.predict_proba(test_x)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=test_y.unique()[1])
                # Saving ROC Curve
                AccurayGraph().save_roc_curve_plot_binary_classification(fpr, tpr, self.project_id,
                                                                         execution_id=self.logger_object.execution_id,
                                                                         file_object=self.file_object,
                                                                         title=title)
            # Adding model's score to the model scores list
            self.score.append(random_forest_score)

            # Saving the accuracy of all the models as a single bar graph for comparison
            AccurayGraph().save_accuracy_bar_graph(
                model_name_list=self.model_name,
                accuracy_score_list=self.score,
                project_id=self.project_id,
                execution_id=self.logger_object.execution_id,
                file_object=self.file_object,
                x_label="Model List",
                y_label="Accuracy score comparison {}".format(self.model_name),
                title="Accuracy Score "
            )

            # Saving accuracy data based on model on mongo db
            # Creating a unique execution id each time
            execution_model_comparison_id = str(uuid.uuid4())
            for data in zip(self.model_name, self.score):
                # Saving accuracy data in MongoDB database
                self.save_accuracy_data(model_name=data[0], score=data[1],
                                        execution_model_comparision_id=execution_model_comparison_id)
            # Returning the best model based on the score
            return self.get_best_model_on_score(model_name=self.model_name, model=self.model, score=self.score)

        except Exception as e:
            model_finder = ModelFinderException(
                "Failed in [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFinder.__name__,
                            self.get_best_model_forest_cover.__name__))
            raise Exception(model_finder.error_message_detail(str(e), sys)) from e
