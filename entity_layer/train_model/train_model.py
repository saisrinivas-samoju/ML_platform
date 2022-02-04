import json

from entity_layer.project.project import Project
from entity_layer.project.project_configuration import ProjectConfiguration
from project_library_layer.initializer.initializer import Initializer
from exception_layer.generic_exception.generic_exception import GenericException as TrainModelException
import sys
from project_library_layer.project_training_prediction_mapper.project_training_prediction_mapper import \
    get_training_validation_and_training_model_class_name
from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from entity_layer.email_sender.email_sender import EmailSender


class TrainModel:

    def __init__(self, project_id, executed_by, execution_id, socket_io=None):

        try:
            self.project_id = project_id                  # Id of the project (very important)
            self.executed_by = executed_by                # User who logged in
            self.execution_id = execution_id              # Random id
            self.project_detail = Project()               # by giving project_id to get_project_detail method, we will get the project details
            self.project_config = ProjectConfiguration()  # by giving project_id to get_project_configuration_detail method, we will get the project configuration details
            self.initializer = Initializer()              # Initializer class for absolute paths
            self.socket_io = socket_io                    # socket for broadcasting the logs (Optional use)

        except Exception as e:
            train_model_exception = TrainModelException(
                "Failed during instantiation in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ProjectConfiguration.__name__,
                            self.__init__.__name__))
            raise Exception(train_model_exception.error_message_detail(str(e), sys)) from e

    # Method for training the model
    def training_model(self):
        try:
            # If project_id is None, raise the Exception as training is not possible without project_id
            if self.project_id is None:
                raise Exception("Project id not found")

            # If project_id is not None, get the project_details using project_id
            project_detail = self.project_detail.get_project_detail(project_id=self.project_id)

            # If the status of the project is False, update in the record that the project is failed
            if not project_detail['status']:
                project_detail.update(
                    {'is_failed':True,'message':"Project detail not found",'message_status': 'info', 'project_id': self.project_id})

                # Return the project detail
                return project_detail

            # If the status of the project is True
                # Get the project_cofiguration details
            project_config_detail = self.project_config.get_project_configuration_detail(project_id=self.project_id)
            # If the status in get_project_configuration_detail is False, update it in the database, return the project_config_details and exit
            # This happens when the project configuration details are not provided for the particular project
            if not project_config_detail['status']:
                project_config_detail.update(
                    {'is_failed':True,'message':"Project configuration not found",'message_status': 'info', 'project_id': self.project_id})
                return project_config_detail

            # If project_config_detail key is present inside the project_config_detail record
            if 'project_config_detail' in project_config_detail:
                # reassign project_config_detail variable to the data given for project_cofig_detail key in the record
                project_config_detail = project_config_detail['project_config_detail']

            # If project_config_detail key is present inside the project_config_detail record, return the below response record and exit
            if project_config_detail is None:
                response = {'is_failed':True,'status': False, 'message': 'project configuration not found',
                            'message_status': 'info', 'project_id': self.project_id}
                return response

            # Once, project details and project_cofiguration details are captured successfully. Take the training_batch_files path for the project with the given project_id
            training_file_path = self.initializer.get_training_batch_file_path(project_id=self.project_id)
            cloud_storage = None
            # Capture the cloud storage used for this specific project with the given project_id, from the project_config_detail
            if 'cloud_storage' in project_config_detail:
                cloud_storage = project_config_detail['cloud_storage']

            # If there is no value assigned for cloud_storage key, we won't be having data to load, so the process will be failed.
            if cloud_storage is None:
                result = {'status': False,'is_failed':True,
                          'message': 'Cloud Storage location not found',
                          'message_status': 'info', 'project_id': self.project_id}

                return result

            # Unpacking the tuple which contains training validation class object, and training model class object for the specific project
            TrainingValidation, TrainingModel = get_training_validation_and_training_model_class_name(self.project_id)

            # If the TrainingValidation class is not None
            if TrainingValidation is not None:
                # initialize the class object
                train_validation_object = TrainingValidation(project_id=self.project_id,
                                                             training_file_path=training_file_path,
                                                             executed_by=self.executed_by,
                                                             execution_id=self.execution_id,
                                                             cloud_storage=cloud_storage,
                                                             socket_io=self.socket_io)

                # Execute the training validation process using train_validation method.
                # Methods are common for all the TrainingValidation objects for all the projects
                train_validation_object.train_validation()

                # Once the validation process is completed, initialize the TrainingModel object
                training_model_object = TrainingModel(project_id=self.project_id,
                                                      executed_by=self.executed_by,
                                                      execution_id=self.execution_id,
                                                      cloud_storage=cloud_storage,
                                                      socket_io=self.socket_io)

                # Execute the model training process using training_model method.
                # Methods are common for all the TrainingModel objects for all the projects
                training_model_object.training_model()

                response = {'status': True, 'message': 'Training completed successfully', 'is_failed': False,
                            'message_status': 'info', 'project_id': self.project_id}

            else:
                # This is specific to Sentiment Analysis
                training_data = MongoDBOperation().get_record("sentiment_data_training", "sentiment_input",
                                                              {'execution_id': self.execution_id})
                print(training_data)
                if training_data is None:
                    raise Exception("Training data not found")
                sentiment_user_id = int(training_data['sentiment_user_id'])
                sentiment_data = json.loads(training_data['sentiment_data'])
                sentiment_project_id = int(training_data['sentiment_project_id'])
                train_model = TrainingModel(self.project_id, execution_id=self.execution_id,
                                            executed_by=self.executed_by)

                res = train_model.trainModel(global_project_id=self.project_id,
                                             projectId=sentiment_project_id,
                                             userId=sentiment_user_id,
                                             data=sentiment_data,
                                             )
                if res:
                    response = {'status': True, 'message': 'Training completed successfully', 'is_failed': False,
                                'message_status': 'info', 'project_id': self.project_id}

                else:
                    response = {'status': False, 'message': 'Training Failed',
                                'message_status': 'info', 'project_id': self.project_id, 'is_failed': True, }
            return response
        except Exception as e:
            train_model_exception = TrainModelException(
                "Failed during model training in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ProjectConfiguration.__name__,
                            self.training_model.__name__))

            raise Exception(train_model_exception.error_message_detail(str(e), sys)) from e
