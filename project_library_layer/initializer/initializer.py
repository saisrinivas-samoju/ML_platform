import yaml

# For having general folder structure.

config = yaml.safe_load(open("project_credentials.yaml"))
root_folder = config["root_folder"]
root_file_training_path="mycompany/training/data/project"
root_file_prediction_path="mycompany/prediction/data/project"
root_archive_training_path="mycompany/training/archive/project"
root_archive_prediction_path="mycompany/prediction/archive/project"
root_graph_path="mycompany/report/graph/project"
from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation as MongoDB
mongodb = MongoDB()

def get_watcher_input_file_path(project_id):
    return "mycompany/training/data/project/project_id_{}".format(project_id)


def get_project_id(file_path):
    try:
        reverse_file_path=file_path[-1::-1]
        end_index=reverse_file_path.find("_")
        project_id=reverse_file_path[:end_index]
        if project_id.isdigit():
            project_id=project_id[-1::-1]
            return int(project_id)
        else:
            return False
    except Exception as e:
        raise e

class Initializer():
    def __init__(self):
        pass

    def get_aws_bucket_name(self):
        return root_folder

    def get_azure_container_name(self):
        return root_folder

    def get_google_bucket_name(self):
        return root_folder

    def get_session_secret_key(self):

        data = mongodb.get_record("session", "secretKey")
        return data['secret-key']

    def get_training_database_name(self):
        return "training_system_log"

    def get_prediction_database_name(self):
        return "prediction_system_log"

    def get_project_system_database_name(self):
        return "project_system"

    def get_schema_training_collection_name(self):
        return "schema_training"

    def get_schema_prediction_collection_name(self):
        return "schema_prediction"

    def get_training_data_collection_name(self):
        return "good_raw_data"

    def get_prediction_data_collection_name(self):
        return "good_raw_data"

    def get_column_validation_log_collection_name(self):
        return "column_validation_log"

    def get_data_transform_log_collection_name(self):
        return "data_transform_log"

    def get_db_insert_log_collection_name(self):
        return "db_insert_log"

    def get_export_to_csv_log_collection_name(self):
        return "export_to_csv"

    def get_general_log_collection_name(self):
        return "general_log"

    def get_missing_values_in_column_collection_name(self):
        return "missing_values_in_column"

    def get_model_training_log_collection_name(self):
        return "model_training"

    def get_name_validation_log_collection_name(self):
        return "name_validation_log"

    def get_training_main_log_collection_name(self):
        return "training_main_log"

    def get_prediction_main_log_collection_name(self):
        return "prediction_main_log"

    def get_values_from_schema_validation_collection_name(self):
        return "values_from_schema_validation"

    def get_project_collection_name(self):
        return "project"

    def get_project_configuration_collection_name(self):
        return "project_configuration"

    def get_training_batch_file_path(self,project_id):
        try:
            project_data=mongodb.get_record(self.get_project_system_database_name(),self.get_project_collection_name(),
                                            {'project_id':project_id})

            # If there is not record present in the given project database, and project collection name, raise the exception
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_training_batch_file_path.__name__)
                raise Exception(message)

            project_name = None

            # If project_name key is present in the project_data record, then assign the value of the project_name key to project_name variable
            if 'project_name' in project_data:
                project_name = project_data['project_name']

            # If project_name key is not present in the project_data record
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_training_batch_file_path.__name__)
                raise Exception(message)

            # Create a path for training_batch_files and return it
            path="{}/{}/{}".format(root_file_training_path,project_name,"training_batch_files")

            return path

        except Exception as e:
            raise e

    def get_project_report_graph_path(self,project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_prediction_batch_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_prediction_batch_file_path.__name__)
                raise Exception(message)
            path = "{}/{}".format(root_graph_path, project_name)
            return path
        except Exception as e:
            raise e

    def get_project_report_graph_file_path(self,project_id,execution_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_prediction_batch_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_prediction_batch_file_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_graph_path, project_name,self.get_time_stamp_as_file_name_of_execution_id(execution_id))
            return path
        except Exception as e:
            raise e

    def get_prediction_batch_file_path(self,project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_prediction_batch_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_prediction_batch_file_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_file_prediction_path, project_name, "prediction_batch_files")
            return path
        except Exception as e:
            raise e

    def get_training_good_raw_data_file_path(self,project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_training_good_raw_data_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_training_good_raw_data_file_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_file_training_path, project_name, "good_raw_data_files")
            return path
        except Exception as e:
            raise e

    def get_training_bad_raw_data_file_path(self,project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_training_bad_raw_data_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_training_bad_raw_data_file_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_file_training_path, project_name, "bad_raw_data_files")
            return path
        except Exception as e:
            raise e

    def get_prediction_good_raw_data_file_path(self, project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_prediction_good_raw_data_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_prediction_good_raw_data_file_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_file_prediction_path, project_name, "good_raw_data_files")
            return path
        except Exception as e:
            raise e

    def get_prediction_bad_raw_data_file_path(self, project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_prediction_bad_raw_data_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_prediction_bad_raw_data_file_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_file_prediction_path, project_name, "bad_raw_data_files")
            return path
        except Exception as e:
            raise e

    def get_training_archive_bad_raw_data_file_path(self,project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_training_archive_bad_raw_data_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_training_archive_bad_raw_data_file_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_archive_training_path, project_name, "bad_raw_data_files")
            return path
        except Exception as e:
            raise e

    def get_prediction_archive_bad_raw_data_file_path(self,project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_prediction_archive_bad_raw_data_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_prediction_archive_bad_raw_data_file_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_archive_prediction_path, project_name, "bad_raw_data_files")
            return path
        except Exception as e:
            raise e

    def get_training_good_raw_data_collection_name(self,project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_training_good_raw_data_collection_name.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_training_good_raw_data_collection_name.__name__)
                raise Exception(message)

            collection_name="{}_{}_{}".format(project_name,"good_raw_data",project_id)
            return collection_name
        except Exception as e:
            raise e

    def get_prediction_good_raw_data_collection_name(self,project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_prediction_good_raw_data_collection_name.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_prediction_good_raw_data_collection_name.__name__)
                raise Exception(message)

            collection_name="{}_{}_{}".format(project_name,"good_raw_data",project_id)
            return collection_name
        except Exception as e:
            raise e

    def get_training_file_from_db_path(self,project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message='Project not found. failed in initializer.py method {}'.format(
                    self.get_training_file_from_db_path.__name__)
                raise Exception(message)
            project_name=None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message='Project name not found failed in initializer.py method {}'.format(
                    self.get_training_file_from_db_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_file_training_path, project_name, "training_file_from_db")
            return path
        except Exception as e:
            raise e

    def get_prediction_file_from_db_path(self,project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message='Project not found. failed in initializer.py method {}'.format(
                    self.get_prediction_file_from_db_path.__name__)
                raise Exception(message)
            project_name=None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message='Project name not found failed in initializer.py method {}'.format(
                    self.get_prediction_file_from_db_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_file_prediction_path, project_name, "prediction_file_from_db")
            return path
        except Exception as e:
            raise e

    def get_training_input_file_name(self):
        try:
            return "InputFile.csv"
        except Exception as e:
            raise e

    def get_prediction_input_file_name(self):
        try:
            return "InputFile.csv"
        except Exception as e:
            raise e

    def get_encoder_pickle_file_path(self,project_id):
        try:
            project_data=mongodb.get_record(self.get_project_system_database_name(),self.get_project_collection_name(),{'project_id':project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_training_batch_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_training_batch_file_path.__name__)
                raise Exception(message)
            path="{}/{}/{}".format(root_file_training_path,project_name,"EncoderPickle")
            return path
        except Exception as e:
            raise e

    def get_encoder_pickle_file_name(self):
        return "encoder.pickle"

    def get_training_preprocessing_data_path(self,project_id):
        try:
            project_data=mongodb.get_record(self.get_project_system_database_name(),self.get_project_collection_name(),{'project_id':project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_training_batch_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_training_batch_file_path.__name__)
                raise Exception(message)
            path="{}/{}/{}".format(root_file_training_path,project_name,"preprocessing_data")
            return path
        except Exception as e:
            raise e

    def get_null_value_csv_file_name(self):
        return "null_values.csv"

    def get_model_directory_path(self,project_id):
        try:
            project_data=mongodb.get_record(self.get_project_system_database_name(),self.get_project_collection_name(),{'project_id':project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_training_batch_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_training_batch_file_path.__name__)
                raise Exception(message)
            path="{}/{}/{}".format(root_file_training_path,project_name,"model")
            return path
        except Exception as e:
            raise e

    def get_model_directory_archive_path(self,project_id):
        try:
            project_data=mongodb.get_record(self.get_project_system_database_name(),self.get_project_collection_name(),{'project_id':project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_training_batch_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_training_batch_file_path.__name__)
                raise Exception(message)
            path="{}/{}/{}".format(root_archive_training_path,project_name,"model")
            return path
        except Exception as e:
            raise e

    def get_kmean_folder_name(self):
        try:
            return "KMeans"
        except Exception as e:
            raise e

    def get_prediction_output_file_path(self,project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(),
                                              self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_prediction_output_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_prediction_output_file_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_file_prediction_path, project_name, "prediction_output_file")
            return path
        except Exception as e:
            raise e

    def get_prediction_output_file_name(self):
        return "Output.csv"

    def get_training_thread_database_name(self):
        return "training_prediction_thread"

    def get_prediction_thread_database_name(self):
        return "training_prediction_thread"

    def get_thread_status_collection_name(self):
        return "thread_status"

    def get_add_quotes_to_string_values_in_column_collection_name(self):
        return "add_quotes_to_string_values_in_column"

    def get_encoded_column_name_file_path(self, project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(),
                                              self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_prediction_output_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_prediction_output_file_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_file_training_path, project_name, "encoder_column_name")
            return path
        except Exception as e:
            raise e

    def get_encoded_column_file_name(self):
        return "data_input_column_name.csv"

    def get_accuracy_metric_database_name(self):
        return "accuracy_metric"

    def get_accuracy_metric_collection_name(self):
        return "accuracy_metric_model_collection"

    def get_time_stamp_as_file_name_of_execution_id(self,execution_id):
        try:
            result=mongodb.get_record(self.get_training_thread_database_name(),
                                     self.get_thread_status_collection_name(),
                                     {'execution_id':execution_id})
            if result is not None:
                start_date=result.get('start_date',None)
                start_date='' if start_date is None else start_date.__str__().replace("-","_")
                start_time=result.get('start_time',None)
                start_time='' if start_date is None else start_time.__str__().replace(":","_")
                file_name="{}_{}".format(start_date,start_time)
                if len(file_name)==1:
                    file_name=execution_id
                return file_name
            return execution_id
        except Exception as e:
            raise e

    def get_scheduler_database_name(self):
        return "schedulers"

    def get_scheduler_collection_name(self):
        return "schedulers_job"
