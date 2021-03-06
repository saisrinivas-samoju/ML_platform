from threading import Thread
from entity_layer.train_model.train_model import TrainModel, TrainModelException
from project_library_layer.initializer.initializer import Initializer
from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from project_library_layer.datetime_libray.date_time import get_date, get_time
import sys
from logging_layer.logger.log_exception import LogExceptionDetail
from entity_layer.email_sender.email_sender import EmailSender

# Inheriting Thread class in TrainModelThread class and overwriting the run method of the Thread class
class TrainModelThread(Thread):

    def __init__(self, project_id, executed_by, execution_id, socket_io=None, log_writer=None):
        Thread.__init__(self)
        self.project_id = project_id
        self.executed_by = executed_by
        self.execution_id = execution_id
        self.mongo_db = MongoDBOperation()
        self.initialize = Initializer()
        self.training_thread_database_name = self.initialize.get_training_thread_database_name()
        self.thread_status_collection_name = self.initialize.get_thread_status_collection_name()
        self.socket_io = socket_io   # socket_io is used for sending messages to the user everytime
        self.log_writer = log_writer

        if self.socket_io is not None:
            socket_io.emit("started_training",
                           {'message': "We have just instantiated a object of training model thread" + executed_by}
                           , namespace="/training_model")

    def get_max_status_id(self):
        # Get the maximum value in status id, as this will be used for creating id for the next record. So, the status_id for the next record is maximum of status id present + 1
        try:
            max_status_id=None
            if self.mongo_db.is_database_present(self.mongo_db.get_database_client_object(),
                                                 self.training_thread_database_name):
                database_obj = self.mongo_db.create_database(self.mongo_db.get_database_client_object(),
                                                             self.training_thread_database_name)
                if self.mongo_db.is_collection_present(self.thread_status_collection_name, database_obj):
                    max_status_id = self.mongo_db.get_max_value_of_column(self.training_thread_database_name,
                                                                          self.thread_status_collection_name,
                                                                          {'project_id': self.project_id},
                                                                          'status_id')

            return max_status_id
        except Exception as e:
            train_model_exception = TrainModelException(
                "Failed during get_max_status_id in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, TrainModelThread.__name__,
                            self.get_max_status_id.__name__))
            raise Exception(train_model_exception.error_message_detail(str(e), sys)) from e

    def get_running_status_of_training_thread(self):
        try:
            max_status_id=self.get_max_status_id()
            if max_status_id is not None:
                response = self.mongo_db.get_record(self.training_thread_database_name,
                                                    self.thread_status_collection_name,
                                                    {'project_id': self.project_id, 'status_id': max_status_id})

                return response
            else:
                return None
        except Exception as e:
            train_model_exception = TrainModelException(
                "Failed during get_running_status_of_training_thread in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, TrainModelThread.__name__,
                            self.get_running_status_of_training_thread.__name__))
            raise Exception(train_model_exception.error_message_detail(str(e), sys)) from e

    # overwriting the run method of Thread class. Whenever you use
    def run(self):
        record = None
        status_id = None
        try:
            # Assuming no training is already running
            is_training_already_running = False

            max_status_id=self.get_max_status_id()
            # If some status ids are present, then some processes might have happened/ happening.
            if max_status_id is not None:
                # Check the status of training thread in the records, and assign is_running value from the database to is_training_already_running variable.
                response = self.get_running_status_of_training_thread()
                if response is not None:
                    if 'is_running' in response:
                        is_training_already_running = response['is_running']

            # If the training is already running
            if is_training_already_running:

                result = {'status': True, 'message': "Training/prediction is already in progress please wait..."}

                # If the training is already running email will be sent with the execution details. EmailSender class from entity_layer/email_sender.py
                EmailSender().send_email(f"""

                Training/prediction is already in progress. Try again once current operation finished.

                Execution Detail:
                Execution_id: {self.execution_id}
                Executed by: {self.executed_by}
                project_id: {self.project_id}


                """,subject=f"Training notification email of project_id {self.project_id}")

                if self.socket_io is not None:
                    self.socket_io.emit("started_training"+str(self.project_id),
                                        {
                                            'message': result['message']}
                                        , namespace="/training_model")
                self.log_writer.log_stop(result)

                return result

            # If there is no existing records, status_id will start from 1, so 1 is assigned to status_id variable
            if max_status_id is None:
                status_id = 1

            # Else, 1 is added to the maximum value in the existing status ids
            else:
                status_id = max_status_id + 1

            # The above details will be stored inside the database, and an email will be sent
            record = {'project_id': self.project_id,
                      'execution_id': self.execution_id,
                      'executed_by': self.executed_by,
                      'status_id': status_id,
                      'is_running': True,
                      'start_date': get_date(),
                      'start_time': get_time(),
                      'message': 'Training is in  progress ...',
                      'is_Failed': None,           # While training process is happening, is_Failed is None. If the training process is failed, is_Failed=True, if the training process is successful, is_Failed=False.
                      'process_type':'training'
                      }
            EmailSender().send_email(f"""

                            Training is in progress ....

                            Execution Detail:
                            Execution_id:{self.execution_id}
                            Executed by:{self.executed_by}
                            project_id:{self.project_id}


                            """, subject=f"Training notification email of project_id {self.project_id}")

            if self.socket_io is not None:
                self.socket_io.emit("started_training"+str(self.project_id),
                                    {
                                        'message': record['message']}
                                    , namespace="/training_model")

            # The details will be inserted into the database
            self.mongo_db.insert_record_in_collection(self.training_thread_database_name,
                                                      self.thread_status_collection_name,
                                                      record)

            # Creating training model object for training (TrainModel class from entity layer/train_model)
            training_model_obj = TrainModel(project_id=self.project_id, executed_by=self.executed_by,
                                            execution_id=self.execution_id, socket_io=self.socket_io)
            # Training process executed
            training_result = training_model_obj.training_model()

            # Printing the training results
            print(training_result)

            # Updating the record to update it in the database, once the training process is completed
            record.update({
                'is_running': False,
                'is_Failed': training_result['is_failed'],
                'message': str(training_result['message']),
                'stop_time': get_time(),
                'stop_date': get_date()

            })

            if self.socket_io is not None:
                self.socket_io.emit("started_training"+str(self.project_id),
                                    {
                                        'message': record['message']}
                                    , namespace="/training_model")

            # remove MongoDB generated _id key and value
            if '_id' in record:
                record.pop('_id')

            # Update the updated record after the training is completed(with or without failing)
            self.mongo_db.update_record_in_collection(self.training_thread_database_name,
                                                      self.thread_status_collection_name,
                                                      {'status_id': status_id,'project_id':self.project_id}, record)

            # Capture the response from the training result and log it.
            response = {'message': str(training_result['message']), 'status': training_result['status'],
                        'message_status': 'info', 'project_id': self.project_id}

            if self.socket_io is not None:
                self.socket_io.emit("training_completed"+str(self.project_id),
                                    {'message': response['message']}
                                    , namespace="/training_model")

            # Log the captured response from the training_result
            self.log_writer.log_stop(response)

            # Send the details via email
            EmailSender().send_email(f"""

                                        Message: {response['message']}

                                        Execution Detail:
                                        Execution_id:{self.execution_id}
                                        Executed by:{self.executed_by}
                                        project_id:{self.project_id}


                                        """, subject=f"Training notification email of project_id {self.project_id}")
            return response

        except Exception as e:
            EmailSender().send_email(f"""

                                                    Error message: {str(e)}

                                                    Execution Detail:
                                                    Execution_id:{self.execution_id}
                                                    Executed by:{self.executed_by}
                                                    project_id:{self.project_id}


                                                    """,
                                     subject=f"Training notification email of project_id {self.project_id}")
            if record is not None and status_id is not None:
                record.update({
                    'is_running': False,
                    'is_Failed': True,
                    'message': 'Training failed due to :{}'.format(str(e)),
                    'stop_time': get_time(),
                    'stop_date': get_date()

                })
                if '_id' in record:
                    record.pop('_id')
                self.mongo_db.update_record_in_collection(self.training_thread_database_name,
                                                          self.thread_status_collection_name,
                                                          {'status_id': status_id,'project_id':self.project_id}, record)

            train_model_exception = TrainModelException(
                "Failed during model training in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, TrainModelThread.__name__,
                            self.run.__name__))
            if self.socket_io is not None:
                self.socket_io.emit("training_completed"+str(self.project_id),
                                    {'message': train_model_exception.error_message_detail(str(e), sys)}
                                    , namespace="/training_model")

            log_exception = LogExceptionDetail(self.log_writer.executed_by, self.log_writer.execution_id)
            log_exception.log(train_model_exception.error_message_detail(str(e), sys))
