from thread_layer.train_model_thread.train_model_thread import TrainModelThread
from thread_layer.predict_from_model_thread.predict_from_model_thread import PredictFromModelThread
from logging_layer.logger.log_exception import LogExceptionDetail
from logging_layer.logger.log_request import LogRequest
from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from project_library_layer.initializer.initializer import Initializer
from entity_layer.scheduler.scheduler_storage import SchedulerStorage

class ScheduleTask:
    def __init__(self,project_id,executed_by,execution_id,socket_io=None):
        self.project_id = project_id      # (project_id of the project)
        self.executed_by = executed_by    # (email_address is taken)
        self.execution_id = execution_id  # (unique string code)
        self.socket_io = socket_io
        self.log_writer = LogRequest(executed_by=self.executed_by,execution_id=self.execution_id)  # For logging the start and end time of the task
        self.train_predict_thread_obj_ref=None   # (thread object will be assigned based on the action)
        self.sch_storage=SchedulerStorage()      # Used for storing the schedule records in the database

    def start_training(self):
        # If the action is training, TrainModelThread class object is used
        try:
            train_predict_thread_obj_ref = TrainModelThread(project_id=self.project_id, executed_by=self.executed_by,
                                                            execution_id=self.execution_id, socket_io=self.socket_io,
                                                            log_writer=self.log_writer)
            response = train_predict_thread_obj_ref.get_running_status_of_training_thread()
            previous_execution_thread_status = None
            if response is not None:
                if 'is_running' in response:
                    previous_execution_thread_status = response['is_running']
            if previous_execution_thread_status:
                pass
                # Socket coce is not being used due to some error in the later steps during deployment

                # self.socket_io.emit("training_completed" + str(self.project_id),
                #      {'message': "Training/ Prediction model thread is already running"},
                #      namespace="/training_model")

            # If previous_execution_thread_status is either None or False, no training is happening. So, we can start the training process.
            if previous_execution_thread_status is None or not previous_execution_thread_status:
                self.sch_storage.update_job_record(job_id=self.execution_id,status='running')
                train_predict_thread_obj_ref.start()

                # self.socket_io.emit("started_training" + str(self.project_id),
                #      {'message': "We have started training and execution started  by {} and execution id  "
                #                  "is {}".format(self.executed_by,
                #                                 self.execution_id)},
                #      namespace="/training_model")

                train_predict_thread_obj_ref.join()
                self.sch_storage.update_job_record(job_id=self.execution_id,status='successful')

        except Exception as e:
            self.sch_storage.update_job_record(job_id=self.execution_id, status='failed')
            if self.log_writer is not None:
                self.log_writer.log_stop({'status': False, 'error_message': str(e)})
            log_exception = LogExceptionDetail(self.log_writer.executed_by, self.log_writer.execution_id)
            log_exception.log(str(e))

            # self.socket_io.emit("started_training" + str(self.project_id), {'message': str(e)}, namespace="/training_model")


    def start_prediction(self):
        # Similar to the above training method
        try:
            train_predict_thread_obj_ref = PredictFromModelThread(project_id=self.project_id, executed_by=self.executed_by,
                                                                  execution_id=self.execution_id, socket_io=self.socket_io,
                                                                  log_writer=self.log_writer)
            response = train_predict_thread_obj_ref.get_running_status_of_prediction_thread()
            previous_execution_thread_status = None
            if response is not None:
                if 'is_running' in response:
                    previous_execution_thread_status = response['is_running']
            if previous_execution_thread_status:
                pass

                # self.socket_io.emit("prediction_completed" + str(self.project_id),
                #      {'message': "Training/ Prediction model thread is already running"},
                #      namespace="/training_model")


            if previous_execution_thread_status is None or not previous_execution_thread_status:
                self.sch_storage.update_job_record(job_id=self.execution_id, status='running')
                train_predict_thread_obj_ref.start()


                # self.socket_io.emit("prediction_started" + str(self.project_id),
                #      {'message': "We have started prediction and execution started  by {} and execution id  "
                #                  "is {}".format(self.executed_by,
                #                                 self.execution_id)},
                #      namespace="/training_model")

                train_predict_thread_obj_ref.join()
                self.sch_storage.update_job_record(job_id=self.execution_id, status='successful')
        except Exception as e:
            self.sch_storage.update_job_record(job_id=self.execution_id, status='failed')
            if self.log_writer is not None:
                self.log_writer.log_stop({'status': False, 'error_message': str(e)})
            log_exception = LogExceptionDetail(self.log_writer.executed_by, self.log_writer.execution_id)
            log_exception.log(str(e))

            # self.socket_io.emit("prediction_started" + str(self.project_id), {'message': str(e)}, namespace="/training_model")



    def start_training_prediction_both(self):
        # Uses both the above methods in series
        try:
            self.start_training()
            self.start_prediction()

        except Exception as e:
            if self.log_writer is not None:
                self.log_writer.log_stop({'status': False, 'error_message': str(e)})
            log_exception = LogExceptionDetail(self.log_writer.executed_by, self.log_writer.execution_id)
            log_exception.log(str(e))

            # self.socket_io.emit("prediction_started" + str(self.project_id), {'message': str(e)},
            #                namespace="/training_model")
