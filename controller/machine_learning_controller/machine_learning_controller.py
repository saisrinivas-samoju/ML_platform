import os
import sys
from os import abort
from flask import render_template, redirect, url_for, jsonify, session, request, Response, stream_with_context
import threading
import json
import time

from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from project_library_layer.initializer.initializer import Initializer

from integration_layer.file_management.file_manager import FileManager
from cloud_storage_layer.aws.amazon_simple_storage_service import AmazonSimpleStorageService
from entity_layer.registration.registration import Register
from logging_layer.logger.log_request import LogRequest
from logging_layer.logger.log_exception import LogExceptionDetail
from entity_layer.project.project import Project
from entity_layer.project.project_configuration import ProjectConfiguration
from thread_layer.train_model_thread.train_model_thread import TrainModelThread
from thread_layer.predict_from_model_thread.predict_from_model_thread import PredictFromModelThread
from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
import json

import uuid
from logging_layer.logger.logger import AppLogger

global process_value


class MachineLearningController:

    def __init__(self):
        self.registration_obj = Register()
        self.project_detail = Project()
        self.project_config = ProjectConfiguration()
        self.WRITE = "WRITE"
        self.READ = "READ"

    def predict_route_client(self):
        project_id = None
        try:
            # Writing the log to MongoDB database
            log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))  # str(uuid.uudi4()) returns a string of unique id for execution_id
            try:
                # log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))

                # If the user logs in, email_address key will be available in session
                # If logged_in
                if 'email_address' in session:
                    # The
                    log_writer.executed_by = session['email_address']
                    log_writer.log_start(request)
                    requested_project_data = json.loads(request.data)
                    project_id = None
                    if 'project_id' in requested_project_data:
                        project_id = int(requested_project_data['project_id'])

                    if project_id is None:
                        raise Exception('Project id required')

                    result = self.registration_obj.validate_access(session['email_address'], operation_type=self.WRITE)
                    if not result['status']:
                        log_writer.log_stop(result)
                        result.update(
                            {'message_status': 'info', 'project_id': project_id,
                             'execution_id': log_writer.execution_id})
                        return jsonify(result)

                    database_name = Initializer().get_training_thread_database_name()
                    collection_name = Initializer().get_thread_status_collection_name()
                    query = {'project_id': project_id, 'is_running': True}
                    result = MongoDBOperation().get_record(database_name=database_name, collection_name=collection_name,
                                                           query=query)
                    if result is not None:
                        execution_id = result['execution_id']
                    else:
                        execution_id = None

                    if execution_id is not None:
                        result = {'message': 'Training/prediction is in progress.', 'execution_id': execution_id,
                                  'status': True, 'message_status': 'info'}
                        log_writer.log_stop(result)
                        return jsonify(result)

                    result = {}

                    predict_from_model_obj = PredictFromModelThread(project_id=project_id,
                                                                    executed_by=log_writer.executed_by,
                                                                    execution_id=log_writer.execution_id,
                                                                    log_writer=log_writer)
                    predict_from_model_obj.start()
                    result.update(
                        {'message': 'Prediction started your execution id {0}'.format(log_writer.execution_id)})
                    result.update({'message_status': 'info', 'project_id': project_id, 'status': True,
                                   'execution_id': log_writer.execution_id})
                    return jsonify(result)
                else:
                    result = {'status': True, 'message': 'Please login to your account',
                              'execution_id': log_writer.execution_id}
                    log_writer.log_stop(result)
                    return jsonify(result)
            except Exception as e:
                result = {'status': False, 'message': str(e), 'message_status': 'info', 'project_id': project_id,
                          'execution_id': log_writer.execution_id}
                log_writer.log_stop(result)
                log_exception = LogExceptionDetail(log_writer.executed_by, log_writer.execution_id)
                log_exception.log(str(e))
                return jsonify(result)

        except Exception as e:
            return jsonify({'status': False,
                            'message': str(e)
                               , 'message_status': 'info', 'project_id': project_id})

    def train_route_client(self):
        project_id = None
        try:
            # Writing the log to MongoDB database
            log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))  # str(uuid.uudi4()) returns a string of unique id for execution_id

            try:
                # log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))

                # If the user logs in, email_address key will be available in session
                # If logged_in
                if 'email_address' in session:
                    # The process is executed by the email_address logged in
                    log_writer.executed_by = session['email_address']

                    # Data will be inserted in the database
                    log_writer.log_start(request)

                    # Capturing data from the website
                    requested_project_data = json.loads(request.data)

                    # Assigning None to project_id
                    project_id = None

                    # If project_id is present in the requested data from the website
                    if 'project_id' in requested_project_data:
                        # Assign the running project_id to the project_id variable
                        project_id = int(requested_project_data['project_id'])

                    # Else, project_id will be equal to None, if it is None, training cannot happen.
                    if project_id is None:
                        raise Exception('Project id required')

                    # Training requires writing access for the user, as the model files and other files should be created during the process
                    # Validate if the logged in user has writing access or not.
                    result = self.registration_obj.validate_access(session['email_address'], operation_type=self.WRITE)

                    # If the user has no writing access
                    if not result['status']:
                        # Stop logging by writing result in the MongoDB database
                        log_writer.log_stop(result)
                        result.update(
                            {'message_status': 'info', 'project_id': project_id,
                             'execution_id': log_writer.execution_id})
                        return jsonify(result)

                    # If the user has writing access
                        # Get training thread name from the Initializer class, which is also the database name for training
                    database_name = Initializer().get_training_thread_database_name()
                        # Get the thread status collection name from the Initializer class, for writing thread status log.
                    collection_name = Initializer().get_thread_status_collection_name()

                    # Get the record where the project with project_id is running.
                    query = {'project_id': project_id, 'is_running': True}
                    result = MongoDBOperation().get_record(database_name=database_name, collection_name=collection_name,
                                                           query=query)

                    # If record is found for the query, capture the execution_id
                    if result is not None:
                        execution_id = result['execution_id']
                    # If no record is found, execution_id is None i.e. nothing has been executed.
                    else:
                        execution_id = None

                    # If execution_id is present,, that means the project with the project_id is already running, so new training/prediction process cannot be done.
                    if execution_id is not None:
                        result = {'message': 'Training/prediction is in progress.', 'execution_id': execution_id,
                                  'status': True, 'message_status': 'info'}
                        log_writer.log_stop(result)
                        return jsonify(result)

                    result = {}

                    train_model = TrainModelThread(project_id=project_id, executed_by=log_writer.executed_by,
                                                   execution_id=log_writer.execution_id, log_writer=log_writer)

                    # 'start' method executes 'run' method in the 'TrainModelThread' class which inherited 'Thread' class
                    # This time consuming process will run in another thread and we can see the responses immediately in the UI using the main thread
                    train_model.start()

                    result.update({'status': True, 'message': 'Training started. keep execution_id[{}] to track'.format(
                        log_writer.execution_id),
                                   'message_status': 'info', 'project_id': project_id,
                                   'execution_id': log_writer.execution_id})

                    log_writer.log_stop(result)

                    return jsonify(result)

                else:
                    result = {'status': True, 'message': 'Please login to your account',
                              'execution_id': log_writer.execution_id}
                    log_writer.log_stop(result)
                    return jsonify(result)

            except Exception as e:
                result = {'status': False, 'message': str(e), 'message_status': 'info', 'project_id': project_id,
                          'execution_id': log_writer.execution_id}
                log_writer.log_stop(result)
                log_exception = LogExceptionDetail(log_writer.executed_by, log_writer.execution_id)
                log_exception.log(str(e))
                return render_template('error.html',
                                       context=result)


        except Exception as e:
            result = {'status': False,
                      'message': str(e)
                , 'message_status': 'info', 'project_id': project_id, 'execution_id': None}
            return render_template('error.html',
                                   context=result)

    def prediction_output_file(self):

        project_id = None
        try:
            # Writing the log to MongoDB database
            log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
            try:
                # log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))

                # If the user logs in, email_address key will be available in session
                # If logged_in
                if 'email_address' in session:
                    # The process is executed by the email_address logged in
                    log_writer.executed_by = session['email_address']

                    # Data will be inserted in the database
                    log_writer.log_start(request)

                    # Capturing data from the website
                    project_id = request.args.get('project_id', None)

                    # Starting with an empty string as error_message
                    error_message = ""
                    if project_id is None:
                        # If project_id is not present, add the error to the error message
                        error_message = error_message + "Project id required"

                    # If the project_id is present, make sure you take the integer value
                    project_id = int(project_id)

                    # Get the project details using project_id
                    result = self.project_detail.get_project_detail(project_id=project_id)
                    project_detail = result.get('project_detail', None)
                    project_name = project_detail.get('project_name', None)
                    result = self.registration_obj.validate_access(session['email_address'], operation_type=self.READ)

                    # If the status of the project is False
                    if not result['status']:
                        # Append the error to the error_message
                        error_message = error_message + result['message']
                        # If the project is failed, output file won't be created.
                        context = {'status': True, 'project_name': project_name, 'output_file': None,
                                   'message': error_message}
                        # Failed log will be written
                        log_writer.log_stop(context)
                        # Taken back to prediction_output page
                        return render_template('prediction_output.html', context=context)

                    # If the project is successful, the status of the project will be true. By using Initializer class, getting the path and file names
                    prediction_file_path = Initializer().get_prediction_output_file_path(project_id=project_id, )
                    prediction_file = Initializer().get_prediction_output_file_name()
                    project_config_detail = self.project_config.get_project_configuration_detail(project_id=project_id)
                    project_config_detail = project_config_detail.get('project_config_detail', None)

                    # If the project is set up, and project configuration is not set up, output file won't be created.
                    if project_config_detail is None:
                        context = {'status': True, 'project_name': project_name, 'output_file': None,
                                   'message': 'project config missing'}
                        log_writer.log_stop(context)
                        return render_template('prediction_output.html', context=context)

                    # If the project configuration is set up, extracting the details from the project configuration.
                    cloud_name = project_config_detail['cloud_storage']   # Grabbing the cloud storage.
                    file_manager = FileManager(cloud_name)                # Managing files in the given cloud storage platform.
                    result = file_manager.read_file_content(directory_full_path=prediction_file_path,
                                                            file_name=prediction_file)  # Get the prediction output file
                    file_content = result.get('file_content', None)   # To read the file content in the prediction output file.
                    # If there is no content in the prediction output file, log it as no output file
                    if file_content is None:
                        context = {'status': True, 'project_name': project_name, 'output_file': None,
                                   'message': 'Output file not found'}
                        log_writer.log_stop(context)
                        return render_template('prediction_output.html', context=context)

                    # If content is present in the prediction file, save to html and show it in the page.
                    context = {'status': True, 'project_name': project_name,
                               'output_file': file_content.to_html(header="true"),
                               'message': 'Output file retrived', }
                    log_writer.log_stop(context)
                    return render_template('prediction_output.html', context=context)

                else:
                    # If email address is not present in the session, i.e. not logged in by any user.
                    # Show the error message to log in.
                    result = {'status': True, 'message': 'Please login to your account'}
                    log_writer.log_stop(result)
                    return Response(result)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                exception_type = e.__repr__()
                exception_detail = {'exception_type': exception_type,
                                    'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                    'detail': sys.exc_info().__str__()}
                print(exception_detail)
                return render_template('error.html',
                                       context={'message': None, 'status ': False, 'message_status': 'info',
                                                'error_message': exception_detail.__str__()})
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            print(exception_detail)
            return render_template('error.html',
                                   context={'message': None, 'status ': False, 'message_status': 'info',
                                            'error_message': exception_detail.__str__()})

    def get_log_detail(self):
        project_id = None
        try:
            # Writing the log to MongoDB database
            log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))

            try:
                # log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))

                # If the user logs in, email_address key will be available in session
                # If logged_in
                if 'email_address' in session:
                    # The process is executed by the email_address logged in
                    log_writer.executed_by = session['email_address']

                    # Data will be inserted in the database
                    log_writer.log_start(request)

                    # Capturing data from the website
                    project_id = request.args.get('project_id', None)
                    execution_id = request.args.get('execution_id', None)
                    error_message = ""

                    # If project_id is None, append error message to with the relevant messsage
                    if project_id is None:
                        error_message = error_message + "Project id required"
                    # If execution_id is None, append error message to with the relevant messsage
                    if execution_id is None:
                        error_message = error_message + "Execution id required"
                    # Check if the logged in user has permission to READ the data
                    result = self.registration_obj.validate_access(session['email_address'], operation_type=self.READ)

                    # If no access is permitted, append it to the error message
                    if not result['status']:
                        error_message = error_message + result['message']

                    # If any above error happened, the len of the error message will be more than 0. If it is more than 0.
                    # Write it in the log, and show it as the reponse
                    if len(error_message) > 0:
                        log_writer.log_stop({'status': True, 'message': error_message})
                        return Response(error_message)

                    # Load the record having training thread details
                    result = MongoDBOperation().get_record(Initializer().get_training_thread_database_name(),
                                                           Initializer().get_thread_status_collection_name(),
                                                           {'execution_id': execution_id})

                    # If there is no record present, show the response that no record like that is present.
                    if result is None:
                        return Response("We don't have any log yet with execution id {}".format(execution_id))

                    # If the record exists, capture the details and show them as response
                    process_type = result['process_type']
                    project_id = int(project_id)
                    return Response(
                        stream_with_context(AppLogger().get_log(project_id=project_id, execution_id=execution_id,
                                                                process_type=process_type)))
                else:
                    # If email address is not present in the session, i.e. not logged in by any user.
                    # Show the error message to log in.
                    result = {'status': True, 'message': 'Please login to your account'}
                    log_writer.log_stop(result)
                    return Response(result)

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                exception_type = e.__repr__()
                exception_detail = {'exception_type': exception_type,
                                    'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                    'detail': sys.exc_info().__str__()}
                result = {'status': False, 'message': f"{exception_detail}", 'message_status': 'info', 'project_id': project_id}
                log_writer.log_stop(result)
                log_exception = LogExceptionDetail(log_writer.executed_by, log_writer.execution_id)
                log_exception.log(f"{exception_detail}")
                return render_template('error.html',
                                       context={'message': None, 'status ': False, 'message_status': 'info',
                                                'error_message': f"{exception_detail}"})

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}

            return render_template('error.html',
                                   context={'message': None, 'status ': False, 'message_status': 'info',
                                            'error_message': f"{exception_detail}"})
