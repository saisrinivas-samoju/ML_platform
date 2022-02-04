import os
import sys

from flask import render_template, redirect, url_for, jsonify, session
from flask import request
from werkzeug.utils import secure_filename

import json, uuid
from integration_layer.file_management.file_manager import FileManager
from cloud_storage_layer.aws.amazon_simple_storage_service import AmazonSimpleStorageService
from entity_layer.registration.registration import Register
from logging_layer.logger.log_request import LogRequest
from logging_layer.logger.log_exception import LogExceptionDetail

global process_value


class FileOperationController:
    def __init__(self):
        self.registration_obj = Register()
        self.WRITE = "WRITE"
        self.READ = "READ"

    def get_cloud_provider_name(self, cloud_provider):
        # If cloud provider is present in the record, return the cloud provider name.
        cloud_provider_name = ""
        if 'cloud_provider' in cloud_provider:
            if 'Microsoft' in cloud_provider['cloud_provider']:
                cloud_provider_name = "microsoft"
            if 'Amazon' in cloud_provider['cloud_provider']:
                cloud_provider_name = "amazon"
            if 'Google' in cloud_provider['cloud_provider']:
                cloud_provider_name = 'google'
            return cloud_provider_name
        return None

    def cloud_list_directory(self, ):
        """
        To get the list of all the directories in the cloud main directory
        """
        try:
            # for writing log
            log_writer = LogRequest(executed_by=None,execution_id=str(uuid.uuid4()))

            # If the user is logged in
            if 'email_address' in session:
                # Grab the email address of the user for writing in log
                log_writer.executed_by = session['email_address']
                log_writer.log_start(request)

                # Check if the user has access to read the data
                result = self.registration_obj.validate_access(session['email_address'], operation_type=self.READ)

                # If the user has no access, stop the logging part, and return the result
                if not result['status']:
                    log_writer.log_stop(result)
                    return jsonify(result)
                # If the user the access, delete the result, capture the cloud provider data
                del result
                cloud_provider = json.loads(request.data)

                # Cloud provider key is not present in the loaded record, return the message
                if 'cloud_provider' not in cloud_provider:
                    return jsonify({'status': False, 'message': 'cloud provider name not found'})

                # If clould provider key is present in the loaded record, get the name of the cloud provider
                cloud_provider_name = self.get_cloud_provider_name(cloud_provider)

                # Start the file manager with the cloud provider name
                file_manager = FileManager(cloud_provider_name)

                # List directory of the parent folder
                result = file_manager.list_directory("")

                # If the list_directory returns something, i.e. it exists
                if result['status']:
                    # Remove the '/' at the end of every folder name
                    result['directory_list'] = [dir.replace("/", "") for dir in result['directory_list']]
                    # Find the no. of folders in the current directory
                    result['n_directory'] = len(result['directory_list'])
                    # Stop logging and return the result
                    log_writer.log_stop(result)
                    return jsonify(result)
                else:
                    # If the directory doesn't exist, stop logging, and return the result
                    log_writer.log_stop(result)
                    return jsonify(result)
            else:
                # If the user not logged in
                log_writer.log_start(request)
                result = {'status': True, 'message': 'Please login to your account'}
                log_writer.log_stop(result)
                return jsonify(result)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            log_exception=LogExceptionDetail(log_writer.executed_by,log_writer.execution_id)
            log_exception.log(str(exception_detail))
            context={'status': False, 'message': str(exception_detail)}
            log_writer.log_stop(context)
            return jsonify(context)

    def list_directory(self, ):
        # Almost same as the previous cloud_list_directory, but returns response for UI, instead of result

        # for writing log
        log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
        try:
            # If the user is logged in
            if 'email_address' in session:
                # Grab the email address of the user for writing in log
                log_writer.executed_by = session['email_address']
                log_writer.log_start(request)

                # Check if the user has access to read the data
                response = self.registration_obj.validate_access(session['email_address'], operation_type=self.READ)

                # If the user has no access, stop the logging part, and return the response
                if not response['status']:
                    log_writer.log_stop(response)
                    return jsonify(response)
                # If the user the access, delete the response, capture the cloud provider data
                del response
                cloud_provider = json.loads(request.data)

                # Cloud provider key is not present in the loaded record, return the message
                if 'cloud_provider' not in cloud_provider:
                    return jsonify({'status': False, 'message': 'cloud provider name not found'})

                # If clould provider key is present in the loaded record, get the name of the cloud provider
                cloud_provider_name = self.get_cloud_provider_name(cloud_provider)

                # If no folders are present in the cloud provider main directory
                if 'folder_name' not in cloud_provider:
                    return jsonify({'status': False, 'message': 'Folder name is missing'})

                # If folder(s) are present in the cloud provider main directory
                folder_name = cloud_provider['folder_name']

                # Start the file manager with the cloud provider name
                file_manager = FileManager(cloud_provider_name)

                # List directory of the parent folder
                response = file_manager.list_directory(folder_name)

                # If the list_directory returns something, i.e. it exists
                if response['status']:
                    # Remove the '/' at the end of every folder name
                    response['directory_list'] = [dir.replace("/", "") for dir in response['directory_list']]
                    # Find the no. of folders in the current directory
                    response['n_directory'] = len(response['directory_list'])

                    # Stop logging and return the response
                    log_writer.log_stop(response)
                    return jsonify(response)
                else:
                    # If the directory doesn't exist, stop logging, and return the response
                    log_writer.log_stop(response)
                    return jsonify(response)
            else:
                # If the user not logged in
                log_writer.log_start(request)
                response = {'status': False, 'message': 'Please login to your account'}
                log_writer.log_stop(response)
                return jsonify(response)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            log_exception=LogExceptionDetail(log_writer.executed_by,log_writer.execution_id)
            log_exception.log(str(exception_detail))
            context={'status': False, 'message': str(exception_detail)}
            log_writer.log_stop(context)
            return jsonify(context)

    def upload_files(self, ):
        # For uploading files

        # For writting logs
        log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
        try:
            # If the user is logged in
            if 'email_address' in session:
                # Grab the email address of the user for writing in log
                log_writer.executed_by = session['email_address']
                log_writer.log_start(request)

                # Check if the user has access to write the data
                response = self.registration_obj.validate_access(session['email_address'], operation_type=self.WRITE)

                # If the user has no access, stop the logging part, and return the response
                if not response['status']:
                    log_writer.log_stop(response)
                    return jsonify(response)

                # If the user the access, delete the response, request the details of the upload, and get the cloud provider name
                del response
                upload_files_list = request.files.getlist("files")
                upload_folder = request.form['upload_folder_name_of_file']
                cloud_provider = self.get_cloud_provider_name({'cloud_provider': request.form['cloud_to_upload']})

                # Start the file manager with the cloud provider name
                file_manager = FileManager(cloud_provider)

                # If more than 0 files are given for uploading i.e. if any files are given for uploading, take the folder name of the folder where the files are given to upload without '/' at the end.
                response = []
                if upload_folder.__len__() > 0:
                    if upload_folder[-1] == "/":
                        upload_folder = upload_folder[:-1]

                # upload files one by one from the upload files list taken from the user request
                files_uploaded = []
                for upload_file in upload_files_list:
                    result = file_manager.upload_file(upload_folder, upload_file.filename, upload_file, False,
                                                      over_write=True)
                    response.append(result)
                    if result['status']:
                        files_uploaded.append(upload_file.filename)

                # Get the string of all the messages in response got during uploading
                message = ""
                for msg in response:
                    message = message + " " + msg['message']

                # Use this full string of messages for the new response
                response = {'status': True, 'message': message, 'uploaded_files_on_cloud': files_uploaded,
                            'n_file_uploaded': len(files_uploaded)}

                # If status of the response is True, stop logging and return the response
                if response['status']:
                    log_writer.log_stop(response)
                    return jsonify(response)
                # If the status of the response is false, just stop logging
                else:
                    log_writer.log_stop(response)
                    return jsonify(response)
            else:
                # If the user not logged in
                log_writer.log_start(request)
                response = {'status': False, 'message': 'Please login to your account'}
                log_writer.log_stop(response)
                return jsonify(response)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            log_exception=LogExceptionDetail(log_writer.executed_by,log_writer.execution_id)
            log_exception.log(str(exception_detail))
            context={'status': False, 'message': str(exception_detail)}
            log_writer.log_stop(context)
            return jsonify(context)

    def upload_file_(self, ):
        log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
        try:

            if 'email_address' in session:
                log_writer.executed_by = session['email_address']
                log_writer.log_start(request)
                response = self.registration_obj.validate_access(session['email_address'], operation_type=self.READ)
                if not response['status']:

                    context = {'message': response['message'], 'message_status': 'info'}
                    log_writer.log_stop(context)
                    return render_template('file_manager.html', context=context)
                del response
                if request.method == "POST":
                    file_name = request.files['file']
                    filename = secure_filename(file_name.filename)
                    m = AmazonSimpleStorageService()
                    response = m.upload_file("company_name/data/project/wafer_fault_detection/Training_Batch_Files",
                                             file_name.filename, file_name)
                    context = {'message': response['message'], 'message_status': 'info'}
                    log_writer.log_stop(context)
                    return render_template('file_manager.html', context=context)
                context = {'message': None, 'message_status': 'info'}
                log_writer.log_stop(context)
                return render_template('file_manager.html', context=context)
            else:
                log_writer.log_start(request)
                log_writer.log_stop({'navigating':'login'})
                return redirect(url_for('login'))
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            log_exception=LogExceptionDetail(log_writer.executed_by,log_writer.execution_id)
            log_exception.log(str(exception_detail))
            context = {'message': None, 'message_status': 'info', 'error_message': str(exception_detail)}
            log_writer.log_stop(context)
            return render_template('error.html',context=context)

    def create_folder(self, ):
        log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
        try:

            if 'email_address' in session:
                log_writer.executed_by=session['email_address']
                log_writer.log_start(request)
                response = self.registration_obj.validate_access(session['email_address'], operation_type=self.WRITE)
                if not response['status']:
                    log_writer.log_stop(response)
                    return jsonify(response)
                del response
                received_data = json.loads(request.data)  # accepting received data

                cloud_provider = self.get_cloud_provider_name(received_data)  # fetching cloud provider name
                msg = ""
                if cloud_provider is None:
                    msg = msg + "cloud provide name is missing"
                folder_name = None
                if 'folder_name' in received_data:
                    folder_name = received_data['folder_name']
                if folder_name is None:
                    msg = msg + " Folder name is missing"
                upload_folder = None
                if 'upload_folder_name' in received_data:
                    upload_folder = received_data['upload_folder_name']
                if upload_folder is None:
                    msg = msg + "Parent folder is missing"

                if len(msg) > 0:
                    response = {'status': False, 'message': msg}
                    log_writer.log_stop(response)
                    return jsonify(response)

                file_manager = FileManager(cloud_provider)  # crating FIle manager object for specific cloud provider
                if upload_folder.__len__() > 0:
                    if upload_folder[-1] == "/":
                        upload_folder = upload_folder[:-1]
                if upload_folder.__len__() == 0:
                    response = file_manager.create_directory(directory_full_path=folder_name)  # creating directory
                else:
                    response = file_manager.create_directory(
                        directory_full_path=upload_folder + "/" + folder_name)  # creating directory

                if response['status']:
                    response['folder_name'] = folder_name
                    log_writer.log_stop(response)
                    return jsonify(response)
                else:
                    log_writer.log_stop(response)
                    return jsonify(response)
            else:
                log_writer.log_start(request)
                response = {'status': False, 'message': 'Please login to your account'}
                log_writer.log_stop(response)
                return jsonify(response)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            log_exception=LogExceptionDetail(log_writer.executed_by,log_writer.execution_id)
            log_exception.log(str(exception_detail))
            context={'status': False, 'message': str(exception_detail)}
            log_writer.log_stop(context)
            return jsonify(context)

    def delete_folder(self, ):
        log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
        try:

            if 'email_address' in session:
                log_writer.executed_by = session['email_address']
                log_writer.log_start(request)
                response = self.registration_obj.validate_access(session['email_address'], self.WRITE)
                if not response['status']:
                    log_writer.log_stop(response)
                    return jsonify(response)
                del response
                received_data = json.loads(request.data)
                msg = ""
                cloud_provider = self.get_cloud_provider_name(received_data)
                if cloud_provider is None:
                    msg = "Cloud provider name is missing"
                if 'folder_names' in received_data:
                    folder_names = received_data['folder_names']
                else:
                    msg = msg + " folder_names missing"
                if 'directory' in received_data:
                    directory = received_data['directory']
                else:
                    msg = msg + " Directory is missing"

                if msg.__len__() > 0:
                    response = {'status': False, 'message': msg}

                file_manager = FileManager(cloud_provider)

                if directory.__len__() > 0:
                    if directory[-1] == "/":
                        directory = directory[:-1]
                folder_name_list = folder_names.split(";")
                folder_name_list = folder_name_list[:-1]
                msg = ""
                for folder_name in folder_name_list:
                    if directory == "":
                        response = file_manager.remove_directory(directory_full_path=folder_name)
                        msg = msg + response['message']
                    else:
                        response = file_manager.remove_directory(directory_full_path=directory + "/" + folder_name)
                        msg = msg + response['message']
                response = file_manager.list_directory(directory)

                response['message'] = response['message'] + msg
                if response['status']:
                    response['n_directory'] = len(response['directory_list'])
                    response['message'] = response['message'] + msg
                    log_writer.log_stop(response)
                    return jsonify(response)
                else:
                    log_writer.log_stop(response)
                    return jsonify(response)
            else:
                log_writer.log_start(request)
                response = {'status': False, 'message': 'Please login to your account'}
                log_writer.log_stop(response)
                return jsonify(response)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            log_exception=LogExceptionDetail(log_writer.executed_by,log_writer.execution_id)
            log_exception.log(str(exception_detail))
            context={'status': False, 'message': str(exception_detail)}
            log_writer.log_stop(context)
            return jsonify(context)

    def delete_file(self, ):
        log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
        try:

            if 'email_address' in session:
                log_writer.executed_by = session['email_address']
                log_writer.log_start(request)
                response = self.registration_obj.validate_access(session['email_address'], self.WRITE)
                if not response['status']:
                    log_writer.log_stop(response)
                    return jsonify(response)
                del response
                received_data = json.loads(request.data)
                msg = ""
                cloud_provider = self.get_cloud_provider_name(received_data)
                if cloud_provider is None:
                    msg = "Cloud provider name is missing"
                if 'file_names' in received_data:
                    file_names = received_data['file_names']
                else:
                    msg = msg + " file_names missing"
                if 'directory' in received_data:
                    directory = received_data['directory']
                else:
                    msg = msg + " Directory is missing"

                if msg.__len__() > 0:
                    response = {'status': False, 'message': msg}

                file_manager = FileManager(cloud_provider)

                if directory.__len__() > 0:
                    if directory[-1] == "/":
                        directory = directory[:-1]

                msg = ""

                file_name_list = file_names.split(";")
                if len(file_name_list) > 0:
                    file_name_list = file_name_list[:-1]
                for file_name in file_name_list:
                    response = file_manager.remove_file(directory_full_path=directory, file_name=file_name)
                    if response['status']:
                        msg = msg + response['message']
                response = file_manager.list_directory(directory)
                response['message'] = response['message'] + msg
                if response['status']:
                    response['n_directory'] = len(response['directory_list'])
                    log_writer.log_stop(response)
                    return jsonify(response)
                else:
                    log_writer.log_stop(response)
                    return jsonify(response)
            else:
                log_writer.log_start(request)
                response = {'status': False, 'message': 'Please login to your account'}
                log_writer.log_stop(response)
                return jsonify(response)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            log_exception=LogExceptionDetail(log_writer.executed_by,log_writer.execution_id)
            log_exception.log(str(exception_detail))
            context={'status': False, 'message': str(exception_detail)}
            log_writer.log_stop(context)
            return jsonify(context)
