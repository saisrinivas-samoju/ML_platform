import time
from datetime import datetime, timedelta

# To get https protocol when running this in heroku
from flask_sslify import SSLify

from wsgiref import simple_server

from flask import Flask, session, request, Response, stream_with_context, jsonify

from threading import Thread, Event

# atexit library will allow you to perform a custom function to perform when the application crashes/stops.
import atexit
import uuid
import os
import os

from flask_cors import CORS, cross_origin

global process_value
from controller.home_controller.home_controller import HomeController
from controller.file_operation_controller.file_operation_controller import FileOperationController
from controller.authentication_contoller.authentication_controller import AuthenticationController
from controller.project_controller.project_controller import ProjectController
from controller.machine_learning_controller.machine_learning_controller import MachineLearningController

from controller.visualization_controller.visualization_controller import VisualizationController
from project_library_layer.initializer.initializer import Initializer
from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from controller.scheduler_controller.scheduler_controller import SchedulerController
from controller.watcher_controller.watcher_controller import WatcherController


class Config:
    SCHEDULER_API_ENABLED = True

def on_exit_app():
    print("Closed!")
    database_name = Initializer().get_training_thread_database_name()
    collection_name = Initializer().get_thread_status_collection_name()
    running_projects = MongoDBOperation().get_records(database_name=database_name, collection_name=collection_name,
                                                      query={'is_running': True})
    for running_project in running_projects:
        # Takes the running projects one by one, removes the MongoDB _id column
        running_project.pop('_id')
        # Saves the running project as copy for using in the update_record_in_collection method
        old_record = running_project.copy()
        # Updates that the running project is failed and not running as application is exited
        running_project.update({'is_running': False, 'message': 'Failure due to exit of application', 'is_Failed': True})

        MongoDBOperation().update_record_in_collection(database_name=database_name,
                                                       collection_name=collection_name,
                                                       query=old_record,
                                                       new_value=running_project)
try:
    UPLOAD_FOLDER = 'file/'
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

    thread = Thread()
    thread_stop_event = Event()

    initial = Initializer()
    app = Flask(__name__)
    app.secret_key = initial.get_session_secret_key()  # Takes secret session key from session database, and secretKey collection
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    CORS(app)

    if 'DYNO' in os.environ:  # only trigger SSLify if the app is running on Heroku, to get https url
        sslify = SSLify(app)

    # When you schedule a task, the control will come here.
    scheduler_controller = SchedulerController()
    scheduler_controller.get_scheduler_object().start()

    atexit.register(on_exit_app)    # Registering on_exit_app function to atexit class will allow you to perform on_exit_app whenever the app stops

    home_controller = HomeController()
    app.add_url_rule('/', view_func=home_controller.index, methods=['GET', 'POST'])

    authentication_controller = AuthenticationController()
    app.add_url_rule('/validate_email_address', view_func=authentication_controller.validate_email_address,
                     methods=['GET', 'POST'])
    app.add_url_rule('/register', view_func=authentication_controller.register, methods=['GET', 'POST'])
    app.add_url_rule('/login', view_func=authentication_controller.login, methods=['GET', 'POST'])
    app.add_url_rule('/logout', view_func=authentication_controller.logout, methods=['GET', 'POST'])

    file_operation_controller = FileOperationController()
    app.add_url_rule('/cloud_list_directory', view_func=file_operation_controller.cloud_list_directory,
                     methods=['GET', 'POST'])
    app.add_url_rule('/list_directory', view_func=file_operation_controller.list_directory, methods=['GET', 'POST'])
    app.add_url_rule('/upload_files', view_func=file_operation_controller.upload_files, methods=['GET', 'POST'])
    app.add_url_rule('/create_folder', view_func=file_operation_controller.create_folder, methods=['GET', 'POST'])
    app.add_url_rule('/delete_folder', view_func=file_operation_controller.delete_folder, methods=['GET', 'POST'])
    app.add_url_rule('/delete_file', view_func=file_operation_controller.delete_file, methods=['GET', 'POST'])
    app.add_url_rule('/upload_file', view_func=file_operation_controller.upload_file_, methods=['GET', 'POST'])

    project_controller = ProjectController()
    app.add_url_rule('/project', view_func=project_controller.project_list, methods=['GET', 'POST'])
    app.add_url_rule('/project_detail', view_func=project_controller.project_detail, methods=['GET', 'POST'])
    app.add_url_rule('/save_project', view_func=project_controller.save_project_data, methods=['GET', 'POST'])
    app.add_url_rule('/save_project_config', view_func=project_controller.save_project_configuration,
                     methods=['GET', 'POST'])


    machine_learning_controller = MachineLearningController()
    app.add_url_rule('/train', view_func=machine_learning_controller.train_route_client, methods=['GET', 'POST'])
    app.add_url_rule('/predict', view_func=machine_learning_controller.predict_route_client, methods=['GET', 'POST'])
    app.add_url_rule('/stream', view_func=machine_learning_controller.get_log_detail, methods=['GET', 'POST'])
    app.add_url_rule('/prediction_output', view_func=machine_learning_controller.prediction_output_file,
                     methods=['GET', 'POST'])

    visualization_controller = VisualizationController()

    app.add_url_rule('/dashboard', view_func=visualization_controller.dashboard, methods=['GET', 'POST'])
    """
    app.add_url_rule('/dashboard', view_func=visualiztion_controller.report, methods=['GET', 'POST'])
    app.add_url_rule('/bar', view_func=visualiztion_controller.change_features, methods=['GET', 'POST'])
    """
    app.add_url_rule('/report', view_func=visualization_controller.visualization_project_list, methods=['GET', 'POST'])
    app.add_url_rule('/report_detail', view_func=visualization_controller.report_detail, methods=['GET', 'POST'])
    app.add_url_rule('/graph', view_func=visualization_controller.display_graph, methods=['GET', 'POST'])

    app.add_url_rule('/scheduler', view_func=scheduler_controller.scheduler_index, methods=['GET', 'POST'])
    app.add_url_rule('/scheduler_ajax', view_func=scheduler_controller.scheduler_ajax_index, methods=['GET', 'POST'])
    app.add_url_rule('/add_job_at_specific_time', view_func=scheduler_controller.add_job_at_specific_time,
                     methods=['GET', 'POST'])
    app.add_url_rule('/add_job_within_a_day', view_func=scheduler_controller.add_job_within_a_day, methods=['GET', 'POST'])
    app.add_url_rule('/add_job_on_week_day', view_func=scheduler_controller.add_job_in_week_day, methods=['GET', 'POST'])
    app.add_url_rule('/remove_existing_job', view_func=scheduler_controller.remove_existing_job, methods=['GET', 'POST'])
    watcher_controller = WatcherController()

    app.add_url_rule('/watcher_detail', view_func=watcher_controller.display_captured_event, methods=['GET', 'POST'])

finally:
    pass

# port = int(os.getenv("PORT", 5000))
port = int(os.getenv("PORT", 80))

if __name__ == "__main__":
    host = '0.0.0.0'
    httpd = simple_server.make_server(host=host, port=port, app=app)
    httpd.serve_forever()
