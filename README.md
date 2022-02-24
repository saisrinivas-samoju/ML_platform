# Machine Learning Platform Project

This Project will allow you host your Machine Learning Projects

## Project Link

<a href="https://ml-platform.azurewebsites.net/">Azure app link [Click here]</a>

## Problem Statement

The goal of this project is to build a platform for hosting machine learning projects and all the required files in cloud, and automate the training and prediction process of the hosted projects.

## Architecture

<a href="https://ibb.co/8sGGcdW"><img src="https://i.ibb.co/zxCCJ7y/ML-platform-architecture-diagram.png" alt="ML-platform-architecture-diagram" border="0"></a>

## Brief description of the layers

* **Database layer:** All the logs and credentials are stored in the databases, operated by the database layer.
* **Cloud Storage:** All the files related to the machine learning projects including the machine learning models will be stored in three different cloud platforms and all their related operations are done in the cloud storage layer.
* **Entity layer:** This layers consists of entities that perform different operations like registration, encryption of the user details etc.
* **Logging layer:** All the logging processes are performed under this layer.
* **Exception layer:** All the exceptions are logged by this layer.
* **Integration layer:** It integrates all the cloud platforms and performs file operations.
* **Project Library layer:** It has 4 modules that help in connection, verification and project execution.
* **Thread Layer**: It creates separate threads for training/prediction process and logging.
* **Controllers**: This helps the user to connect with the system and execute operations.

## Screenshots for the training process execution

### Login
####
<a href="https://ibb.co/yPxZJZ5"><img src="https://i.ibb.co/2hpCBCP/01-login.png" alt="01-login" border="0"></a>

### Welcome page
####
<a href="https://ibb.co/NWhQ8X4"><img src="https://i.ibb.co/KVZRMY1/02-logged-in.png" alt="02-logged-in" border="0"></a>

### File Management Home page
Total three cloud storage services are allocated for storing the files... AWS, Azure, and GCP.
####
<a href="https://ibb.co/S7T5hFy"><img src="https://i.ibb.co/Wp9Gbm5/03-file-management.png" alt="03-file-management" border="0"></a>

### Project Configuration page
Add the details of your project including scheme details of the project for validation.

All your project related files will be stored in the same cloud storage you choose here.
####
<a href="https://ibb.co/G5LqCsh"><img src="https://i.ibb.co/VqZsMHK/04-Project-configuration.png" alt="04-Project-configuration" border="0"></a>

### Training
Select your project in the project management tab after configuring the project.

Click on the train button for executing the training process.

Note: you can only train the models if you have admin access.
####

<a href="https://ibb.co/XFkh9kj"><img src="https://i.ibb.co/MP5Yb5Z/05-start-training.png" alt="05-start-training" border="0"></a>

Though we started the training process, without data our machine learning models cannot get trained. That's why, we see a detailed description of the error in the message column.

### Upload Training Batch Files
For uploading the training batch files to train our models, we have to go to the **File Management** tab, choose the cloud service we chose while configuring the project and then go to a specific directory to upload the batch files for training.
####
<a href="https://ibb.co/Wpr9y56"><img src="https://i.ibb.co/3YxXNd7/06-upload-files-for-training.png" alt="06-upload-files-for-training" border="0"></a>

### Folder Structure
You can see the same folder structure in the chosen cloud service account (AWS S3 Bucket, in this case)
####
<a href="https://ibb.co/HP7pF9f"><img src="https://i.ibb.co/C9Bs1S3/06-5-same-folder-structure-in-the-cloud-service.png" alt="06-5-same-folder-structure-in-the-cloud-service" border="0"></a>

### Start Training
Once the batch files are uploaded in the proper directory, go to the **Project Management** tab and click Train button.
####
<a href="https://ibb.co/xSm80RP"><img src="https://i.ibb.co/HDNpsk1/07-re-start-training.png" alt="07-re-start-training" border="0"></a>

### Training message
Once you click Train, it will show a message that the *training is in progress*.
####
<a href="https://ibb.co/wrmTvjd"><img src="https://i.ibb.co/JFJNhgc/08-Training-will-start.png" alt="08-Training-will-start" border="0"></a>

### Notifications
All the important notifications regarding training, prediction and error reports will be directly mailed to my email address along with the other mail addresses I added.
####
<a href="https://ibb.co/C77W08n"><img src="https://i.ibb.co/XssVJLz/09-status-mail.png" alt="09-status-mail" border="0"></a>

### Monitoring the logs
The user and anyone who has access, can view the logs in real-time in the same page.
###
<a href="https://ibb.co/fdCrvMw"><img src="https://i.ibb.co/1G9R6ZW/10-monitor-the-logs.png" alt="10-monitor-the-logs" border="0"></a>
