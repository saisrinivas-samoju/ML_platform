import yaml
import json
from entity_layer.encryption.encrypt_confidential_data import EncryptData
from project_library_layer.credentials.credential_data import save_azure_blob_storage_connection_str, \
    save_aws_credentials, save_google_cloud_storage_credentials, save_watcher_checkpoint_storage_account_connection_str, \
    save_azure_input_file_storage_connection_str, save_user_detail, save_azure_event_hub_namespace_connection_str, \
    save_flask_session_key,save_email_configuration

def read_params(file_path):

    try:
        configuration = None
        with open(file_path) as f:
            configuration = yaml.safe_load(f)
        return configuration

    except Exception as e:
        raise e

def save_configuration(configuration, file_name):

    try:
        with open(file_name, "w") as f:
            yaml.dump(configuration, f)
        return True

    except Exception as e:
        raise e

if __name__ == "__main__":
    file_name = "project_config.yaml"
    config_detail = read_params(file_name)
    encrypter = EncryptData()
    # Generating a key and storing it in the project project_credientials.yaml, instead of using key from the enviornment variable.
    key = encrypter.generate_key()
    # Grabing the credientials from project_config.yaml file, encrypting them, and adding them to project_credientials.yaml file
    user_name, password, url = config_detail['mongodb']['user_name'], \
                               config_detail['mongodb']['password'], \
                               config_detail['mongodb']['url']
    user_name_encry = encrypter.encrypt_message(user_name, key)
    password_encry = encrypter.encrypt_message(password, key)
    url = encrypter.encrypt_message(url, key)
    project_credentials = {
        "key": key,
        "mongodb": {
            "user_name": user_name_encry,
            "password": password_encry,
            "url": url,
            "is_cloud": config_detail["mongodb"]["is_cloud"],
        },
        "root_folder": config_detail["root_folder"]
    }
    save_configuration(project_credentials, "project_credentials.yaml")

    import time
    time.sleep(5)    # For writing content in the project_credentials.yaml file

    # Inserting cloud storage credientials from the project_config.yaml file to the database
    cloud_storage = config_detail["cloud_storage"]

    # Azure
    azure_connection_str = cloud_storage['azure_blob_storage']['connection_str']
    save_azure_blob_storage_connection_str(connection_str=azure_connection_str)

    # Amazon
    aws_s3_bucket_credentials = cloud_storage['aws_s3_bucket']
    save_aws_credentials(data=aws_s3_bucket_credentials)

    # Saving Google cloud credientials into a gcp.json file
    # gcp
    gcp_json_path = cloud_storage["gcp"]
    gcp_detail = dict(json.load(open(gcp_json_path)))
    save_google_cloud_storage_credentials(gcp_detail)


    watcher_storage_account = config_detail["watcher_checkpoint_storage_account_connection_str"]["connection_str"]
    azure_input_file_connection_str = config_detail["azure_input_file_storage_connection_str"]["connection_str"]
    event_hub_namespace_connection_str = config_detail["event_hub_name_space"]["connection_str"]

    save_watcher_checkpoint_storage_account_connection_str(watcher_storage_account)

    save_azure_input_file_storage_connection_str(azure_input_file_connection_str)

    save_azure_event_hub_namespace_connection_str(event_hub_namespace_connection_str)

    # flask session
    secret_key = config_detail['session']['secret-key']
    save_flask_session_key(secret_key)

    # Adding user details to the database from project_config.yaml file
    user_detail = config_detail['user_detail']
    email = user_detail["email_id"]
    role_id = user_detail['role_id']
    save_user_detail(email, role_id)

    # Adding email details to the database from project_config.yaml file
    email_config = config_detail["email_config"]
    email_config["passkey"] = encrypter.encrypt_message(email_config["passkey"],key)
    save_email_configuration(email_config)
    print(config_detail)
