from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from exception_layer.generic_exception.generic_exception import GenericException as RegistrationException
from entity_layer.encryption.encrypt_confidential_data import EncryptData
from project_library_layer.datetime_libray.date_time import get_time, get_date
import sys
import re


class Register:
    def __init__(self):
        self.mongo_db = MongoDBOperation()
        self.database_name = "registration"
        self.collection_name_user = "user"
        self.collection_name_user_allow = "user_allowed"
        self.admin_email_id = "saisrinivas.samoju@gmail.com"
        self.collection_name_user_role = "user_role"
        self.n_attempt = 5

    def is_email_address_allowed(self, email_address):
        """
        Takes the email address in the form.
        Checks if the given email address is allowed to register or not.
        """
        try:
            record = self.mongo_db.get_record(self.database_name, self.collection_name_user_allow,
                                              {'email_address': email_address})
            if record is None:
                return {'status': False, 'message': "Email address [{0}] is not allow !! please contact admin on email "
                                                    "id [{1}] ".format(email_address, self.admin_email_id)}
            return {'status': True, 'message': 'Email address can be used for registration.'}

        except Exception as e:
            registration_exception = RegistrationException("Failed email address validation in class [{0}] method [{1}]"
                                                           .format(Register.__name__,
                                                                   self.is_email_address_allowed.__name__))
            raise Exception(registration_exception.error_message_detail(str(e), sys)) from e

    def is_valid_email(self, email_address):
        """
        Takes the email address in the form.
        Checks if the given email address is in valid format or not.
        """
        try:
            regex = '^[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+$'
            if re.search(regex, email_address):
                return {'status': True, 'message': "Valid email address"}

            else:
                return {'status': False, 'message': "Invalid email address [{0}]".format(email_address)}

        except Exception as e:
            registration_exception = RegistrationException("Failed email address validation in class [{0}] method [{1}]"
                                                           .format(Register.__name__,
                                                                   self.is_valid_email.__name__))
            raise Exception(registration_exception.error_message_detail(str(e), sys)) from e

    def validate_user_detail(self, user_name, email_address, password, confirm_password):
        """
        Takes the user_name, email_address, password, confirm_password.
        Validates all the details
        """
        try:
            error_message = ""
            # If password is not matching with the confirm password
            if password != confirm_password:
                error_message = "Password  and confirm password didn't matched"

            # Checks if the email address is in valid format
            response = self.is_valid_email(email_address)
            if not response['status']:
                error_message = "{0} {1}".format(error_message, response['message'])

            # Checks if the email address is allowed to register or not
            response = self.is_email_address_allowed(email_address)
            if not response['status']:
                error_message = "{0} {1}".format(error_message, response['message'])

            # Checks if the email address is already registered or not
            response = self.is_email_address_used(email_address)
            if response['status']:
                error_message = "{0} {1}".format(error_message, response['message'])

            # Checks if there is any error occurred in the above checkings or not.
            if error_message.__len__() == 0:
                # If there is no error in the previous steps, the user details are allowed to register.
                return {'status': True, 'message': "user detail validated successfully."}
            # Else, not allowed to register
            return {'status': False, 'message': error_message}

        except Exception as e:
            registration_exception = RegistrationException(
                "Failed user detail validation in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Register.__name__,
                            self.validate_user_detail.__name__))
            raise Exception(registration_exception.error_message_detail(str(e), sys)) from e

    def register_user(self, user_name, email_address, password, confirm_password):
        """
        Takes the user_name, email_address, password, confirm_password for registering the user.
        """
        try:
            # Check if the user details are valid to register or not
            response = self.validate_user_detail(user_name, email_address, password, confirm_password)
            if not response['status']:
                # If the details are not valid to register
                return {'status': False, 'message': response['message']}
            # If the details are valid to register
            encryptor = EncryptData()
            # Encrypt the user password
            encrypted_password = encryptor.get_encrypted_text(password)
            # Insert the user details with the encrypted password into the database(='registration') and collection(='user').
            self.mongo_db.insert_record_in_collection(self.database_name, self.collection_name_user,
                                                      {'user_name': user_name,
                                                       'email_address': email_address,
                                                       'password': encrypted_password,
                                                       'register_date': get_date(),
                                                       'register_time': get_time(),
                                                       'updated_time': get_time(),
                                                       'updated_date': get_date(),
                                                       'n_attempt': 0,
                                                       'is_locked': False
                                                       })
            return {'status': True, 'message': "Registration successful."}

        except Exception as e:
            registration_exception = RegistrationException(
                "Failed to save user detail database in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Register.__name__,
                            self.register_user.__name__))
            raise Exception(registration_exception.error_message_detail(str(e), sys)) from e

    def is_email_address_used(self, email_address):
        """
        Takes the email address, checks if it is already used.
        """
        try:
            # Find the email address in the database
            user = self.mongo_db.get_record(self.database_name, self.collection_name_user,
                                            {'email_address': email_address})
            # If the email address is not present in 'registration' database and 'user' collection, then it is not used
            if user is None:
                return {'status': False, 'message': "Email address is not used {0}".format(email_address)}

            # Else it is used
            else:
                return {'status': True, 'message': "Email address is used {0}".format(email_address)}

        except Exception as e:
            registration_exception = RegistrationException(
                "Login failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Register.__name__,
                            self.is_email_address_used.__name__))
            raise Exception(registration_exception.error_message_detail(str(e), sys)) from e

    def verify_user(self, email_address, password):
        """
        Takes email address and password, and verifies if they are present in the database or not
        """
        # Used for logging in
        try:
            # Gets the record from the database with the given email address and password
            user = self.mongo_db.get_record(self.database_name, self.collection_name_user,
                                            {'email_address': email_address})

            # If there is no such record, verification is failed.
            if user is None:
                return {'status': False, 'message': "Invalid email_address {0}".format(email_address)}
            else:
                # No. of attempts by the user to log in will be stored
                n_attempt = int(user['n_attempt'])
                # Account will be locked if the no. of attempts is 5 or more
                is_locked = bool(user['is_locked'])
                print(is_locked)
                if is_locked:
                    return {'status': False, 'message': 'Account locked contact admin emaild id:' + self.admin_email_id}

                encryptor = EncryptData()
                response = encryptor.verify_encrypted_text(password, user['password'])
                # If the password is correct
                if response:
                    self.mongo_db.update_record_in_collection(self.database_name, self.collection_name_user,
                                                              {'email_address': email_address},
                                                              {"n_attempt": 0, 'is_locked': False,
                                                               'updated_time': get_time(),
                                                               'updated_date': get_date()})

                    # Login will be successful and no. of attempts made by the used for logging in will be 0.
                    return {'status': response, 'message': 'Login successfully'}
                else:
                    # Else, no. of attempts for logging in will be appended
                    n_attempt += 1
                    is_locked = False
                    # When the no. of attempts is 5, is_locked will be True.
                    if n_attempt == self.n_attempt:
                        is_locked = True

                    # The details for locking the account will be added to the database
                    self.mongo_db.update_record_in_collection(self.database_name, self.collection_name_user,
                                                              {'email_address': email_address},
                                                              {"n_attempt": n_attempt, 'is_locked': is_locked,
                                                               'updated_time': get_time(),
                                                               'updated_date': get_date()})
                    return {'status': False, 'message': 'Invalid password'}

        except Exception as e:
            registration_exception = RegistrationException(
                "Login failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Register.__name__,
                            self.verify_user.__name__))
            raise Exception(registration_exception.error_message_detail(str(e), sys)) from e

    def reset_password(self, email_address, password, confirm_password):
        """
        Takes email_address, password, confirm_password and resets password
        """
        try:
            error_message = ""
            # Checks if the email address is valid or not
            response = self.is_valid_email(email_address)

            # If the email address is not valid
            if not response['status']:
                # Error message will get a string
                error_message = "{0} {1}".format(error_message, response['message'])

            # Checks the record for the given email address in the 'registration' database and 'collection'
            response = self.mongo_db.get_record(self.database_name, self.collection_name_user,
                                                {'email_address': email_address})
            # If there is no such record
            if response is None:
                # Error message will be a string value
                error_message = "{0} {1}".format(error_message,
                                                 'No account exist with email address [{}]'.format(email_address))

            # Checks if the password and confirm password are same
            if password != confirm_password:
                # If password and confirm password are not same. Error message will be a string value
                error_message = "{0} {1}".format(error_message, "Password  and confirm password didn't matched")

            # Checks if the email address is allowed or not
            response = self.is_email_address_allowed(email_address)

            # If it is not allowed
            if not response['status']:
                # Error message will be a string value
                error_message = "{0} {1}".format(error_message, response['message'])

            # If the length of error_message is equal 0 i.e. no error occurred in the above steps
            if error_message.__len__() == 0:
                # Then, encrypt the password and update the password
                encryptor = EncryptData()
                encrypted_password = encryptor.get_encrypted_text(password)
                self.mongo_db.update_record_in_collection(self.database_name, self.collection_name_user,
                                                          {'email_address': email_address},
                                                          {"password": encrypted_password, 'updated_time': get_time(),
                                                           'updated_date': get_date()})
                return {'status': True, 'message': "password updated successfully."}
            return {'status': False, 'message': error_message}

        except Exception as e:
            registration_exception = RegistrationException(
                "Login failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Register.__name__,
                            self.reset_password.__name__))
            raise Exception(registration_exception.error_message_detail(str(e), sys)) from e

    def add_user_role(self, role_name):
        """
        Takes role_name and adds it to the database
        """
        try:
            # Checks if records with the given role name, and assigns it to records variable
            records = self.mongo_db.get_record(self.database_name, self.collection_name_user_role,
                                               {'user_role': role_name})

            # If the no record with the given role_name
            if records is None:
                # Get the maximum value present in the user_role_id column (more the value, less the access)
                user_role_id = self.mongo_db.get_max_value_of_column(self.database_name,
                                                                     self.collection_name_user_role,
                                                                     query={},
                                                                     column='user_role_id'
                                                                     )

                # If there are no values in the 'user_role_id' column, assign 1 (admin access)
                if user_role_id is None:
                    user_role_id = 1

                # Else, add 1 to the existing maximum value for the user_role_id
                else:
                    user_role_id = user_role_id + 1

                # Create a new record with the given role name and the newly created role_id
                record = {'user_role_id': user_role_id, 'user_role': role_name}

                # Insert this new record to the database
                result = self.mongo_db.insert_record_in_collection(
                    self.database_name,
                    self.collection_name_user_role,
                    record
                )

                # If at least one record is inserted
                if result > 0:
                    return {'status': True, 'message': 'User {} role added. '.format(role_name)}
            # Else
            else:
                return {'status': False, 'message': 'User {} already present. '.format(role_name)}

        except Exception as e:
            registration_exception = RegistrationException(
                "Add user role in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Register.__name__,
                            self.reset_password.__name__))
            raise Exception(registration_exception.error_message_detail(str(e), sys)) from e

    def validate_access(self, email_address, operation_type="WRITE"):
        """
        Takes email address and operation_type
        Returns a record to add in the database 'registration' to allow access or not.
        """
        try:
            admin = 'admin'
            viewer = 'viewer'
            RW = ['READ', 'WRITE']
            R = ['READ']
            return {'status': True, 'message': 'You  have all access for '.format(RW)}
            record = self.mongo_db.get_record(self.database_name, self.collection_name_user_allow,
                                              {'email_address': email_address})
            role_id = record['user_role_id']
            role = self.mongo_db.get_record(self.database_name, self.collection_name_user_role,
                                            {'user_role_id': role_id})
            role_name = role['user_role']
            # If the operation_type is read and write, and role_name is admin, then give the access as both are matching
            if operation_type in RW and role_name == admin:
                return {'status': True, 'message': 'You  have all access for '.format(RW)}

            # If the operation_type is read, and role_name is viewer, then give the access as both are matching
            if operation_type in R and role_name == viewer:
                return {'status': True, 'message': 'You have all access for'.format(R)}
            # Else, no access
            else:
                return {'status': False, 'message': 'You can not perform this action due to insufficient privilege '}

        except Exception as e:
            registration_exception = RegistrationException(
                "validating access in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Register.__name__,
                            self.reset_password.__name__))
            raise Exception(registration_exception.error_message_detail(str(e), sys)) from e
