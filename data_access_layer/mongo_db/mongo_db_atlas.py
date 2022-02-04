import ssl
import urllib
import pymongo
import json
import pandas as pd
import sys

# Getting MongoDB credentials
from project_library_layer.credentials.mongo_db_credential import get_mongo_db_credentials
# Importing MongoDB Exception class
from exception_layer.generic_exception.generic_exception import GenericException as MongoDbException


class MongoDBOperation:
    def __init__(self, user_name=None, password=None):
        """
        Takes username, password, url, cloud type (0:offline(compass) or 1:online(atlas)) from the project_credentials.yaml
        If the details are not present in project_credentials.yaml file, takes username and password manually.
        These details are loaded as attributes for this class.
        """
        try:
            if user_name is None or password is None:
                # creating initial object to fetch mongodb credentials
                credentials = get_mongo_db_credentials()  # return dictionary with username, password, url, cloud type(0:offline(compass) or 1:online(atlas))
                self.user_name = credentials['user_name']
                self.password = credentials['password']
                self.url = credentials["url"]
                self.is_cloud = int(credentials["is_cloud"])
            else:
                # If it fails to get the information from project_credientials.yaml file, pass the username and password manually.
                self.user_name = user_name
                self.password = password
        except Exception as e:
            mongo_db_exception = MongoDbException(
                "Failed to instantiate mongo_db_object in module [{0}] class [{1}] method [{2}]"
                    .format(MongoDBOperation.__module__.__str__(), MongoDBOperation.__name__,"__init__"))
            raise Exception(mongo_db_exception.error_message_detail(str(e), sys)) from e

    def get_mongo_db_url(self):
        """
        Returns MongoDB url
        """
        try:
            # Using .format() adding username and password for MongoDB connection link.
            url = self.url.format(urllib.parse.quote_plus(self.user_name), urllib.parse.quote_plus(self.password))

            if not self.is_cloud:
                # If is_cloud==0 i.e. mongodb compass(local version), then take local link.
                url = "localhost:27017"

            return url
        except Exception as e:
            mongo_db_exception = MongoDbException(
                "Failed to fetch  mongo_db url in module [{0}] class [{1}] method [{2}]"
                    .format(MongoDBOperation.__module__.__str__(), MongoDBOperation.__name__,
                            self.get_mongo_db_url.__name__))
            raise Exception(mongo_db_exception.error_message_detail(str(e), sys)) from e

    def get_database_client_object(self):
        """
        Return pymongoClient object to perform action with MongoDB
        """
        try:
            client = pymongo.MongoClient(self.get_mongo_db_url(),
                                         ssl_cert_reqs=ssl.CERT_NONE)  # creating database client object
            return client

        except Exception as e:
            mongo_db_exception = MongoDbException(
                "Failed to fetch  data base client object in module [{0}] class [{1}] method [{2}]"
                    .format(MongoDBOperation.__module__.__str__(), MongoDBOperation.__name__,
                            self.get_database_client_object.__name__))
            raise Exception(mongo_db_exception.error_message_detail(str(e), sys)) from e

    def close_database_client_object(self, obj_name):
        """
        Takes mongodb client object, closes it, and returns True.
        """
        try:
            obj_name.close()
            return True
        except Exception as e:
            mongo_db_exception = MongoDbException(
                "Failed to close data base client object in module [{0}] class [{1}] method [{2}]"
                    .format(MongoDBOperation.__module__.__str__(), MongoDBOperation.__name__,
                            self.close_database_client_object.__name__))
            raise Exception(mongo_db_exception.error_message_detail(str(e), sys)) from e

    def is_database_present(self, client, db_name):
        """
        Takes client object, and database name, checks if the database is present.
        if it is present, returns True.
        else, returns False.
        """
        try:
            if db_name in client.list_database_names():
                return True
            else:
                return False
        except Exception as e:
            mongo_db_exception = MongoDbException(
                "Failed during checking database  in module [{0}] class [{1}] method [{2}]"
                    .format(MongoDBOperation.__module__.__str__(), MongoDBOperation.__name__,
                            self.is_database_present.__name__))
            raise Exception(mongo_db_exception.error_message_detail(str(e), sys)) from e

    def create_database(self, client, db_name):
        """
        Takes client object, and database name, creates a database with the given database name, if it is not present already, and returns the database.
        If the database is already present, it just returns the database object.
        """
        try:
            return client[db_name]
        except Exception as e:
            mongo_db_exception = MongoDbException(
                "Failure occured duing database creation steps in module [{0}] class [{1}] method [{2}]"
                    .format(MongoDBOperation.__module__.__str__(), MongoDBOperation.__name__,
                            self.create_database.__name__))
            raise Exception(mongo_db_exception.error_message_detail(str(e), sys)) from e

    def create_collection_in_database(self, database, collection_name):
        """
        Takes database object, and collection name.
        Creates a collection with the given collection name in the given database
        """
        try:
            return database[collection_name]
        except Exception as e:
            mongo_db_exception = MongoDbException(
                "Failed during creating collection in database  in module [{0}] class [{1}] method [{2}]"
                    .format(MongoDBOperation.__module__.__str__(), MongoDBOperation.__name__,
                            self.create_collection_in_database.__name__))
            raise Exception(mongo_db_exception.error_message_detail(str(e), sys)) from e

    def is_collection_present(self, collection_name, database):
        """
        Takes database object, and collection name.
        Checks if the collection with the given collection name is present inside the given database.
        """
        try:
            collection_list = database.list_collection_names()

            if collection_name in collection_list:
                # print("Collection:'{COLLECTION_NAME}' in Database:'{DB_NAME}' exists")
                print(f"Collection:'{collection_name}' in Database:'{database}' exists")
                return True

            # print(f"Collection:'{COLLECTION_NAME}' in Database:'{DB_NAME}' does not exists OR\n    no documents are present in the collection")
            print(f"Collection:'{collection_name}' in Database:'{database}' does not exists OR\n    no documents are present in the collection")
            return False

        except Exception as e:
            mongo_db_exception = MongoDbException(
                "Failed during checking collection  in module [{0}] class [{1}] method [{2}]"
                    .format(MongoDBOperation.__module__.__str__(), MongoDBOperation.__name__,
                            self.is_collection_present.__name__))
            raise Exception(mongo_db_exception.error_message_detail(str(e), sys)) from e

    def get_collection(self, collection_name, database):
        """
        Takes collection name and database object, returns the collection object
        """
        try:
            collection = self.create_collection_in_database(database, collection_name)
            return collection
        except Exception as e:
            mongo_db_exception = MongoDbException(
                "Failed in retrival of collection  in module [{0}] class [{1}] method [{2}]"
                    .format(MongoDBOperation.__module__.__str__(), MongoDBOperation.__name__,
                            self.get_collection.__name__))
            raise Exception(mongo_db_exception.error_message_detail(str(e), sys)) from e

    def is_record_present(self, db_name, collection_name, record):
        """
        Takes database name, collection name, and record to search.
        Returns True, If the record is present inside the given database, and collection.
        Else, returns False.
        """
        try:
            client = self.get_database_client_object()                   # client object
            database = self.create_database(client, db_name)             # database object
            collection = self.get_collection(collection_name, database)  # collection object
            record_found = collection.find(record)                       # fetching record
            if record_found.count() > 0:
                client.close()
                return True
            else:
                client.close()
                return False
        except Exception as e:
            mongo_db_exception = MongoDbException(
                "Failed in fetching record  in module [{0}] class [{1}] method [{2}]"
                    .format(MongoDBOperation.__module__.__str__(), MongoDBOperation.__name__,
                            self.is_record_present.__name__))
            raise Exception(mongo_db_exception.error_message_detail(str(e), sys)) from e

    def create_record(self, collection, data):
        """
        Takes collection object, and single record for inserting and returns 1, if the record insertion is successful.

        Requires client object, database object and collection object to be established already.
        """
        try:
            collection.insert_one(data)  # insertion of record in collection
            return 1
        except Exception as e:
            mongo_db_exception = MongoDbException(
                "Failed in inserting record in module [{0}] class [{1}] method [{2}]"
                    .format(MongoDBOperation.__module__.__str__(), MongoDBOperation.__name__,
                            self.create_record.__name__))
            raise Exception(mongo_db_exception.error_message_detail(str(e), sys)) from e

    def create_records(self, collection, data):
        """
        Takes collection object, and more than 1 record as a list of records for inserting.
        Returns the no. of records inserted, if the record insertion is successful.

        Requires client object, database object and collection object to be established already.
        """
        try:
            collection.insert_many(data)
            return len(data)
        except Exception as e:
            mongo_db_exception = MongoDbException(
                "Failed in inserting records in module [{0}] class [{1}] method [{2}]"
                    .format(MongoDBOperation.__module__.__str__(), MongoDBOperation.__name__,
                            self.create_records.__name__))
            raise Exception(mongo_db_exception.error_message_detail(str(e), sys)) from e


    # The above methods are used in the below methods.
    # The below methods are directly used during the implementation
    def insert_record_in_collection(self, db_name, collection_name, record):
        """
        Takes database name, collection name, and a single record object(as dict)
        Establishes a connection to the client object, creates a database with the given database name, creates a collection object
        Uses all these objects for inserting a record, and returns the no. of records inserted i.e. 1
        """
        try:
            no_of_row_inserted = 0
            client = self.get_database_client_object()
            database = self.create_database(client, db_name)
            collection = self.get_collection(collection_name, database)
            if not self.is_record_present(db_name, collection_name, record):
                no_of_row_inserted = self.create_record(collection=collection, data=record)
            client.close()
            return no_of_row_inserted
        except Exception as e:
            mongo_db_exception = MongoDbException(
                "Failed in inserting record  in collection module [{0}] class [{1}] method [{2}]"
                    .format(MongoDBOperation.__module__.__str__(), MongoDBOperation.__name__,
                            self.insert_record_in_collection.__name__))
            raise Exception(mongo_db_exception.error_message_detail(str(e), sys)) from e


    def insert_records_in_collection(self, db_name, collection_name, records):
        """
        Takes database name, collection name, and multiple records as a list of records(as list of dicts)
        Establishes a connection to the client object, creates a database with the given database name, creates a collection object
        Uses all these objects for inserting the records, and returns the no. of records inserted.
        """
        try:
            no_of_row_inserted = 0
            client = self.get_database_client_object()
            database = self.create_database(client, db_name)
            collection = self.get_collection(collection_name, database)
            for record in records:
                if not self.is_record_present(db_name, collection_name, record):
                    no_of_row_inserted = no_of_row_inserted + self.create_record(collection=collection, data=records)
            client.close()
            return no_of_row_inserted
        except Exception as e:
            mongo_db_exception = MongoDbException(
                "Failed in inserting records in collection module [{0}] class [{1}] method [{2}]"
                    .format(MongoDBOperation.__module__.__str__(), MongoDBOperation.__name__,
                            self.insert_record_in_collection.__name__))
            raise Exception(mongo_db_exception.error_message_detail(str(e), sys)) from e


    def drop_collection(self, db_name, collection_name):
        """
        Takes database name, and collection name.
        Drops the collection with the given collection name, if the collection is present inside the database.
        And, returns True.
        """
        try:
            client = self.get_database_client_object()
            database = self.create_database(client, db_name)
            if self.is_collection_present(collection_name, database):
                collection_name = self.get_collection(collection_name, database)
                collection_name.drop()
            return True
        except Exception as e:
            mongo_db_exception = MongoDbException(
                "Failed in droping collection module [{0}] class [{1}] method [{2}]"
                    .format(MongoDBOperation.__module__.__str__(), MongoDBOperation.__name__,
                            self.drop_collection.__name__))
            raise Exception(mongo_db_exception.error_message_detail(str(e), sys)) from e


    def insert_dataframe_into_collection(self, db_name, collection_name, data_frame):
        """
        Takes database name, collection name, data_frame object.
        Converts the dataframe into list of records(or dictionaries)
        Inserts the records into the given collection in the database
        Returns the no. of records(/no. of rows in the dataframe) inserted
        """
        try:
            data_frame.reset_index(drop=True, inplace=True)
            records = list(json.loads(data_frame.T.to_json()).values())
            client = self.get_database_client_object()
            database = self.create_database(client, db_name)
            collection = self.get_collection(collection_name, database)
            collection.insert_many(records)
            return len(records)
        except Exception as e:
            mongo_db_exception = MongoDbException(
                "Failed in inserting dataframe in collection module [{0}] class [{1}] method [{2}]"
                    .format(MongoDBOperation.__module__.__str__(), MongoDBOperation.__name__,
                            self.insert_dataframe_into_collection.__name__))
            raise Exception(mongo_db_exception.error_message_detail(str(e), sys)) from e

    def get_record(self, database_name, collection_name, query=None):
        """
        Takes database name, collection name, and query
        Returns the record found as the given query.
        """
        try:
            client = self.get_database_client_object()
            database = self.create_database(client, database_name)
            collection = self.get_collection(collection_name=collection_name, database=database)
            record = collection.find_one(query)
            return record
        except Exception as e:
            mongo_db_exception = MongoDbException(
                "Failed in retriving record in collection module [{0}] class [{1}] method [{2}]"
                    .format(MongoDBOperation.__module__.__str__(), MongoDBOperation.__name__,
                            self.get_record.__name__))
            raise Exception(mongo_db_exception.error_message_detail(str(e), sys)) from e

    def get_min_value_of_column(self, database_name, collection_name, query, column):
        """
        Takes database name, collection name, query, and column name (or key)
        In the given database and collection, and as per the query, this method returns the minimum value of the column.
        """
        try:
            client = self.get_database_client_object()
            database = self.create_database(client, database_name)
            collection = self.get_collection(collection_name=collection_name, database=database)
            min_value = collection.find(query).sort(column, pymongo.ASCENDING).limit(1)
            value = [min_val for min_val in min_value]
            if len(value) > 0:
                if column in value[0]:
                    return value[0][column]
                else:
                    return None
            else:
                return None
        except Exception as e:
            mongo_db_exception = MongoDbException(
                "Failed in getting minimum value from column in collection module [{0}] class [{1}] method [{2}]"
                    .format(MongoDBOperation.__module__.__str__(), MongoDBOperation.__name__,
                            self.get_record.__name__))
            raise Exception(mongo_db_exception.error_message_detail(str(e), sys)) from e

    def get_max_value_of_column(self, database_name, collection_name, query, column):
        """
        Takes database name, collection name, query, and column name (or key)
        In the given database and collection, and as per the query, this method returns the maximum value of the column.
        """
        try:
            client = self.get_database_client_object()
            database = self.create_database(client, database_name)
            collection = self.get_collection(collection_name=collection_name, database=database)
            max_value = collection.find(query).sort(column, pymongo.DESCENDING).limit(1)
            value = [max_val for max_val in max_value]
            if len(value) > 0:
                if column in value[0]:
                    return value[0][column]
                else:
                    return None
            else:
                return None

        except Exception as e:
            mongo_db_exception = MongoDbException(
                "Failed in getting maximum value from column in collection module [{0}] class [{1}] method [{2}]"
                    .format(MongoDBOperation.__module__.__str__(), MongoDBOperation.__name__,
                            self.get_record.__name__))
            raise Exception(mongo_db_exception.error_message_detail(str(e), sys)) from e

    def get_records(self, database_name, collection_name, query=None):
        """
        Takes database name, collection name, and query
        Returns the records which are selected in the given database and collection, based on the query given.
        """
        try:
            client = self.get_database_client_object()
            database = self.create_database(client, database_name)
            collection = self.get_collection(collection_name=collection_name, database=database)
            record = collection.find(query)
            return record
        except Exception as e:
            mongo_db_exception = MongoDbException(
                "Failed in retriving records in collection module [{0}] class [{1}] method [{2}]"
                    .format(MongoDBOperation.__module__.__str__(), MongoDBOperation.__name__,
                            self.get_record.__name__))
            raise Exception(mongo_db_exception.error_message_detail(str(e), sys)) from e

    def update_record_in_collection(self, database_name, collection_name, query, new_value):
        """
        Takes database name, collection name, query, and new_value to be used for updating.
        Updates the record as per the query with the new_value/record given.
        returns n_updated row
        """
        try:
            client = self.get_database_client_object()
            database = self.create_database(client, database_name)
            collection = self.get_collection(collection_name=collection_name, database=database)
            update_query = {'$set': new_value}
            result = collection.update_one(query, update_query)
            client.close()
            return result.raw_result["nModified"]
        except Exception as e:
            mongo_db_exception = MongoDbException(
                "Failed updating record in collection module [{0}] class [{1}] method [{2}]"
                    .format(MongoDBOperation.__module__.__str__(), MongoDBOperation.__name__,
                            self.update_record_in_collection.__name__))
            raise Exception(mongo_db_exception.error_message_detail(str(e), sys)) from e

    def get_dataframe_of_collection(self, db_name, collection_name, query=None):
        """
        Takes database name, collection name, and query
        Returns pandas dataframe (after dropping the _id column which is MongoDB id) as per the query present in the given database and collection.
        """
        try:
            client = self.get_database_client_object()
            database = self.create_database(client, db_name)
            collection = self.get_collection(collection_name=collection_name, database=database)
            if query is None:
                query = {}
            df = pd.DataFrame(list(collection.find(query)))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)
            return df.copy()
        except Exception as e:
            mongo_db_exception = MongoDbException(
                "Failed in returning dataframe of collection module [{0}] class [{1}] method [{2}]"
                    .format(MongoDBOperation.__module__.__str__(), MongoDBOperation.__name__,
                            self.get_dataframe_of_collection.__name__))
            raise Exception(mongo_db_exception.error_message_detail(str(e), sys)) from e

    def remove_record(self, db_name, collection_name, query):
        """
        Takes database name, collection name, and query
        Removes a single record as per the query, in the given database and collection.
        """
        try:
            client = self.get_database_client_object()
            database = self.create_database(client, db_name)
            collection = self.get_collection(collection_name=collection_name, database=database)
            collection.delete_one(query)
            return True
        except Exception as e:
            mongo_db_exception = MongoDbException(
                "Failed in collection module [{0}] class [{1}] method [{2}]"
                    .format(MongoDBOperation.__module__.__str__(), MongoDBOperation.__name__,
                            self.remove_record.__name__))
            raise Exception(mongo_db_exception.error_message_detail(str(e), sys)) from e
