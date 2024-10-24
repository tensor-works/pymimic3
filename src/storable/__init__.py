import os
import time
import atexit
import subprocess
from pymongo import MongoClient
from storable.mongo_dict import MongoDict
from storable.decorator import storable
from storable.settings import *

__all__ = ["storable", "MongoDict"]
DB_SERVER = os.getenv('MONGODB_HOST')
if not DB_SERVER:
    DB_SERVER = 'localhost:27017'


def _is_mongodb_running():
    try:
        # Attempt to connect to MongoDB
        with MongoClient(host=DB_SERVER, serverSelectionTimeoutMS=1000) as temp_client:
            # If this succeeds, MongoDB is running
            temp_client.server_info()
        return True
    except:
        return False


def _start_mongodb():
    global MONGODB_PROCESS, CLIENT, IS_MONGODB_OWNER

    if _is_mongodb_running():
        print("MongoDB is already running.")
        CLIENT = MongoClient(host=DB_SERVER)
        return

    try:
        # Start MongoDB process, binding to all interfaces
        MONGODB_PROCESS = subprocess.Popen(["mongod", "--bind_ip_all"],
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
        time.sleep(2)  # Give MongoDB time to start

        # Connect to MongoDB
        CLIENT = MongoClient(host=DB_SERVER)
        CLIENT.server_info()  # Will raise an exception if connection fails
        IS_MONGODB_OWNER = True
        print("MongoDB started and connected successfully.")
    except Exception as e:
        print(f"Failed to start MongoDB: {e}")
        if MONGODB_PROCESS:
            MONGODB_PROCESS.terminate()
            MONGODB_PROCESS.wait()


def _stop_mongodb():
    global MONGODB_PROCESS, CLIENT, IS_MONGODB_OWNER

    if CLIENT:
        CLIENT.close()
        CLIENT = None

    if IS_MONGODB_OWNER and MONGODB_PROCESS:
        MONGODB_PROCESS.terminate()
        MONGODB_PROCESS.wait()
        MONGODB_PROCESS = None
        IS_MONGODB_OWNER = False
        print("MongoDB stopped.")


# Register the cleanup function to be called when the module is unloaded
atexit.register(_stop_mongodb)

# Start MongoDB when the module is imported
_start_mongodb()
