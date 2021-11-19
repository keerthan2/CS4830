import os
from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1
from google.cloud import storage

project_id = "atlantean-axon-307504"
topic_id = "file-name-topic"
subscription_id = "count-line-sub"

# Create a subscriber client
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_id)

def callback(message):
    # Decode the filename back to string
    filename = message.data.decode("utf-8")
    print(f"Received file: {filename}")
    
    # Create a google storage client to read the uploaded file from bucket.
    storage_client = storage.Client()
    bucket = storage_client.get_bucket("bdl_6")

    blob = bucket.blob(filename)
    blob = blob.download_as_string()
    blob = blob.decode('utf-8') # Decodes bytes to string
    lines = blob.split('\n') # Splits the text based on \n. 
    print(f"Number of lines in {filename}: {len(lines)}") # Number of lines = Number of \n
    
    message.ack() # Acknowledge that the message is recieved.

# Default subscriber is a pull subscriber
pull_sub_future = subscriber.subscribe(subscription_path, callback=callback)
print(f"Listening for messages on {subscription_path}..\n")

# By using 'with', the subscriber closes automatically.
with subscriber:
    try:
        # The subscriber listens indefinitely 
       ret = pull_sub_future.result()
    except TimeoutError:
        pull_sub_future.cancel()

