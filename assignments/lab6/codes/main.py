def trigger_pub(data, context):
    from google.cloud import pubsub_v1
    from googleapiclient.discovery import build
    project_id = "atlantean-axon-307504"
    topic_id = "file-name-topic"

    # Create a publisher client to topic
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)

    # Publish path to file in the topic
    pub_data = data['name']
    # Data must be a bytestring
    pub_data = pub_data.encode("utf-8")
    # Client returns future when publisher publish.
    future = publisher.publish(topic_path, pub_data)
    print(future.result())
    print("Published ",pub_data," to ",topic_path)

    