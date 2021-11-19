def trigger_count_lines(data, context):
    from googleapiclient.discovery import build
    
    project = "cobalt-pursuit-304714"
    job = "trigger_dataflow_job"
    template = f"gs://{data['bucket']}/templates/Line_Count"
    input_path = f"gs://{data['bucket']}/{data['name']}"
    output_path = f"gs://trig_bucket/gcf_outputs/output_{data['name'].split('.')[0]}"
    
    parameters = {
        'input': input_path,
        'output': output_path
    }
    
    environment = {'tempLocation': f"gs://{data['bucket']}/temp"}

    service = build('dataflow', 'v1b3', cache_discovery=False)
    
    request = service.projects().locations().templates().launch(
        projectId=project,
        gcsPath=template,
        location='us-central1',
        body={
            'jobName': job,
            'parameters': parameters,
            'environment':environment
        },
    )
    response = request.execute()
    