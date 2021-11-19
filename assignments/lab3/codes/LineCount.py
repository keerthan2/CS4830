import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import GoogleCloudOptions,StandardOptions
input_txt = 'gs://iitmbd/out.txt'
output = 'gs://bd_lab3/outputs/count_lines'
options = PipelineOptions()
google_cloud_options = options.view_as(GoogleCloudOptions)
google_cloud_options.project = 'cobalt-pursuit-304714'
google_cloud_options.job_name = 'lab3'
google_cloud_options.temp_location = 'gs://bd_lab3'
google_cloud_options.region='us-central1' 
options.view_as(StandardOptions).runner = 'DataflowRunner'

def line_ind(row):
  return 1

with beam.Pipeline(options=options) as pipeline:
  count = pipeline | 'Read lines' >> beam.io.ReadFromText(input_txt) | 'Making indicator variable' >> beam.Map(line_ind) | 'Counting lines' >> beam.CombineGlobally(sum) | 'Write results' >> beam.io.WriteToText(output)
      