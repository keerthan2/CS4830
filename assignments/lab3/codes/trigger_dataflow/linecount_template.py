import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import StandardOptions

class LinecountOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_value_provider_argument(
            '--input',
            default='gs://dataflow-samples/shakespeare/kinglear.txt',
            help='GS path to input file')
        parser.add_value_provider_argument(
            '--output',
            default="gs://trig_bucket/template_output",
            help='GS path to output file')

runner = 'DataflowRunner' 
project = 'cobalt-pursuit-304714'
region = 'us-central1'
staging_location = "gs://bd_lab3/staging "
temp_location = "gs://trig_bucket/temp "
template_location = "gs://bd_lab3/templates/Line_Count"

pipeline_options = PipelineOptions()
google_cloud_options = pipeline_options.view_as(GoogleCloudOptions)
google_cloud_options.project = project
google_cloud_options.region = region
google_cloud_options.staging_location = staging_location
google_cloud_options.temp_location = temp_location
google_cloud_options.template_location = template_location
pipeline_options.view_as(StandardOptions).runner = runner

linecount_options = pipeline_options.view_as(LinecountOptions)

def line_ind(row):
        return 1

with beam.Pipeline(options=linecount_options) as pipeline:
    count = pipeline | 'Read lines' >> beam.io.ReadFromText(linecount_options.input) | 'Making indicator variable' >> beam.Map(line_ind) | 'Counting lines' >> beam.CombineGlobally(sum) | 'Write results' >> beam.io.WriteToText(linecount_options.output)

