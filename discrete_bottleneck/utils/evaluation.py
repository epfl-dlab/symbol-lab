import jsonlines

# from genie.datamodule.utils import TripletUtils


class Default:
    def __init__(self, path):
        self.input_file_path = path

#     def get_predicted(self, sample_output, verbose=False):
#         return TripletUtils.convert_text_sequence_to_text_triples(sample_output["guess"], verbose)

#     def get_target(self, sample_output, verbose=False):
#         return TripletUtils.convert_text_sequence_to_text_triples(sample_output["raw_output"], verbose)

#     def get_output_data(self):
#         with jsonlines.open(self.input_file_path) as f:
#             data = [sample for sample in f]

#         return data
