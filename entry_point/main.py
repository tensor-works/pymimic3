import os
import argparse
from src import CaseHandler

os.environ["DEBUG"] = "1"

if __name__ == "__main__":
    case = CaseHandler()
    parser = argparse.ArgumentParser(
        description='Run the MIMIC pipeline with configuration from the directory configuration.'
        ' The model configuration need to contain a data_config.json, '
        'pipeline_config.json and model_config.json')
    parser.add_argument('path',
                        type=str,
                        help='Absolute path to the directory containint the config.json.')
    args = parser.parse_args()
    case.run(args.path)
    #
