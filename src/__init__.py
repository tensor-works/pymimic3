import datasets
from datasets.readers import ProcessedSetReader
from pathlib import Path
from utils import load_json, write_json
from utils.IO import info_io, error_io
from sklearn.linear_model import LogisticRegression, SGDClassifier
from model.sklearn.standard.linear_models import StandardLogReg
from sklearn.ensemble import RandomForestClassifier
from model.tf2.lstm import LSTM
from model.tf2.logistic_regression import IncrementalLogReg as IncrementalLogRegTF2
from model.sklearn.incremental.linear_models import IncrementalLogRegSKLearn
from pipelines.nn import MIMICPipeline as MIMICNNPipeline
from pipelines.regression import MIMICPipeline as MIMICRegPipeline
from tensorflow.keras.metrics import AUC
from copy import deepcopy


class MultiCaseHandler(object):
    """_summary_
    """

    def __init__(self):
        """_summary_
        """
        pass

    def run(path: Path):
        """_summary_

        Args:
            path (Path): _description_
        """
        directories = list(path.iterdir())
        for directory in directories:
            case = CaseHandler()
            case.run(directory)


class AbstractCaseHandler(object):
    """_summary_
    """

    def __init__(self) -> None:
        """_summary_

        Args:
            case_folder (Path): _description_
        """
        #TODO! unused
        self.regression_models = ["sgd_classifier", "logistic_regression", "random_forest"]
        self.neural_network_models = ["lstm"]
        self.subcase_configs = list()

    def read_case_config(self, case_config: dict, case_folder: Path):
        """_summary_

        Args:
            case_folder (Path): _description_
        """
        # Constants
        name = case_config["name"]
        task = case_config["task"]
        frame_work = case_config["pipeline_config"]["framework"]
        model_type = case_config["model_type"]

        # Compose default pipeline config
        case_config["pipeline_config"].update({
            "model_name": name,
            "root_path": case_folder,
        })
        if task in ["IHM", "DECOMP"]:
            case_config["pipeline_config"]["output_type"] = "sparse"

        # Compose default data config
        case_config["data_config"] = deepcopy(case_config["data_config"])
        if not "extracted" == Path(case_config["data_config"]["storage_path"]).name:
            case_config["data_config"].update({
                "storage_path": str(Path(case_config["data_config"]["storage_path"], "extracted")),
                "task": task
            })

        # Compose defualt config
        default_config = {
            "name": name,
            "model_type": model_type,
            "data_config": case_config["data_config"],
            "model_config": case_config["model_config"],
            "pipeline_config": case_config["pipeline_config"],
        }

        case_config = self.custom_configs(default_config, case_config)

        return default_config, model_type, frame_work, task, name, case_folder

    def write_case_folder(self):
        """_summary_
        """

    def run(self, case_folder: Path):
        """_summary_

        Args:
            case_folder (Path): _description_
        """
        case_config_data = load_json(Path(case_folder, "config.json"))
        if "subcases" in case_config_data:
            for data in case_config_data["subcases"]:
                (config, \
                model_type, \
                framework, \
                task, \
                name, \
                root_path) = self.read_case_config(data, case_folder)

                info_io(f"Loading model config from: {str(Path(root_path, name))}")

                case_path = Path(root_path, name, task)
                case_path.mkdir(parents=True, exist_ok=True)

                write_json(Path(case_path, "config.json"), config)
                if model_type in self.regression_models:
                    self.regression(config, framework, model_type, task)
                elif model_type in self.neural_network_models:
                    self.neural_network(config, framework, model_type, task)
                else:
                    raise ValueError(
                        f"Model type needs to be in {', '.join(str(x) for x in list([*self.regression_models, *self.neural_network_models]))}"
                    )
        else:
            (config, \
            model_type, \
            framework, \
            task, \
            name, \
            root_path) = self.read_case_config(case_config_data, case_folder)

            info_io(f"Loading model config from: {str(Path(root_path))}")

            case_path = Path(root_path)
            case_path.mkdir(parents=True, exist_ok=True)

            write_json(Path(case_path, "config.json"), config)
            print(config)
            if model_type in self.regression_models:
                self.regression(config, framework, model_type, task)
            elif model_type in self.neural_network_models:
                self.neural_network(config, framework, model_type, task)
            else:
                raise ValueError(
                    f"Model type needs to be in {', '.join(str(x) for x in list([*self.regression_models, *self.neural_network_models]))}"
                )

            # except Exception as e:
            #    print(e)

    def regression(self, config, framework, model_type):
        """_summary_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def neural_network(self, config, framework, model_type):
        """_summary_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def custom_configs(self, config: dict) -> dict:
        return config


class CaseHandler(AbstractCaseHandler):
    """_summary_

    Args:
        AbstractCaseHandler (_type_): _description_
    """

    def __init__(self) -> None:
        """_summary_

        Args:
            case_folder (Path): _description_
        """
        super().__init__()

    def custom_configs(self, config: dict, config_json: dict) -> dict:
        """_summary_

        Args:
            config (dict): _description_
            config_json (dict): _description_

        Returns:
            dict: _description_
        """
        config["pipeline_config"]["task"] = config_json["task"]
        config["task"] = config_json["task"]

        return config

    def regression(self, config, framework, model_type, task):
        """_summary_
        """
        regression_models = {
            "tf2": {
                "logistic_regression": IncrementalLogRegTF2
            },
            "sklearn": {
                "incremental": {
                    "sgd_classifier": SGDClassifier,
                    "logistic_regression": IncrementalLogRegSKLearn,
                },
                "standard": {
                    "random_forest": RandomForestClassifier,
                    "logistic_regression": StandardLogReg
                }
            }
        }

        if framework == "sklearn":
            model_switch = regression_models["sklearn"]
            if task in ["IHM", "PHENO"]:
                model_switch = model_switch["standard"]
            else:
                model_switch = model_switch["incremental"]
            try:
                model = model_switch[model_type](task=task,
                                                 random_state=42,
                                                 **config["model_config"])
            except KeyError:
                raise ValueError(
                    f"For the framework {framework}, only the models \"{', '.join(str(x) for x in model_switch)}\" are available."
                )

        else:
            model_switch = regression_models["tf2"]
            try:
                model = model_switch[model_type](task=task,
                                                 input_dim=714,
                                                 random_state=42,
                                                 **config["model_config"])
            except KeyError:
                raise ValueError(
                    f"For the framework {framework}, only the models \"{', '.join(str(x) for x in model_switch)}\" are available."
                )
        config["data_config"].update({"preprocess": True, "engineer": True})

        self.run_case(MIMICRegPipeline(model, **config["pipeline_config"]), config["data_config"])

    def neural_network(self, config, framework, model_type):
        """_summary_
        """
        custom_objects = {"auc_2": AUC(curve='ROC'), "auc_3": AUC(curve='PR')}
        neural_network_models = {"lstm": LSTM}

        model = neural_network_models[model_type](**config["model_config"])
        config["data_config"].update({"preprocess": True})
        self.run_case(
            MIMICNNPipeline(model, custom_objects=custom_objects, **config["pipeline_config"]),
            config["data_config"])

    def run_case(self, pipeline, data_config):
        """_summary_

        Args:
            pipeline (_type_): _description_
        """
        try:
            if not "chunksize" in data_config.keys():
                timeseries, \
                episodic_data, \
                subject_diagnoses, \
                subject_icu_history = datasets.load_data(**data_config)

                pipeline.fit(timeseries=timeseries,
                             episodic_data=episodic_data,
                             subject_diagnoses=subject_diagnoses,
                             subject_icu_history=subject_icu_history)
            else:
                data_path = datasets.load_data(**data_config)
                pipeline.fit(data_path=data_path)
        except Exception as e:
            error_io("Encountered exception:")
            error_io(e)
            error_io("Case is finalized and shut down!")
