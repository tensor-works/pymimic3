import os
from river import optim
from pathlib import Path
from dotenv import load_dotenv

__all__ = [
    "TEMP_DIR", "EXAMPLE_DIR", "EXAMPLE_DATA_DEMO", "EXAMPLE_DATA", "TEMP_DIR", "YERVA_SPLIT",
    "LOG_METRICS", "NETWORK_METRICS", "BENCHMARK_MODEL", "LOG_REG_PARAMS", "STANDARD_LSTM_PARAMS",
    "STANDARD_LSTM_DS_PARAMS", "CHANNEL_WISE_LSTM_PARAMS"
]

load_dotenv()

EXAMPLE_DIR = Path(os.getenv("EXAMPLES"))
EXAMPLE_DATA = Path(EXAMPLE_DIR, "data")
EXAMPLE_DATA_DEMO = Path(
    EXAMPLE_DATA,
    "physionet.org",
    "files",
    "mimiciii-demo",
    "1.4",
)
TEMP_DIR = Path(EXAMPLE_DATA, "temp")
YERVA_SPLIT = Path(EXAMPLE_DIR, "yerva_nn_benchmark", "data_split")
LOG_METRICS = {
    "PHENO": ["micro_roc_auc", "macro_roc_auc"],
    "DECOMP": ["roc_auc", "pr_auc", "classification_report"],
    "IHM": ["roc_auc", "pr_auc", "classification_report"],
    "LOS": ["cohen_kappa", "mae", "los_classification_report"]
}
LOG_REG_PARAMS = {
    "PHENO": {
        "l2": 0.1,
    },
    "DECOMP": {
        "l2": 0.001,
    },
    "IHM": {
        "l2": 0.001,
    },
    "LOS": {
        "optimizer": optim.SGD(lr=0.00001),
    }
}
NETWORK_METRICS = {
    "PHENO": ["micro_roc_auc", "macro_roc_auc"],
    "DECOMP": [
        "roc_auc",
        "pr_auc",
    ],
    "IHM": ["roc_auc", "pr_auc"],
    "LOS": ["cohen_kappa", "mae"]
}
STANDARD_LSTM_PARAMS = {
    "PHENO": {
        "model": {
            "input_dim": 59,
            "dropout": 0.3,
            "layer_size": 256,
            "depth": 1,
            "final_activation": "sigmoid",
            "output_dim": 25
        },
        "generator_options": {
            "batch_size": 8
        },
        "compile_options": {
            "loss": "binary_crossentropy",
            "optimizer": "adam"
        },
        "training": {
            "epochs": 100
        }
    },
    "DECOMP": {
        "model": {
            "input_dim": 59,
            "final_activation": "sigmoid",
            "layer_size": 128,
            "depth": 1
        },
        "compile_options": {
            "loss": "binary_crossentropy",
            "optimizer": "adam"
        },
        "generator_options": {
            "batch_size": 8,
        },
        "training": {
            "epochs": 100
        }
    },
    "IHM": {
        "model": {
            "input_dim": 59,
            "dropout": 0.3,
            "final_activation": "sigmoid",
            "output_dim": 1,
            "layer_size": 16,
            "depth": 2
        },
        "compile_options": {
            "loss": "binary_crossentropy",
            "optimizer": "adam"
        },
        "generator_options": {
            "batch_size": 8,
        },
        "training": {
            "epochs": 100
        }
    },
    "LOS": {
        "model": {
            "input_dim": 59,
            "dropout": 0.3,
            "layer_size": 64,
            "depth": 1,
            "final_activation": "softmax",  # if partition is none then relu
            "output_dim": 10  # if partition is none then only 1
        },
        "compile_options": {
            "loss":
                "sparse_categorical_crossentropy",  # mean_squared_logarithmic_error if partition is None
            "optimizer": "adam"
        },
        "generator_options": {
            "batch_size": 8,
            "partition":
                "custom"  # Can be custom, log and None and has effects on the bining of the target 
        },
        "training": {
            "epochs": 100
        }
    }
}

STANDARD_LSTM_DS_PARAMS = {
    "PHENO": {
        "dim": 256,
        "depth": 1,
        "batch_size": 8,
        "dropout": 0.3,
        "target_repl_coef": 0.5
    },
    "DECOMP": {
        "dim": 128,
        "depth": 1,
        "batch_size": 8,
        "dropout": 0.3,
        "deep_supervision": True
    },
    "IHM": {
        "dim": 32,
        "depth": 1,
        "batch_size": 8,
        "dropout": 0.3,
        "target_repl_coef": 0.5
    },
    "LOS": {
        "dim": 128,
        "depth": 1,
        "batch_size": 8,
        "dropout": 0.3,
        "partition": "custom",
        "deep_supervision": True
    }
}

CHANNEL_WISE_LSTM_PARAMS = {
    "PHENO": {
        "dim": 16,
        "depth": 1,
        "batch_size": 8,
        "dropout": 0.3,
        "size_coef": 8.0
    },
    "DECOMP": {
        "dim": 16,
        "depth": 1,
        "batch_size": 8,
        "size_coef": 4.0
    },
    "IHM": {
        "dim": 8,
        "depth": 1,
        "batch_size": 8,
        "dropout": 0.3,
        "size_coef": 4.0
    },
    "LOS": {
        "dim": 16,
        "depth": 1,
        "batch_size": 8,
        "size_coef": 8.0,
        "partition": "custom"
    }
}

BENCHMARK_MODEL = Path(EXAMPLE_DATA, "benchmark_models")
