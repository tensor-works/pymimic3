from dotenv import load_dotenv
from settings import *

load_dotenv(verbose=False)

__all__ = [
    'NETWORK_METRICS', 'NETWORK_CRITERIONS_TF2', 'NETWORK_CRITERIONS_TORCH', 'GENERATOR_OPTIONS',
    'FINAL_ACTIVATIONS', 'OUTPUT_DIMENSIONS', 'STANDARD_LSTM_PARAMS', 'STANDARD_LSTM_DS_PARAMS',
    'CHANNEL_WISE_LSTM_PARAMS', 'CHANNEL_WISE_LSTM_DS_PARAMS', 'MULTITASK_STANDARD_LSTM_PARAMS',
    'MULTITASK_CHANNEL_WISE_LSTM_PARAMS'
]

# ------------------------- metric settings -------------------------
# -> Metrics do with task
NETWORK_METRICS = {
    "IHM": ["roc_auc", "pr_auc"],
    "DECOMP": [
        "roc_auc",
        "pr_auc",
    ],
    "LOS": ["cohen_kappa", "custom_mae"],
    "PHENO": ["micro_roc_auc", "macro_roc_auc"]
}

NETWORK_CRITERIONS_TF2 = {
    "IHM": "binary_crossentropy",
    "DECOMP": "binary_crossentropy",
    "LOS": "sparse_categorical_crossentropy",  # Is multilabel
    "PHENO": "binary_crossentropy"  # Is multiclass
}

NETWORK_CRITERIONS_TORCH = {
    "IHM": "binary_crossentropy",
    "DECOMP": "binary_crossentropy",
    "LOS": "categorical_crossentropy",  # Is multilabel
    "PHENO": "binary_crossentropy"  # Is multiclass
}

GENERATOR_OPTIONS = {
    "IHM": {},
    "DECOMP": {},
    "LOS": {
        "bining":
            "custom"  # Can be custom, log and None and has effects on the bining of the target 
    },
    "PHENO": {},
}

FINAL_ACTIVATIONS = {
    "IHM": "sigmoid",
    "DECOMP": "sigmoid",
    "LOS": "softmax",  # if partition is none then relu (No negative remaining los)
    "PHENO": "sigmoid",
}

OUTPUT_DIMENSIONS = {
    "IHM": 1,
    "DECOMP": 1,
    "LOS": 10,  # if partition is none then only 1
    "PHENO": 25,
}

# ------------------------- standard lstm settings --------------------
# Optimizer is always adam lr=0.001, beta=0.99
# Batch size is always 8
# Input dim is always 59
# Batch size is at top level since its needed for generator (tf2) or fit method
STANDARD_LSTM_PARAMS = {
    "IHM": {  # Settings for the in-hospital mortality task
        "model": {
            "layer_size": 16,
            "depth": 2
        }
    },
    "DECOMP": {  # Settings for the decompensation task
        "model": {
            "layer_size": 128,
            "depth": 1
        }
    },
    "LOS": {  # Settings for the length of stay task
        "model": {
            "layer_size": 64,
            "depth": 1
        }
    },
    "PHENO": {  # Settings for the phenotyping task
        "model": {
            "layer_size": 256,
            "depth": 1
        }
    }
}

# ------------------------- standard lstm with deep supervision settings --------------------
STANDARD_LSTM_DS_PARAMS = {
    "IHM": {
        "model": {
            "layer_size": 32,
            "depth": 1,
            "target_repl_coef": 0.5
        }
    },
    "DECOMP": {
        "model": {
            "layer_size": 128,
            "depth": 1,
        }
    },
    "LOS": {
        "model": {
            "layer_size": 128,
            "depth": 1
        }
    },
    "PHENO": {
        "model": {
            "layer_size": 256,
            "depth": 1,
            "target_repl_coef": 0.5
        }
    }
}

# ------------------------- channel wise lstm settings --------------------
CHANNEL_WISE_LSTM_PARAMS = {
    "IHM": {  # Settings for the in-hospital mortality task
        "model": {
            "layer_size": 8,
            "depth": 1,
            "size_coef": 4.0
        }
    },
    "DECOMP": {  # Settings for the decompensation task
        "model": {
            "layer_size": 16,
            "depth": 1,
            "size_coef": 4.0
        }
    },
    "LOS": {  # Settings for the length of stay task
        "model": {
            "layer_size": 16,
            "depth": 1,
            "size_coef": 8.0
        }
    },
    "PHENO": {  # Settings for the phenotyping task
        "model": {
            "layer_size": 16,
            "depth": 1,
            "size_coef": 8.0
        }
    }
}

# ------------------------- channel-wise lstm with deep supervision settings --------------------
CHANNEL_WISE_LSTM_DS_PARAMS = {
    "PHENO": {
        "model": {
            "layer_size": 16,
            "depth": 1,
            "size_coef": 8.0,
            "target_repl_coef": 0.5
        }
    },
    "DECOMP": {
        "model": {
            "layer_size": 16,
            "depth": 1,
            "size_coef": 8.0,
            "deep_supervision": True
        }
    },
    "IHM": {
        "model": {
            "layer_size": 16,
            "depth": 1,
            "size_coef": 4.0,
            "target_repl_coef": 0.5
        }
    },
    "LOS": {
        "model": {
            "layer_size": 16,
            "depth": 1,
            "size_coef": 8.0,
            "deep_supervision": True
        }
    }
}

# ------------------------- multitask lstm settings --------------------
MULTITASK_STANDARD_LSTM_PARAMS = {
    "dim": 512,
    "depth": 1,
    "dropout": 0.3,
    "batch_size": 8,
    "timestep": 1.0,
    "partition": "custom",
    "ihm_C": 0.2,
    "decomp_C": 1.0,
    "los_C": 1.5,
    "pheno_C": 1.0,
    "target_repl_coef": 0.5
}

MULTITASK_CHANNEL_WISE_LSTM_PARAMS = {
    "dim": 16,
    "size_coef": 8.0,
    "depth": 1,
    "dropout": 0.3,
    "batch_size": 8,
    "timestep": 1.0,
    "partition": "custom",
    "ihm_C": 0.2,
    "decomp_C": 1.0,
    "los_C": 1.5,
    "pheno_C": 1.0,
    "target_repl_coef": 0.5
}
