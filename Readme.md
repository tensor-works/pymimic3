<div>
    <img src="./docs/assets/FastMimicLogo.png" width="300" height="300" style="float: right; margin: 15px;" />
    <div>
        <h1>FastMIMIC3</h1>
    !This github is under construtction!
    This project is an enhanced recreation of the <a href="https://github.com/YerevaNN/mimic3-benchmarks">original MIMIC benchmarking code base</a>. TDeveloped during my tenure at the Technical University of Munich's <a href="https://cvg.cit.tum.de/">Computer Vision Group</a>, it serves as a versatile tool for investigating concept drift in medical data. 
    <h3>Key Features:</h3>
    <ul> <li><b>Enhanced Performance</b>:
    FastMIMIC3 incorporates several performance optimizations to streamline its operation, particularly on machines with limited resources.</li>
    <li><b>Parallel Processing</b>:
    One standout feature is the ability to extract and preprocess data using multiple cores in chunks of specifiable size. This parallel processing capability significantly accelerates modification of the billions of subject events.</li>
    <li><b>Selective Data Extraction</b>:
    Users can opt to reduce the pool of subjects, eliminating the need to extract the entire database. This selective approach enhances efficiency and conserves computational resources.</li>
    <li><b>Configurable Pipeline</b>:
    FastMIMIC3 boasts a preimplemented JSON-configurable pipeline, facilitating quick experimentation with different settings. This feature enables researchers to conduct back-to-back experiments seamlessly and organize results effectively.</li>
    <li><b>Configurable Datasplit</b>:
    FastMIMIC3 boasts a preimplemented JSON-configurable pipeline, facilitating quick experimentation with different settings. This feature enables researchers to conduct back-to-back experiments seamlessly and organize results effectively.</li>
    <li><b>Advanced Data Splitting Utility:</b> FastMIMIC3 introduces an enhanced data splitting utility that offers splitting by subject features, such as age, insurance type, etc. in order to investigate the concept drift resulting for data, skewed towards a certain type of patient.</li>
    </ul>
    </div>
    <br clear="left"/>

</div>


## Table of Contents
- [Usage](#usage)
- [Installation](#installation)
- [Contributing](#contributing)

## Road to Release
This GitHub repository is currently being reworked from its original state at the Computer Vision Group to its final public version. The goal is to ensure the functionality of the repository through tests while simplifying the API and providing documentation. The following steps still need to be taken to make the repository available as a pip or conda package:
- Rework: Generators, Pipeline, CaseHandler, Datasplit
- Examples
- Documentation
- Integrate with anyscale

## Usage

Processing the data and obtaining a reader:

```python
import datasets
from datasets.readers import ExtractedSetReader

reader = datasets.load_data(source_path=/path/to/my/mimic/csvs,
                            storage_path=/path/to/my/output,
                            chunksize=10000,
                            num_subjects=1000,
                            preprocess=True,
                            engineer=True,
                            task=task_name)
```
Explanation:
- The `source_path` should indicate the location of your mimic download. Due to the discrepancies between the demo dataset and the actual full sized dataset, the `load_data` function might fail for the demo version. Check out `tests/etc/setup`.sh-or-ps1 for a step to step guide on how to fix the deom dataset. 
- The `storage_path` should indicate the loaction where the processing output is stored.  
- The `chunksize`  specifies how many subject events should be processed per core at a time. The system will try and utilize all cores except one for the extraction. Not specifying this option will revert to the classic on-cored approach.
- The `num_subjects` specifies how many subjects you want to extract. When not specified, all subjects will be extracted. You can also pass specific subject IDs using the `subject_ids` option.
- The `preprocess` and `engineer` option indicate wether you want the data to be preprocessed or engineered, that is prepared for a neural network of for a logistic/forest classification method.
- The `task` can be one of IHM, DECOMP, PHENO or LOS and indicates for what task the data should be processed.

The implementation is crash safe and will rerun the processing only when the subject pool is extended in the call.

Now that the processing is done, we have a reader on hand. The reader is a core object for the database with all other objects building upon its imediate access to the data. The reader allows us to retrieve a specific subject, a number of subjects, or a random subject:
```python
X, y = reader.read_samples([1234, 1235], read_ids=True).values()
# X contains the processed data, while y the contains the labels. The returns are sturcutred as follows:
#  subject | stay | data
# {
#   1234:{ 
#           4321: pd.Dataframe
#         }
#   1235:{ 
#           4322: pd.Dataframe
#           4323: pd.Dataframe
#         }
# }
#

X, y = reader.read_samples(read_ids=True).values()
# Reads all available samples

X, y = reader.read_sample(1234, read_ids=True).values()
# Reads only 1234 without the subject level in the dictionary

X, y = reader.random_samples(10, read_ids=True).values()
# Reads 10 random samples
```
When not specifying the `read_ids` option, the data will be read as a single list of subject stay data frames (`List[pd.DataFrame]`).


Next, the frames need to be discretized. This includes:
- **Imputation**: with the strategies `next`, `previous`, `zero`, `normal`
- **Data binning**: start at `zero` or `first` and choose the step size
- **Categorization**: one-hot categorizes non-numeric columns

```python
from preprocessing.discretizers import BatchDiscretizer

discretizer = BatchDiscretizer(reader,
                               storage_path="/path/to/output/",
                               time_step_size=1.0,
                               start_at_zero="zero",
                               impute_strategy="next")

reader = discretizer.transform()
```
The reader now points towards the discretized dataset.

In order to run our neural network, we need to create a generator object. First, the dataset is split into train, test, and validation sets:

```python
from generators.nn import BatchGenerator
from preprocessing.normalizer import Normalizer

normalizer = Normalizer("path/to/normalizer/storage")
normalizer.fit_reader(reader)

generator = BatchGenerator(reader=reader,
                           normalizer=normalizer,
                           batch_size=16,
                           shuffle=True)
```

Finally, we can train the model, assuming that the input and output shape fit the task:
```python
model.fit(generator,
          steps_per_epoch=generator.steps,
          epochs=10)
```

Alternatively, we can use the dedicated pipeline to automate callbacks and output creation:
```python
from pipelines.nn import Pipeline

pipe = Pipeline(storage_path="/path/to/your/output/",
               reader=reader,
               model=model,
               generator_options={
                  "batch_size": 16,
                  "shuffle": True
               },
               model_options={
                  "epochs":10
               }).fit()
```
This will add checkpoints and history callbacks to the run and store the results at `storage_path`.

Once fitted, the resulting model can be retrieved or evaluated:
```python
pipe.evaluate()

model = pipe.model
```

To further ease the operation of the pipeline, a JSON configurable case handler is implemented. The handler only needs a path to the case directory containing the `config.json` file. The handler is then invoked by:
 ```python
 from casehandler import CaseHandler

 case = CaseHandler()
 case.run(case_dir="path/to/your/case/dir", model)
 ```
The `config.json` now configures the pipeline:
```json
{
    "name": "test",
    "task": "IHM",
    "data_option": {
        "chunksize": 5000000,
        "source_path": "/path/to/your/dataset",
        "storage_path": "/path/to/your/data/output/"
    },
    "generator_options": {
        "batch_size": 8,
        "shuffle": true
    },
    "model_options": {
        "epochs": 10
    },
    "split_options": {
        "validation_fraction": 0.2,
        "test_fraction": 0.2
    }
}
```

## Installation

The repository comes with a setup script that will:
- Fetch necessary config file from the original github repo.
- Create the conda environment.
- Setup necessary environment variables and store them in the /.env file.

Dependencies are a functioning conda install with the recent libmamba solver, as well as a working installation of python. and wget for Ubuntu. Installation scripts are availabe for Windows: 

```
etc\setup.ps1
```
and for Linux (Ubuntu):
```
sudo bash etc\setup.sh
```
When run in vscode, the environment variables will be set in the integrated terminal from the settings.json. When using a standalone terminal, the environment variable can be sourced from the /.env file. For Ubuntu this can be done by sourcing the envvars.sh:
```
source etc/envvars.sh
```
and for windows:
```
TODO!
```

## Contributing

If you have any feature requests, open an issue. If you want to provide a hotfix or a feature yourself, open a pull request. Quickly describe the feature and its expected functionality, as well as testing for the new feature. I will try to get back to you as fast as possible. Make sure that the modified version of the code passes all tests and that the code is linted using yapf and the local settings.

To setup the tests, you need to have setup the repository with an active mimic3 conda environment. Next you can setup the tests in Linux by calling:
```bash
sudo bash tests/etc/setup.sh 
```
or in Windows by calling: 
```bash
tests/etc/setup.ps1 
```

This might take a while. Once you are done, you can run the tests by calling: 
```bash
pytest tests -v 
```
It is recommended to do so in a stand-alone terminal due to pytests tendency to get stuck in execution.