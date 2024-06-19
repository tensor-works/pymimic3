import datasets
from pathlib import Path
from generators.pytorch import TorchGenerator
from preprocessing.scalers import AbstractScaler
from utils.IO import *
from datasets.readers import ProcessedSetReader
from pipelines import AbstractPipeline


class TorchPipeline(AbstractPipeline):

    def _create_generator(self, reader: ProcessedSetReader, scaler: AbstractScaler,
                          **generator_options):
        return TorchGenerator(reader=reader, scaler=scaler, **generator_options)

    def fit(self,
            epochs: int,
            no_subdirs: bool = False,
            result_name: str = None,
            patience: int = None,
            save_best_only: bool = True,
            restore_best_weights: bool = True,
            sample_weights: dict = None,
            val_frequency=1,
            restore_last_run: bool = False):

        self._init_result_path(result_name=result_name,
                               restore_last_run=restore_last_run,
                               no_subdirs=no_subdirs)

        info_io(f"Training model in directory\n{self._result_path}")
        self._model.fit(self._train_generator,
                        model_path=self._result_path,
                        epochs=epochs,
                        patience=patience,
                        save_best_only=save_best_only,
                        restore_best_weights=restore_best_weights,
                        sample_weights=sample_weights,
                        val_frequency=val_frequency,
                        val_generator=self._val_generator)

        return self._model


if __name__ == "__main__":
    from models.pytorch.lstm import LSTMNetwork
    from tests.settings import *
    reader = datasets.load_data(chunksize=75836,
                                source_path=TEST_DATA_DEMO,
                                storage_path=SEMITEMP_DIR,
                                discretize=True,
                                time_step_size=1.0,
                                start_at_zero=True,
                                impute_strategy='previous',
                                task="IHM")

    reader = datasets.train_test_split(reader, test_size=0.2, val_size=0.1)

    model = LSTMNetwork(10,
                        0.2,
                        59,
                        bidirectional=False,
                        recurrent_dropout=0.,
                        task=None,
                        target_repl=False,
                        output_dim=1,
                        depth=1)

    pipe = TorchPipeline(storage_path=Path(TEMP_DIR, "torch_pipeline"),
                         reader=reader,
                         model=model,
                         compile_options={
                             "optimizer": "adam",
                             "loss": "logits_binary_crossentropy"
                         },
                         generator_options={
                             "batch_size": 1,
                             "shuffle": True
                         }).fit(epochs=10, save_best_only=False)
