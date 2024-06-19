import datasets
from preprocessing.scalers import MinMaxScaler
from generators.pytorch import TorchGenerator


def test_lstm_with_deep_supervision():
    reader = datasets.train_test_split(reader, test_size=0.2, val_size=0.1)

    scaler = MinMaxScaler().fit_reader(reader.train)
    train_generator = TorchGenerator(reader=reader.train,
                                     scaler=scaler,
                                     batch_size=2,
                                     deep_supervision=True,
                                     shuffle=True)
    val_generator = TorchGenerator(reader=reader.val,
                                   scaler=scaler,
                                   batch_size=2,
                                   deep_supervision=True,
                                   shuffle=True)

    model_path = Path(TEMP_DIR, "torch_lstm")
    model_path.mkdir(parents=True, exist_ok=True)
    model = LSTMNetwork(10,
                        0.2,
                        59,
                        recurrent_dropout=0.,
                        output_dim=1,
                        depth=2,
                        deep_supervision=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.compile(optimizer=optimizer, loss=criterion)
    # Example training loop
    history = model.fit(train_generator=train_generator, val_generator=val_generator, epochs=40)
    print(history)
