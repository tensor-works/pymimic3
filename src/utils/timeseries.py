import numpy as np
import pandas as pd
from metrics import CustomBins, LogBins
from utils.arrays import _transform_array
from typing import List, Tuple, Union


def read_timeseries(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    row_only=False,
    bining="none",
    one_hot=False,
    dtype=np.ndarray,
    preserve_dtype: bool = True
) -> Tuple[List[Union[np.ndarray, pd.DataFrame]], \
           List[Union[np.ndarray, pd.DataFrame]], \
           List[Union[np.ndarray, pd.DataFrame]]]:
    """Read sample and label frames time step by time step and return as list.

    Args:
        X_df (pd.DataFrame): Sample df or array.
        y_df (pd.DataFrame): Target df or array
        row_only (bool, optional): Return only the current sample row vs. the entire preceding data frame. 
            Defaults to False.
        bining (str, optional): Bining mode to apply to the labels. Can be "none", "custom" or "log". 
            Defaults to "none".
        one_hot (bool, optional): One-hot encode categorical labels. Defaults to False.
        dtype (_type_, optional): Type of the data contained in the returned list. 
            Defaults to np.ndarray.
        preserve_dtype (bool, optional): Wether to preserve the numpy dtypes in the target primitives. 
            Defaults to True.

    Returns:
        _type_: _description_
    """
    if bining == "log":
        y_df = y_df.apply(lambda x: LogBins.get_bin_log(x, one_hot=one_hot))
        if not isinstance(y_df, pd.DataFrame):
            y_df = y_df.to_frame()
    elif bining == "custom":
        y_df = y_df.apply(lambda x: CustomBins.get_bin_custom(x, one_hot=one_hot), axis=1)
        if not isinstance(y_df, pd.DataFrame):
            y_df = y_df.to_frame()

    if row_only:
        Xs = [
            X_df.loc[timestamp].values if dtype in [np.ndarray, np.array] else X_df.loc[timestamp]
            for timestamp in y_df.index
        ]
    else:
        Xs = [
            X_df.loc[:timestamp].values if dtype in [np.ndarray, np.array] else X_df.loc[:timestamp]
            for timestamp in y_df.index
        ]

    ys = _transform_array(y_df.values, preserve_dtype=preserve_dtype)
    ts = _transform_array(y_df.index.values, preserve_dtype=preserve_dtype)
    # ts = y_df.index.tolist()

    return Xs, ys, ts


def make_prediction_vector(model, generator, batches=20, bin_averages=None):
    """_summary_
    """
    # TODO! fix bin averages instead
    Xs = list()
    ys = list()

    for _ in range(batches):
        X, y = generator.next()
        Xs.append(X)
        ys.append(y)

    y_true = np.concatenate(ys)
    y_pred = np.concatenate([model.predict(X, verbose=0) for X in Xs])

    if bin_averages:
        # TODO! shape mismatch betwenn prediction vector length and length of bin averages
        y_pred = np.array([
            bin_averages[int(label)]
            if label < len(bin_averages) else bin_averages[len(bin_averages) - 1]
            for label in np.argmax(y_pred, axis=1)
        ]).reshape(1, -1)

        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)

        y_true = np.array([
            bin_averages[int(label)]
            if label < len(bin_averages) else bin_averages[len(bin_averages) - 1]
            for label in y_true
        ]).reshape(1, -1)

    # TODO! this should be y_pred y_true
    return y_pred, y_true
