import numpy as np
import pandas as pd
from utils.numeric import is_numerical
from typing import Iterable, Union, Dict


def get_iterable_dtype(iterable: Iterable):
    """
    Determines the data type (dtype) of elements within an iterable.

    This function analyzes the first element of the given iterable to determine its dtype.
    If the iterable contains nested iterables, it will recursively check for the dtype of the 
    innermost elements.

    Parameters
    ----------
    iterable : Iterable
        The iterable whose dtype needs to be determined.

    Returns
    -------
    dtype : type or None
        The dtype of the first element in the iterable. If the iterable is empty, returns None.
    
    Examples
    --------
    >>> get_iterable_dtype([1, 2, 3])
    <class 'numpy.int64'>
    >>> get_iterable_dtype([[1.1, 2.2], [3.3, 4.4]])
    <class 'numpy.float32'>
    """
    if not len(iterable):
        return None
    if isinstance(iterable[0], (float, np.float32, np.float64)):
        dtype = np.float32
    elif isinstance(iterable[0], (int, np.int64, np.int32, bool)):
        dtype = np.int64
    elif isinstance(iterable[0], np.ndarray):
        dtype = iterable[0].dtype
    elif isinstance(iterable[0], (list, tuple)):
        dtype = get_iterable_dtype(iterable[0])
    elif isinstance(iterable[0], pd.DataFrame):
        dtype = iterable[0].dtypes.iloc[0]
    else:
        raise RuntimeError(f"Could not resolve iterable dtypes! Iterable is {iterable}")
    return dtype


def zeropad_samples(data: np.ndarray, length: int = None, axis: int = 0) -> np.ndarray:
    """
    Pads each sample in a collection of arrays along a specified axis to a uniform length.

    If the arrays have varying lengths along the specified axis, this function pads them with
    zeros so that all arrays in the input `data` have the same length. The dtype of the input 
    data is conserved.

    Parameters
    ----------
    data : np.ndarray
        A collection of arrays that need to be padded. Typically, these are 2D or 3D arrays.
    length : int, optional
        The target length to pad the arrays to. If not provided, the length of the largest array 
        along the specified axis will be used as the target length.
    axis : int, optional
        The axis along which padding should be applied. Default is 0.

    Returns
    -------
    np.ndarray
        An array with padded samples, ensuring all samples have the same length along the specified axis.

    Examples
    --------
    >>> data = [np.array([1, 2]), np.array([3])]
    >>> zeropad_samples(data)
    array([[[1., 2.]],
           [[3., 0.]]], dtype=float32)
    """
    if length is None:
        length = max([x.shape[axis] for x in data])
    dtype = get_iterable_dtype(data)
    ret = [np.concatenate([
           x,
           np.zeros(x.shape[:axis] + (length - x.shape[axis],) + x.shape[axis + 1:],\
                dtype=dtype)
           ],
           axis=axis,
           dtype=dtype) for x in data]
    if len(data[0].shape) == 3:
        return np.concatenate(ret)
    return np.atleast_3d(np.array(ret, dtype=dtype))


def _transform_array(arr: np.ndarray, preserve_dtype=True):
    """Listifies an array only along the first dimension
    """
    # Check the shape of the array
    if (len(arr.shape) > 1 and arr.shape[1] == 1) or len(arr.shape) < 2:
        # If second dimension is 1, convert it to a list of integers
        if preserve_dtype:
            return list(arr.flatten())
        return arr.flatten().tolist()
    else:
        # If second dimension is greater than 1, convert to a list of NumPy arrays
        return [row for row in arr]


def is_allnan(data: Union[pd.DataFrame, pd.Series, np.ndarray]):
    """
    Check if all elements in the data are NaN.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, np.ndarray]
        The data to check.

    Returns
    -------
    bool
        True if all elements are NaN, False otherwise.
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.isna().all().all()
    elif isinstance(data, np.ndarray):
        return np.isnan(data).all()
    else:
        raise TypeError("Input must be a pandas DataFrame, Series, or numpy array.")


def isiterable(obj):
    """
    Check if an object is iterable.

    Parameters
    ----------
    obj : object
        The object to check.

    Returns
    -------
    bool
        True if the object is iterable, False otherwise.
    """
    return hasattr(obj, '__iter__')


def is_colwise_numerical(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Check if each column in a DataFrame is numerical.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check.

    Returns
    -------
    dict
        A dictionary with column names as keys and boolean values indicating if the column is numerical.
    """
    return {col: is_numerical(df[[col]]) for col in df.columns}
