from decimal import Decimal
import numpy as np
from typeguard import typechecked
from typing import Union

@typechecked
def col_to_dec_arr(collection: Union[np.ndarray, list]) -> np.ndarray:
    """
    This helper method maps collections (such as arrays or lists) into 
    an np.np.ndarray of Decimal objects for easy operations and high-fidelity
    numbers.

    Args:
        collection (np.np.ndarray | list): A sequential collection of numbers.
    Returns:
        np.np.ndarray[Decimal]: A numpy array with Decimal objects.
    """
    def safe_decimal(x):
        return Decimal(str(x))
    
    if isinstance(collection, list):
        return np.array([safe_decimal(x) for x in collection], dtype=object)
    elif all(isinstance(x, Decimal) for x in collection.flat):
        return collection
    else:
        return np.vectorize(safe_decimal)(collection)
    
def dec_arr_to_float_arr(decimal_array: np.ndarray) -> np.ndarray:
    """ Converts decimal arrays into float arrays for use with external libraries. """
    return np.array(decimal_array.tolist(), dtype=np.float64)