from decimal import Decimal
import numpy as np
from typeguard import typechecked

@typechecked
def col_to_dec_arr(collection: np.ndarray | list) -> np.ndarray[Decimal]:
    """
    This helper method maps collections (such as arrays or lists) into 
    an np.ndarray of Decimal objects for easy operations and high-fidelity
    numbers.

    Args:
        collection (np.ndarray | list): A sequential collection of numbers.
    Returns:
        np.ndarray[Decimal]: A numpy array with Decimal objects.
    """
    def safe_decimal(x):
        return Decimal(str(x))
    
    if isinstance(collection, list):
        return np.array([safe_decimal(x) for x in collection], dtype=object)
    elif all(isinstance(x, Decimal) for x in collection.flat):
        return collection
    else:
        return np.vectorize(safe_decimal)(collection)
    
def dec_arr_to_float_arr(decimal_array: np.ndarray[Decimal]) -> np.ndarray[float]:
    """ Converts decimal arrays into float arrays for use with external libraries. """
    return np.vectorize(float)(decimal_array)