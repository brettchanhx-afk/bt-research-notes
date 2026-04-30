from typing import Union

import numpy as np


def identity(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return x


def tanh(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.tanh(x)


def sign(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.where(x > 0.0, 1, 0)


def ReLU(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return sign(x) * x


def Exp(x:Union[float,np.ndarray])->Union[float, np.ndarray]:
    
    return np.exp(x)