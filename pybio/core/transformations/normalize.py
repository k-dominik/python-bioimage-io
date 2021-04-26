from typing import List, Optional, Sequence, Tuple, Union

from .base import TensorTransformation
from pybio.core.protocols import Tensor


class ZeroMeanUnitVariance(TensorTransformation):
    def __init__(self, *, mean: Optional[Union[float, Tensor]] = None, std: Optional[Union[float, Tensor]] = None, eps=1.0e-6):
        super().__init__()
        self.eps = eps
        if mean is None and std is not None or mean is not None and std is None:
            raise ValueError("Specify mean and std, not only one of them")

        if isinstance(mean, Tensor):
            assert isinstance(std, Tensor)
        else:
            assert isinstance(std, Tensor)

        self.mean = mean
        self.std = std

    def apply(self, tensor: Tensor) -> Tensor:
        mean = self.mean
        std = self.std
        if mean is None:
            # mode: per_sample
            assert std is None
            mean = tensor.mean()
            std = tensor.std()
        else:
            # mode: fixed
            assert std is not None

        return (tensor - mean) / (std + self.eps)
