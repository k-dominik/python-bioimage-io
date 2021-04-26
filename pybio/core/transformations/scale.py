from pybio.core.transformations import TensorTransformation


class ScaleRange(TensorTransformation):
    def __init__(
        self,
        *,
        gain: float,
        offset: float,
        output_min: float,
        output_max: float,
        data_min: Optional[float] = None,
        data_max: Optional[float] = None,
        minimal_data_range=1.0e-6,
        **super_kwargs,
    ):
        if data_min is not None and data_max is not None:
            assert data_min < data_max, (data_min, data_max)
        assert isinstance(apply_to, int), type(apply_to)
        super().__init__(apply_to=(apply_to,), **super_kwargs)

        self.output_min = output_min
        self.output_max = output_max
        self.data_min = data_min
        self.data_max = data_max
        self.minimal_data_range = minimal_data_range

    def apply_to_chosen(self, array: Tensor) -> Tensor:
        data_min = array.min() if self.data_min is None else self.data_min
        data_max = array.max() if self.data_max is None else self.data_max
        data_range = max(self.minimal_data_range, data_max - data_min)
        ret = array.astype("float32")
        ret -= data_min
        ret /= data_range
        output_range = self.output_max - self.output_min
        ret *= output_range
        ret += self.output_min
        return ret
