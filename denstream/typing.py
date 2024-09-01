from typing import Optional, TypedDict

import numpy as np

FloatArrayType = np.typing.NDArray[np.float32]
IntArrayType = np.typing.NDArray[np.int32]


class InputDict(TypedDict):
    feature_array: FloatArrayType
    time: int
    label: Optional[int]


class MetricsDict(TypedDict):
    name: str
    value: float


class UpdateParameters(TypedDict, total=False):
    time: int
    cf1_score: FloatArrayType
    weight: FloatArrayType
