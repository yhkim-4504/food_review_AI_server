from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class FoodImage(_message.Message):
    __slots__ = ["id", "image"]
    ID_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    id: int
    image: bytes
    def __init__(self, id: _Optional[int] = ..., image: _Optional[bytes] = ...) -> None: ...

class GpuStatus(_message.Message):
    __slots__ = ["status"]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: bool
    def __init__(self, status: bool = ...) -> None: ...

class PredictionResult(_message.Message):
    __slots__ = ["food_type", "probability"]
    FOOD_TYPE_FIELD_NUMBER: _ClassVar[int]
    PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    food_type: int
    probability: float
    def __init__(self, food_type: _Optional[int] = ..., probability: _Optional[float] = ...) -> None: ...
