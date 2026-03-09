from enum import Enum


class StrEnum(str, Enum):
    def __str__(self):
        return self.value


class VideoSamplingMethod(StrEnum):
    I_FRAME = "i_frame"
    THUMBNAIL = "thumbnail"
    SIMPLE = "simple"
