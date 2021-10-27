from enum import Enum


class Flags(Enum):
    CAPTCHA_APPEAR = 2
    CAPTCHA_NONE = -2

    CHECK_ROD_OK = 3
    CHECK_ROD_BROK = -3

    STOP_FISHING = -100

    FIND_IMG_ERROR = -99

    TRUE = 1
    FALSE = 0

    REGION_SCREEN_SHOT_ERROR = -98
    PIXEL_SCREEN_SHOT_ERROR = -97
