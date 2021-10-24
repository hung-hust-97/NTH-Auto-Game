from enum import Enum


class Flags(Enum):
    CAPTCHA_APPEAR = 2
    CAPTCHA_NONE = -2

    CHECK_ROD_OK = 3
    CHECK_ROD_BROK = -3

    STOP_FISHING = -100
