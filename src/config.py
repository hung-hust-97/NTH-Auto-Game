import configparser
import cv2
from threading import Lock
from src.Base64Image import *
import datetime
import logging as log
from copy import deepcopy

HIDE_TEXT_BOX_STYLE = "border: 0px; background-color: rgba(0, 0, 0, 10);"
BUTTON_COLOR = "background-color: rgb(182, 227, 199)"

# Statics config cho size 960x540
DEFAULT_EMULATOR_SIZE = [960, 540]
RADIUS_FISHING_REGION = 150
OPEN_BACKPACK_POS = [925, 275]
CLOSE_BACKPACK_POS = [400, 300]
TOOLS_POS = [690, 60]
CASTING_ROD_POS = [765, 330]
PULLING_ROD_POS = [840, 430]
PRESERVATION_BUTTON_POS = [750, 425]
FISHING_RESULT_REGION = [622, 90, 46, 14]

CONFIRM_BUTTON_POS = [485, 410]
OK_BUTTON_POS = [485, 410]
LIST_FISHING_ROD_POS = [[0, 0], [580, 260], [730, 260], [880, 260], [580, 450], [730, 450], [880, 450]]

BACKPACK_REC = [40, 40]
CHECK_TYPE_FISH_POS = [770, 220]
FISH_IMG_REGION = [625, 42, 295, 295]
FONT_SCALE_DEFAULT = 1
MAX_CONTOUR = 3000
MIN_CONTOUR = 100
LIST_CAPTCHA_REGION = [[255, 122, 120, 120],
                       [525, 112, 90, 90], [625, 112, 90, 90], [725, 112, 90, 90],
                       [525, 212, 90, 90], [625, 212, 90, 90], [725, 212, 90, 90],
                       [525, 312, 90, 90], [625, 312, 90, 90], [725, 312, 90, 90]]
OK_CAPTCHA_POS = [480, 460]
OK_CAPTCHA_REC = [60, 60]
OK_CAPTCHA_COMPLETE = [480, 400]
REFRESH_CAPTCHA = [200, 470]
MARK_PIXEL_DISTANCE = 14
MARK_PIXEL_RADIUS = 50


# Tương tự như C++ get con trỏ Object Config
class SingletonMeta(type):
    __instance = {}
    __mutex = Lock()

    def __call__(cls, *args, **kwargs):
        with cls.__mutex:
            if cls not in cls.__instance:
                instance = super().__call__(*args, **kwargs)
                cls.__instance[cls] = instance
        return cls.__instance[cls]


class Config(metaclass=SingletonMeta):
    def __init__(self):
        self.mDateTime = str(datetime.datetime.now()).replace(":", "-").replace(" ", "_").split(".")[0]
        self.__mMutex = Lock()

        self.mConfigPath = 'config/config.ini'
        self.mConfigParser = configparser.ConfigParser()
        self.mConfigParser.read(self.mConfigPath)
        self.mConfig = self.mConfigParser['CONFIG']

        self.mLogPath = f'log/{self.mDateTime}.log'
        # self.mLogPath = f'log/log.log'
        self.mAdbPath = 'adb/adb.exe'

        self.mWindowRatio = 1
        self.mAdbHost = "127.0.0.1"
        self.mAdbPort = 5037
        self.mWindowName = self.mConfig['window_name']
        self.mEmulatorSizeId = self.mConfig.getint('emulator_size_id')
        self.mSendKeyCheck = self.mConfig.getboolean('send_key')
        self.mFishingPeriod = self.mConfig.getint('fishing_period')
        self.mWaitingMarkTime = self.mConfig.getint('waiting_mark_time')
        self.mWaitingFishTime = self.mConfig.getint('waiting_fish_time')
        self.mFishDetectionCheck = self.mConfig.getboolean('fish_detection')
        self.mFishSize = self.mConfig.getint('fish_size')
        self.mFishingRodIndex = self.mConfig.getint('fishing_rod_id')
        self.mDelayTime = self.mConfig.getfloat('delay_time')
        self.mLicense = self.mConfig.get('license')
        self.mDebugMode = self.mConfig.getboolean('debug_mode')
        self.mVersion = self.mConfig.get('version')

        self.mListBackpackImgPath = ['data/backpack1280.png',
                                     'data/backpack960.png',
                                     'data/backpack640.png']

        self.mListPreservationImgPath = ['data/preservation1280.png',
                                         'data/preservation960.png',
                                         'data/preservation640.png']

        self.mListOKCaptchaImgPath = ['data/okcaptcha1280.png',
                                      'data/okcaptcha960.png',
                                      'data/okcaptcha640.png']

        self.mBackpackImg = None
        self.mOKCaptchaImg = None
        self.mPreservationImg = None
        self.mLdLogo = None
        self.mNoxLogo = None
        self.mMemuLogo = None

        self.mListEmulatorSize = [[1280, 720], [960, 540], [640, 360]]
        self.mStrListEmulatorSize = ['1280x720 dpi 240', '960x540 dpi 160', '640x360 dpi 120']
        self.mListBlurArg = [15, 5, 3]

        self.mAppTitle = "NTH Auto Game " + self.mVersion
        self.mLicenseText = "Bấm vào Youtube, Facebook để liên hệ tác giả"
        self.mFacebookLink = "https://www.facebook.com/groups/4478925988809953"
        self.mYoutubeLink = "https://www.youtube.com/channel/UCaEW8YUslMbGv3839jzdQ6g/featured"
        self.mMediaFire = "https://www.mediafire.com/folder/zpq42aonjn4bh/Auto_Game_PlayTogether"
        self.mWaitStatus = "Auto đang đóng chu trình câu\nVui lòng đợi trong giây lát"
        self.mAppLogo = LOGO_NTH_AUTO_GAME
        self.mIcon = ICON_NTH_AUTO_GAME

        self.mYoutubeImgPath = 'data/youtube.png'
        self.mFacebookImgPath = 'data/facebook.png'

        self.mConfidence = 0.7
        self.mShutdownCheckBox = False
        self.mShutdownTime = 0
        self.mThickness = 1
        self.mAdbAddress = "None"

        # Cac gia tri nay se thay doi theo size cua emulator
        self.mBackpackImgPath = self.mListBackpackImgPath[self.mEmulatorSizeId]
        self.mPreservationImgPath = self.mListPreservationImgPath[self.mEmulatorSizeId]
        self.mOKCaptchaImgPath = self.mListOKCaptchaImgPath[self.mEmulatorSizeId]
        self.mBlur = self.mListBlurArg[self.mEmulatorSizeId]
        self.mListFishingRodPosition = deepcopy(LIST_FISHING_ROD_POS)
        self.mListCaptchaRegion = deepcopy(LIST_CAPTCHA_REGION)
        self.mRadiusFishingRegion = RADIUS_FISHING_REGION
        self.mOpenBackPack = deepcopy(OPEN_BACKPACK_POS)
        self.mCloseBackPack = deepcopy(CLOSE_BACKPACK_POS)
        self.mTools = deepcopy(TOOLS_POS)
        self.mConfirm = deepcopy(CONFIRM_BUTTON_POS)
        self.mOKButton = deepcopy(OK_BUTTON_POS)

        self.mPreservationPos = deepcopy(PRESERVATION_BUTTON_POS)
        self.mFishingResultRegion = deepcopy(FISHING_RESULT_REGION)

        self.mCastingRodPos = deepcopy(CASTING_ROD_POS)
        self.mPullingRodPos = deepcopy(PULLING_ROD_POS)
        self.mBackpackRec = deepcopy(BACKPACK_REC)
        self.mBackpackRegion = [self.mOpenBackPack[0] - self.mBackpackRec[0] // 2,
                                self.mOpenBackPack[1] - self.mBackpackRec[1] // 2,
                                self.mBackpackRec[0],
                                self.mBackpackRec[0]]
        self.mCheckTypeFishPos = deepcopy(CHECK_TYPE_FISH_POS)
        self.mFishImgRegion = deepcopy(FISH_IMG_REGION)
        self.mFontScale = FONT_SCALE_DEFAULT
        self.mEmulatorSize = deepcopy(DEFAULT_EMULATOR_SIZE)

        self.mOKCaptchaPos = deepcopy(OK_CAPTCHA_POS)
        self.mOKCaptchaRec = deepcopy(OK_CAPTCHA_REC)
        self.mOKCaptchaRegion = [self.mOKCaptchaPos[0] - self.mOKCaptchaRec[0] // 2,
                                 self.mOKCaptchaPos[1] - self.mOKCaptchaRec[1] // 2,
                                 self.mOKCaptchaRec[0],
                                 self.mOKCaptchaRec[0]]
        self.mOKCaptchaComplete = deepcopy(OK_CAPTCHA_COMPLETE)
        self.mRefreshCaptcha = deepcopy(REFRESH_CAPTCHA)
        self.mMarkPixelDist = MARK_PIXEL_DISTANCE
        self.mMarkPixelRadius = MARK_PIXEL_RADIUS

        # Cac tham so detect ca
        self.mMaxContour = MAX_CONTOUR
        self.mMinContour = MIN_CONTOUR

        # RGB in QT
        self.mVioletColorRGB = [231, 147, 232]
        self.mBlueColorRGB = [89, 198, 217]
        self.mGreenColorRGB = [163, 228, 103]
        self.mGrayColorRGB = [228, 224, 197]
        self.mTextColor = (255, 255, 255)
        # RGB in openCV is BGR
        self.mVioletColorBGR = [232, 147, 231]
        self.mBlueColorBGR = [217, 198, 89]
        self.mGreenColorBGR = [103, 228, 163]
        self.mGrayColorBGR = [197, 224, 228]

    def __del__(self):
        pass

    def SetEmulatorSize(self, mEmulatorSizeId: int):
        self.__mMutex.acquire()
        self.mEmulatorSizeId = mEmulatorSizeId
        self.mEmulatorSize = self.mListEmulatorSize[mEmulatorSizeId]
        self.mPreservationImgPath = self.mListPreservationImgPath[mEmulatorSizeId]
        self.mBackpackImgPath = self.mListBackpackImgPath[mEmulatorSizeId]
        self.mOKCaptchaImgPath = self.mListOKCaptchaImgPath[mEmulatorSizeId]
        self.mBlur = self.mListBlurArg[mEmulatorSizeId]

        self.mWindowRatio = self.mEmulatorSize[0] / DEFAULT_EMULATOR_SIZE[0]
        self.mRadiusFishingRegion = int(RADIUS_FISHING_REGION * self.mWindowRatio)

        self.mTools = [int(TOOLS_POS[0] * self.mWindowRatio),
                       int(TOOLS_POS[1] * self.mWindowRatio)]
        self.mCastingRodPos = [int(CASTING_ROD_POS[0] * self.mWindowRatio),
                               int(CASTING_ROD_POS[1] * self.mWindowRatio)]
        self.mPullingRodPos = [int(PULLING_ROD_POS[0] * self.mWindowRatio),
                               int(PULLING_ROD_POS[1] * self.mWindowRatio)]

        self.mConfirm = [int(CONFIRM_BUTTON_POS[0] * self.mWindowRatio),
                         int(CONFIRM_BUTTON_POS[1] * self.mWindowRatio)]
        self.mOKButton = [int(OK_BUTTON_POS[0] * self.mWindowRatio),
                          int(OK_BUTTON_POS[1] * self.mWindowRatio)]

        self.mOKCaptchaComplete = [int(OK_CAPTCHA_COMPLETE[0] * self.mWindowRatio),
                                   int(OK_CAPTCHA_COMPLETE[1] * self.mWindowRatio)]
        self.mRefreshCaptcha = [int(REFRESH_CAPTCHA[0] * self.mWindowRatio),
                                int(REFRESH_CAPTCHA[1] * self.mWindowRatio)]
        self.mMarkPixelDist = int(MARK_PIXEL_DISTANCE * self.mWindowRatio)
        self.mMarkPixelRadius = int(MARK_PIXEL_RADIUS * self.mWindowRatio)

        for i in range(1, len(self.mListFishingRodPosition)):
            for j in range(2):
                self.mListFishingRodPosition[i][j] = int(LIST_FISHING_ROD_POS[i][j] * self.mWindowRatio)

        for i in range(len(self.mListCaptchaRegion)):
            for j in range(4):
                self.mListCaptchaRegion[i][j] = int(LIST_CAPTCHA_REGION[i][j] * self.mWindowRatio)

        self.mPreservationPos = [int(PRESERVATION_BUTTON_POS[0] * self.mWindowRatio),
                                 int(PRESERVATION_BUTTON_POS[1] * self.mWindowRatio)]
        for i in range(4):
            self.mFishingResultRegion[i] = int(FISHING_RESULT_REGION[i] * self.mWindowRatio)

        self.mOpenBackPack = [int(OPEN_BACKPACK_POS[0] * self.mWindowRatio),
                              int(OPEN_BACKPACK_POS[1] * self.mWindowRatio)]
        self.mCloseBackPack = [int(CLOSE_BACKPACK_POS[0] * self.mWindowRatio),
                               int(CLOSE_BACKPACK_POS[1] * self.mWindowRatio)]
        self.mBackpackRec = [int(BACKPACK_REC[0] * self.mWindowRatio),
                             int(BACKPACK_REC[1] * self.mWindowRatio)]
        self.mBackpackRegion = [self.mOpenBackPack[0] - self.mBackpackRec[0] // 2,
                                self.mOpenBackPack[1] - self.mBackpackRec[1] // 2,
                                self.mBackpackRec[0],
                                self.mBackpackRec[0]]

        self.mOKCaptchaPos = [int(OK_CAPTCHA_POS[0] * self.mWindowRatio),
                              int(OK_CAPTCHA_POS[1] * self.mWindowRatio)]
        self.mOKCaptchaRec = [int(OK_CAPTCHA_REC[0] * self.mWindowRatio),
                              int(OK_CAPTCHA_REC[1] * self.mWindowRatio)]
        self.mOKCaptchaRegion = [self.mOKCaptchaPos[0] - self.mOKCaptchaRec[0] // 2,
                                 self.mOKCaptchaPos[1] - self.mOKCaptchaRec[1] // 2,
                                 self.mOKCaptchaRec[0],
                                 self.mOKCaptchaRec[0]]

        self.mCheckTypeFishPos = [int(CHECK_TYPE_FISH_POS[0] * self.mWindowRatio),
                                  int(CHECK_TYPE_FISH_POS[1] * self.mWindowRatio)]

        for i in range(4):
            self.mFishImgRegion[i] = int(FISH_IMG_REGION[i] * self.mWindowRatio)

        self.mFontScale = FONT_SCALE_DEFAULT * self.mWindowRatio
        self.mMinContour = MIN_CONTOUR * self.mWindowRatio
        self.mMaxContour = MAX_CONTOUR * self.mWindowRatio

        self.mListBackpackImgPath = ['data/backpack1280.png',
                                     'data/backpack960.png',
                                     'data/backpack640.png']

        self.mListPreservationImgPath = ['data/preservation1280.png',
                                         'data/preservation960.png',
                                         'data/preservation640.png']

        self.mListOKCaptchaImgPath = ['data/okcaptcha1280.png',
                                      'data/okcaptcha960.png',
                                      'data/okcaptcha640.png']

        self.mBackpackImg = cv2.imread(self.mBackpackImgPath, cv2.IMREAD_GRAYSCALE)
        self.mPreservationImg = cv2.imread(self.mPreservationImgPath, cv2.IMREAD_GRAYSCALE)
        self.mOKCaptchaImg = cv2.imread(self.mOKCaptchaImgPath, cv2.IMREAD_GRAYSCALE)
        self.mNoxLogo = cv2.imread('data/noxlogo.png', cv2.IMREAD_GRAYSCALE)
        self.mLdLogo = cv2.imread('data/ldlogo.png', cv2.IMREAD_GRAYSCALE)
        self.mMemuLogo = cv2.imread('data/memulogo.png', cv2.IMREAD_GRAYSCALE)

        if self.mWindowRatio > 1:
            self.mThickness = 2
        else:
            self.mThickness = 1

        log.info(f'mEmulatorSizeId = {self.mEmulatorSizeId}')
        log.info(f'mEmulatorSize = {self.mEmulatorSize}')
        log.info(f'mPreservationImgPath = {self.mPreservationImgPath}')
        log.info(f'mBackpackImgPath = {self.mBackpackImgPath}')
        log.info(f'mBlur = {self.mBlur}')
        log.info(f'mWindowRatio = {self.mWindowRatio}')
        log.info(f'mRadiusFishingRegion = {self.mRadiusFishingRegion}')
        log.info(f'mOpenBackPack = {self.mOpenBackPack}')
        log.info(f'mCloseBackPack = {self.mCloseBackPack}')
        log.info(f'mCastingRodPos = {self.mCastingRodPos}')
        log.info(f'mPullingRodPos = {self.mPullingRodPos}')
        log.info(f'mPreservationPos = {self.mPreservationPos}')
        log.info(f'mConfirm = {self.mConfirm}')
        log.info(f'mOKButton = {self.mOKButton}')

        for i in range(1, len(self.mListFishingRodPosition)):
            log.info(f'mListFishingRodPosition{i} = {self.mListFishingRodPosition[i]}')

        for i in range(len(self.mListCaptchaRegion)):
            log.info(f'mListCaptchaRegion{i} = {self.mListCaptchaRegion[i]}')

        log.info(f'mFishingResultRegion = {self.mFishingResultRegion}')
        log.info(f'mBackpackRec = {self.mBackpackRec}')
        log.info(f'mCheckTypeFishPos = {self.mCheckTypeFishPos}')
        log.info(f'mFishImgRegion = {self.mFishImgRegion}')
        log.info(f'mFontScale = {self.mFontScale}')
        log.info(f'mMinContour = {self.mMinContour}')
        log.info(f'mMaxContour = {self.mMaxContour}')

        self.__mMutex.release()

    def SetDelayTime(self, mDelayTime: float):
        self.__mMutex.acquire()
        self.mDelayTime = mDelayTime
        self.__mMutex.release()

    def SetAdbAddress(self, mAdbAddress: str):
        self.__mMutex.acquire()
        self.mAdbAddress = mAdbAddress
        self.__mMutex.release()

    def SetShutdownCheckBox(self, mShutdownCheckBox: bool):
        self.__mMutex.acquire()
        self.mShutdownCheckBox = mShutdownCheckBox
        self.__mMutex.release()

    def SetShutdownTime(self, mShutdownTime: int):
        self.__mMutex.acquire()
        self.mShutdownTime = mShutdownTime
        self.__mMutex.release()

    def SetWindowName(self, mWindowName: str):
        self.__mMutex.acquire()
        self.mWindowName = mWindowName
        self.__mMutex.release()

    def SetFishingPeriod(self, mFishingPeriod: int):
        self.__mMutex.acquire()
        self.mFishingPeriod = mFishingPeriod
        self.__mMutex.release()

    def SetWaitingFishTime(self, mWaitingFishTime: int):
        self.__mMutex.acquire()
        self.mWaitingFishTime = mWaitingFishTime
        self.__mMutex.release()

    def SetWaitingMarkTime(self, mWaitingMarkTime: int):
        self.__mMutex.acquire()
        self.mWaitingMarkTime = mWaitingMarkTime
        self.__mMutex.release()

    def SetFishDetection(self, mFishDetectionCheck: bool):
        self.__mMutex.acquire()
        self.mFishDetectionCheck = mFishDetectionCheck
        self.__mMutex.release()

    def SetFishSize(self, mFishSize: int):
        self.__mMutex.acquire()
        self.mFishSize = mFishSize
        self.__mMutex.release()

    def SetFishingRod(self, mFishingRod: int):
        self.__mMutex.acquire()
        self.mFishingRodIndex = mFishingRod
        self.__mMutex.release()

    def SetSendKey(self, mSendKey: bool):
        self.__mMutex.acquire()
        self.mSendKeyCheck = mSendKey
        self.__mMutex.release()

    def SaveConfig(self):
        mNewConfig = configparser.ConfigParser()
        mNewConfig['CONFIG'] = {}
        mNewConfig['CONFIG']['window_name'] = self.mWindowName
        mNewConfig['CONFIG']['emulator_size_id'] = str(self.mEmulatorSizeId)
        mNewConfig['CONFIG']['fishing_period'] = str(self.mFishingPeriod)
        mNewConfig['CONFIG']['waiting_fish_time'] = str(self.mWaitingFishTime)
        mNewConfig['CONFIG']['waiting_mark_time'] = str(self.mWaitingMarkTime)
        mNewConfig['CONFIG']['fish_detection'] = str(self.mFishDetectionCheck)
        mNewConfig['CONFIG']['send_key'] = str(self.mSendKeyCheck)
        mNewConfig['CONFIG']['fish_size'] = str(self.mFishSize)
        mNewConfig['CONFIG']['fishing_rod_id'] = str(self.mFishingRodIndex)
        mNewConfig['CONFIG']['delay_time'] = str(self.mDelayTime)
        mNewConfig['CONFIG']['license'] = self.mConfig.get("license")
        mNewConfig['CONFIG']['debug_mode'] = self.mConfig.get('debug_mode')
        mNewConfig['CONFIG']['version'] = self.mConfig.get('version')

        with open(self.mConfigPath, 'w') as mConfigFile:
            mNewConfig.write(mConfigFile)
