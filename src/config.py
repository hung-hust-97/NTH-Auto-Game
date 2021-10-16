import configparser
import os
from threading import Lock
from src.Base64Image import *
import datetime
import logging as log

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
CONFIRM_BUTTON_POS = [485, 410]
OK_BUTTON_POS = [485, 410]
LIST_FISHING_ROD_POS = [[0, 0], [580, 260], [730, 260], [880, 260], [580, 450], [730, 450], [880, 450]]
PRESERVATION_REC = [280, 80]
BACKPACK_REC = [40, 40]
CHECK_TYPE_FISH_POS = [770, 220]
FISH_IMG_REGION = [625, 42, 295, 295]
FONT_SCALE_DEFAULT = 1
MAX_CONTOUR = 3500
MIN_CONTOUR = 100


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
        self.mCurrentPath = os.getcwd()
        self.mDataPath = f'{self.mCurrentPath}\\data\\'
        self.mConfigPath = f'{self.mCurrentPath}\\config\\config.ini'
        self.mConfigParser = configparser.ConfigParser()
        self.mConfigParser.read(self.mConfigPath)
        self.mConfig = self.mConfigParser['CONFIG']

        self.mLogPath = f'{self.mCurrentPath}\\log\\{self.mDateTime}.log'
        self.mAdbPath = f'{self.mCurrentPath}\\adb\\adb.exe'
        self.mWindowRatio = 1
        self.mAdbHost = "127.0.0.1"
        self.mAdbPort = 5037
        self.mWindowName = self.mConfig['window_name']
        self.mEmulatorSizeId = self.mConfig.getint('emulator_size_id')
        self.mFreeMouseCheck = self.mConfig.getboolean('free_mouse')
        self.mWaitingFishTime = self.mConfig.getint('waiting_fish_time')
        self.mPullingFishTime = self.mConfig.getint('pulling_fish_time')
        self.mFishDetectionCheck = self.mConfig.getboolean('fish_detection')
        self.mShowFishCheck = self.mConfig.getboolean('show_fish')
        self.mFishSize = self.mConfig.getint('fish_size')
        self.mFishingRodIndex = self.mConfig.getint('fishing_rod_id')
        self.mDelayTime = self.mConfig.getfloat('delay_time')
        self.mLicense = self.mConfig.get('license')
        self.mDebugMode = self.mConfig.getboolean('debug_mode')
        self.mVersion = self.mConfig.get('version')

        self.mListBackpackImgPath = [f'{self.mDataPath}backpack1280.png',
                                     f'{self.mDataPath}backpack960.png',
                                     f'{self.mDataPath}backpack640.png',
                                     f'{self.mDataPath}backpack480.png']

        self.mListPreservationImgPath = [f'{self.mDataPath}preservation1280.png',
                                         f'{self.mDataPath}preservation960.png',
                                         f'{self.mDataPath}preservation640.png',
                                         f'{self.mDataPath}preservation480.png']

        self.mListEmulatorSize = [[1280, 720], [960, 540], [640, 360], [480, 270]]
        self.mListStrEmulatorSize = ["1280x720", "960x540", "640x360", "480x270"]
        self.mListBlurArg = [19, 7, 5, 3]

        self.mAppTitle = "NTH Auto Game Play Together"
        self.mLicenseText = "Để cập nhật thông tin mới nhất và góp ý cho tác giả, vui lòng bấm vào biểu tượng Youtube hoặc Facebook tại đây"
        self.mFacebookLink = "https://www.facebook.com/NTH-Auto-Game-106164441850257"
        self.mYoutubeLink = "https://www.youtube.com/channel/UCaEW8YUslMbGv3839jzdQ6g/featured"
        self.mAppLogo = LOGO_NTH_AUTO_GAME
        self.mIcon = ICON_NTH_AUTO_GAME

        self.mYoutubeImgPath = f'{self.mDataPath}youtube.png'
        self.mFacebookImgPath = f'{self.mDataPath}facebook.png'
        self.mDifferentColor = 10
        self.mConfidence = 0.7
        self.mShutdownCheckBox = False
        self.mShutdownTime = 0
        self.mThickness = 1
        self.mAdbAddress = "None"

        # Cac gia tri nay se thay doi theo size cua emulator
        self.mBackpackImgPath = self.mListBackpackImgPath[self.mEmulatorSizeId]
        self.mPreservationImgPath = self.mListPreservationImgPath[self.mEmulatorSizeId]
        self.mBlur = self.mListBlurArg[self.mEmulatorSizeId]
        self.mListFishingRodPosition = [[0, 0], [580, 260], [730, 260], [880, 260], [580, 450], [730, 450], [880, 450]]
        self.mRadiusFishingRegion = RADIUS_FISHING_REGION
        self.mOpenBackPack = OPEN_BACKPACK_POS
        self.mCloseBackPack = CLOSE_BACKPACK_POS
        self.mTools = TOOLS_POS
        self.mCastingRodPos = CASTING_ROD_POS
        self.mPullingRodPos = PULLING_ROD_POS
        self.mPreservationPos = PRESERVATION_BUTTON_POS
        self.mConfirm = CONFIRM_BUTTON_POS
        self.mOKButton = OK_BUTTON_POS
        self.mPreservationRec = PRESERVATION_REC
        self.mBackpackRec = BACKPACK_REC
        self.mCheckTypeFishPos = CHECK_TYPE_FISH_POS
        self.mFishImgRegion = FISH_IMG_REGION
        self.mFontScale = FONT_SCALE_DEFAULT
        self.mEmulatorSize = DEFAULT_EMULATOR_SIZE

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
        self.mBlur = self.mListBlurArg[mEmulatorSizeId]

        self.mWindowRatio = self.mEmulatorSize[0] / DEFAULT_EMULATOR_SIZE[0]
        self.mRadiusFishingRegion = int(RADIUS_FISHING_REGION * self.mWindowRatio)
        self.mOpenBackPack = [int(OPEN_BACKPACK_POS[0] * self.mWindowRatio),
                              int(OPEN_BACKPACK_POS[1] * self.mWindowRatio)]
        self.mCloseBackPack = [int(CLOSE_BACKPACK_POS[0] * self.mWindowRatio),
                               int(CLOSE_BACKPACK_POS[1] * self.mWindowRatio)]
        self.mTools = [int(TOOLS_POS[0] * self.mWindowRatio),
                       int(TOOLS_POS[1] * self.mWindowRatio)]
        self.mCastingRodPos = [int(CASTING_ROD_POS[0] * self.mWindowRatio),
                               int(CASTING_ROD_POS[1] * self.mWindowRatio)]
        self.mPullingRodPos = [int(PULLING_ROD_POS[0] * self.mWindowRatio),
                               int(PULLING_ROD_POS[1] * self.mWindowRatio)]
        self.mPreservationPos = [int(PRESERVATION_BUTTON_POS[0] * self.mWindowRatio),
                                 int(PRESERVATION_BUTTON_POS[1] * self.mWindowRatio)]
        self.mConfirm = [int(CONFIRM_BUTTON_POS[0] * self.mWindowRatio),
                         int(CONFIRM_BUTTON_POS[1] * self.mWindowRatio)]
        self.mOKButton = [int(OK_BUTTON_POS[0] * self.mWindowRatio),
                          int(OK_BUTTON_POS[1] * self.mWindowRatio)]

        for i in range(1, len(self.mListFishingRodPosition)):
            self.mListFishingRodPosition[i] = [int(LIST_FISHING_ROD_POS[i][0] * self.mWindowRatio),
                                               int(LIST_FISHING_ROD_POS[i][1] * self.mWindowRatio)]
        self.mPreservationRec = [int(PRESERVATION_REC[0] * self.mWindowRatio),
                                 int(PRESERVATION_REC[1] * self.mWindowRatio)]

        self.mBackpackRec = [int(BACKPACK_REC[0] * self.mWindowRatio),
                             int(BACKPACK_REC[1] * self.mWindowRatio)]
        self.mCheckTypeFishPos = [int(CHECK_TYPE_FISH_POS[0] * self.mWindowRatio),
                                  int(CHECK_TYPE_FISH_POS[1] * self.mWindowRatio)]

        self.mFishImgRegion = [int(FISH_IMG_REGION[0] * self.mWindowRatio),
                               int(FISH_IMG_REGION[1] * self.mWindowRatio),
                               int(FISH_IMG_REGION[2] * self.mWindowRatio),
                               int(FISH_IMG_REGION[3] * self.mWindowRatio)]

        self.mFontScale = FONT_SCALE_DEFAULT * self.mWindowRatio
        self.mMinContour = MIN_CONTOUR * self.mWindowRatio
        self.mMaxContour = MAX_CONTOUR * self.mWindowRatio

        if self.mWindowRatio > 1:
            self.mThickness = 2

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
        log.info(f'mPreservationRec = {self.mPreservationRec}')
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

    def SetFreeMouse(self, mFreeMouse: bool):
        self.__mMutex.acquire()
        self.mFreeMouseCheck = mFreeMouse
        self.__mMutex.release()

    def SetWaitingFishTime(self, mWaitingFishTime: int):
        self.__mMutex.acquire()
        self.mWaitingFishTime = mWaitingFishTime
        self.__mMutex.release()

    def SetPullingFishTime(self, mPullingFishTime: int):
        self.__mMutex.acquire()
        self.mPullingFishTime = mPullingFishTime
        self.__mMutex.release()

    def SetFishDetection(self, mFishDetectionCheck: bool):
        self.__mMutex.acquire()
        self.mFishDetectionCheck = mFishDetectionCheck
        self.__mMutex.release()

    def SetShowFishShadow(self, mShowFishCheck: bool):
        self.__mMutex.acquire()
        self.mShowFishCheck = mShowFishCheck
        self.__mMutex.release()

    def SetFishSize(self, mFishSize: int):
        self.__mMutex.acquire()
        self.mFishSize = mFishSize
        self.__mMutex.release()

    def SetFishingRod(self, mFishingRod: int):
        self.__mMutex.acquire()
        self.mFishingRodIndex = mFishingRod
        self.__mMutex.release()

    def SaveConfig(self):
        mNewConfig = configparser.ConfigParser()
        mNewConfig['CONFIG'] = {}
        mNewConfig['CONFIG']['window_name'] = self.mWindowName
        mNewConfig['CONFIG']['emulator_size_id'] = str(self.mEmulatorSizeId)
        mNewConfig['CONFIG']['free_mouse'] = str(self.mFreeMouseCheck)
        mNewConfig['CONFIG']['waiting_fish_time'] = str(self.mWaitingFishTime)
        mNewConfig['CONFIG']['pulling_fish_time'] = str(self.mPullingFishTime)
        mNewConfig['CONFIG']['fish_detection'] = str(self.mFishDetectionCheck)
        mNewConfig['CONFIG']['show_fish'] = str(self.mShowFishCheck)
        mNewConfig['CONFIG']['fish_size'] = str(self.mFishSize)
        mNewConfig['CONFIG']['fishing_rod_id'] = str(self.mFishingRodIndex)
        mNewConfig['CONFIG']['delay_time'] = str(self.mDelayTime)
        mNewConfig['CONFIG']['license'] = self.mConfig.get("license")
        mNewConfig['CONFIG']['debug_mode'] = self.mConfig.get('debug_mode')
        mNewConfig['CONFIG']['version'] = self.mConfig.get('version')

        with open(self.mConfigPath, 'w') as mConfigFile:
            mNewConfig.write(mConfigFile)
