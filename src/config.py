import configparser
import os
from threading import Lock

HIDE_TEXT_BOX_STYLE = "border: 0px; background-color: rgba(0, 0, 0, 10);"
BUTTON_COLOR = "background-color: rgb(182, 227, 199)"

# Statics config cho size 960x540
DEFAULT_EMULATOR_SIZE = [960, 540]
RADIUS_FISHING_REGION = 150
OPEN_BACKPACK_POS = [915, 290]
CLOSE_BACKPACK_POS = [400, 300]
TOOLS_POS = [690, 60]
CASTING_ROD_POS = [765, 330]
PULLING_ROD_POS = [840, 430]
PRESERVATION_BUTTON_POS = [750, 440]
CONFIRM_BUTTON_POS = [485, 410]
OK_BUTTON_POS = [485, 410]
LIST_FISHING_ROD_POS = [[0, 0], [580, 260], [730, 260], [880, 260], [580, 450], [730, 450], [880, 450]]
PRESERVATION_REC = [320, 160]
BACKPACK_REC = [100, 100]
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
        self.__mMutex = Lock()

        self.__mCurrentPath = os.getcwd()
        self.__mDataPath = f'{self.__mCurrentPath}\\data\\'
        self.__mConfigPath = f'{self.__mCurrentPath}\\config\\config.ini'
        self.__mConfigParser = configparser.ConfigParser()
        self.__mConfigParser.read(self.__mConfigPath)
        self.__mConfig = self.__mConfigParser['CONFIG']

        self.mAdbPath = f'{self.__mCurrentPath}\\adb\\adb.exe'
        self.mIconPath = f'{self.__mDataPath}iconapp.ico'
        self.mWindowRatio = 1
        self.mAdbHost = "127.0.0.1"
        self.mAdbPort = 5037
        self.mWindowName = self.__mConfig['window_name']
        self.mEmulatorSizeId = self.__mConfig.getint('emulator_size_id')
        self.mFreeMouseCheck = self.__mConfig.getboolean('free_mouse')
        self.mWaitingFishTime = self.__mConfig.getint('waiting_fish_time')
        self.mPullingFishTime = self.__mConfig.getint('pulling_fish_time')
        self.mFishDetectionCheck = self.__mConfig.getboolean('fish_detection')
        self.mShowFishCheck = self.__mConfig.getboolean('show_fish')
        self.mFishSize = self.__mConfig.getint('fish_size')
        self.mFishingRodIndex = self.__mConfig.getint('fishing_rod_id')
        self.mDelayTime = self.__mConfig.getfloat('delay_time')

        self.mListBackpackImgPath = [f'{self.__mDataPath}backpack1280.png',
                                     f'{self.__mDataPath}backpack960.png',
                                     f'{self.__mDataPath}backpack640.png',
                                     f'{self.__mDataPath}backpack480.png']
        self.mBackpackImgPath = self.mListBackpackImgPath[0]

        self.mListPreservationImgPath = [f'{self.__mDataPath}preservation1280.png',
                                         f'{self.__mDataPath}preservation960.png',
                                         f'{self.__mDataPath}preservation640.png',
                                         f'{self.__mDataPath}preservation480.png']
        self.mPreservationImgPath = self.mListPreservationImgPath[0]
        self.mYoutubeImgPath = f'{self.__mDataPath}youtube.png'
        self.mFacebookImgPath = f'{self.__mDataPath}facebook.png'
        self.mLogoImgPath = f'{self.__mDataPath}image.png'
        self.mEmulatorSize = DEFAULT_EMULATOR_SIZE
        self.mDifferentColor = 10
        self.mConfidence = 0.7
        self.mShutdownCheckBox = False
        self.mShutdownTime = 0
        self.mAdbAddress = "None"
        self.mFacebook = "https://www.facebook.com/groups/kayty"
        self.mYoutube = "https://www.youtube.com/channel/UCHMv61r6ZZwiJdNGUlLt5NQ"
        self.mAuthor = 'AutoFishing by nth'

        self.mListEmulatorSize = [[1280, 720], [960, 540], [640, 360], [480, 270]]
        self.mListStrEmulatorSize = ["1280x720", "960x540", "640x360", "480x270"]
        self.mListBlurArg = [19, 7, 5, 3]
        self.mBlur = 13

        # Cac gia tri nay se thay doi theo size cua emulator
        self.mListFishingRodPosition = LIST_FISHING_ROD_POS
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
        self.mMaxContour = MAX_CONTOUR * self.mWindowRatio
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

        with open(self.__mConfigPath, 'w') as mConfigFile:
            mNewConfig.write(mConfigFile)