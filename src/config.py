import configparser
import os
from threading import Lock
from enum import Enum

CREATE_NO_WINDOW = 0x08000000
HIDE_TEXT_BOX_STYLE = "border: 0px; background-color: rgba(0, 0, 0, 10);"
AUTHOR = 'AutoFishing by nth'
BUTTON_COLOR = "background-color: rgb(182, 227, 199)"
IMAGE = "image.png"
YOUTUBE = "youtube.png"
FACEBOOK = "facebook.png"
LEULEU = "leuleu.png"
WAIT_STATUS = "Auto đang đóng chu trình câu\nVui lòng đợi trong giây lát"

DEFAULT_EMULATOR_SIZE = [960, 540]
PRESERVATION_REC = [320, 160]
BACKPACK_REC = [100, 100]
CHECK_TYPE_FISH_POS = [770, 220]
FISH_IMG_REGION = [625, 42, 295, 295]
FONT_SCALE_DEFAULT = 1
MAX_CONTOUR = 5000

# RGB in QT
VIOLET_FISH_COLOR = [231, 147, 232]
BLUE_FISH_COLOR = [89, 198, 217]
GREEN_FISH_COLOR = [163, 228, 103]
GRAY_FISH_COLOR = [228, 224, 197]
# RGB in openCV is BGR
VIOLET_FISH_COLOR_BGR = [232, 147, 231]
BLUE_FISH_COLOR_BGR = [217, 198, 89]
GREEN_FISH_COLOR_BGR = [103, 228, 163]
GRAY_FISH_COLOR_BGR = [197, 224, 228]

TEXT_COLOR = (255, 255, 255)


class FishImageColor(Enum):
    RGB = 1
    GRAY = 2
    LEU = 3


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
        self.__mDataPath = self.__mCurrentPath + "\\data\\"
        self.__mConfigPath = self.__mCurrentPath + "\\config\\config.ini"

        self.__mConfigParser = configparser.ConfigParser()
        self.__mConfigParser.read(self.__mConfigPath)
        self.__mConfig = self.__mConfigParser['CONFIG']

        self.__mEmulatorSize = list(map(int, self.__mConfig['emulator_size'].split(',')))

        self.__mAdbHost = self.__mConfig['adb_host']
        self.__mAdbPort = int(self.__mConfig['adb_port'])

        self.__mWindowName = self.__mConfig['window_name']

        self.__mFreeMouse = self.__mConfig.getboolean('free_mouse')

        self.__mDifferentColor = int(self.__mConfig['different_color'])

        self.__mConfidence = float(self.__mConfig['confidence'])

        self.__mWaitingFishTime = int(self.__mConfig['waiting_fish_time'])

        self.__mPullingFishTime = int(self.__mConfig['pulling_fish_time'])

        self.__mFishShadow = self.__mConfig.getboolean('fish_shadow')

        self.__mShowFishShadow = self.__mConfig.getboolean('show_fish_shadow')

        self.__mFishSize = int(self.__mConfig['fish_size'])

        self.__mMinContour = int(self.__mConfig['min_contour'])

        self.__mRadiusFishingRegion = int(self.__mConfig['radius_fishing_region'])
        self.__mRadiusFishingRegion = self.__mRadiusFishingRegion * self.__mEmulatorSize[0] // DEFAULT_EMULATOR_SIZE[0]

        self.__mOpenBackPack = list(map(int, self.__mConfig['open_back_back'].split(',')))
        self.__mOpenBackPack = [self.__mOpenBackPack[0] * self.__mEmulatorSize[0] // DEFAULT_EMULATOR_SIZE[0],
                                self.__mOpenBackPack[1] * self.__mEmulatorSize[1] // DEFAULT_EMULATOR_SIZE[1]]

        self.__mCloseBackPack = list(map(int, self.__mConfig['close_back_pack'].split(',')))
        self.__mCloseBackPack = [self.__mCloseBackPack[0] * self.__mEmulatorSize[0] // DEFAULT_EMULATOR_SIZE[0],
                                 self.__mCloseBackPack[1] * self.__mEmulatorSize[1] // DEFAULT_EMULATOR_SIZE[1]]

        self.__mTools = list(map(int, self.__mConfig['tools'].split(',')))
        self.__mTools = [self.__mTools[0] * self.__mEmulatorSize[0] // DEFAULT_EMULATOR_SIZE[0],
                         self.__mTools[1] * self.__mEmulatorSize[1] // DEFAULT_EMULATOR_SIZE[1]]

        self.__mCastingRod = list(map(int, self.__mConfig['casting_rod'].split(',')))
        self.__mCastingRod = [self.__mCastingRod[0] * self.__mEmulatorSize[0] // DEFAULT_EMULATOR_SIZE[0],
                              self.__mCastingRod[1] * self.__mEmulatorSize[1] // DEFAULT_EMULATOR_SIZE[1]]

        self.__mPullingRod = list(map(int, self.__mConfig['pulling_rod'].split(',')))
        self.__mPullingRod = [self.__mPullingRod[0] * self.__mEmulatorSize[0] // DEFAULT_EMULATOR_SIZE[0],
                              self.__mPullingRod[1] * self.__mEmulatorSize[1] // DEFAULT_EMULATOR_SIZE[1]]

        self.__mPreservation = list(map(int, self.__mConfig['preservation'].split(',')))
        self.__mPreservation = [self.__mPreservation[0] * self.__mEmulatorSize[0] // DEFAULT_EMULATOR_SIZE[0],
                                self.__mPreservation[1] * self.__mEmulatorSize[1] // DEFAULT_EMULATOR_SIZE[1]]

        self.__mConfirm = list(map(int, self.__mConfig['confirm'].split(',')))
        self.__mConfirm = [self.__mConfirm[0] * self.__mEmulatorSize[0] // DEFAULT_EMULATOR_SIZE[0],
                           self.__mConfirm[1] * self.__mEmulatorSize[1] // DEFAULT_EMULATOR_SIZE[1]]

        self.__mOKButton = list(map(int, self.__mConfig['ok_button'].split(',')))
        self.__mOKButton = [self.__mOKButton[0] * self.__mEmulatorSize[0] // DEFAULT_EMULATOR_SIZE[0],
                            self.__mOKButton[1] * self.__mEmulatorSize[1] // DEFAULT_EMULATOR_SIZE[1]]

        self.__mFishingRod = int(self.__mConfig['fishing_rod'])

        self.__mDelayTime = float(self.__mConfig['delay_time'])

        self.__mShutdownCheckBox = False

        self.__mShutdownTime = 0

        self.__mAdbAddress = "None"

        self.__mFacebook = self.__mConfig["facebook"]
        self.__mYoutube = self.__mConfig["youtube"]

        self.__mListFishingRodPosition = []
        self.__mListFishingRodPosition.append(list(map(int, self.__mConfig['fishing_rod_position1'].split(','))))
        self.__mListFishingRodPosition.append(list(map(int, self.__mConfig['fishing_rod_position2'].split(','))))
        self.__mListFishingRodPosition.append(list(map(int, self.__mConfig['fishing_rod_position3'].split(','))))
        self.__mListFishingRodPosition.append(list(map(int, self.__mConfig['fishing_rod_position4'].split(','))))
        self.__mListFishingRodPosition.append(list(map(int, self.__mConfig['fishing_rod_position5'].split(','))))
        self.__mListFishingRodPosition.append(list(map(int, self.__mConfig['fishing_rod_position6'].split(','))))

        for i in range(len(self.__mListFishingRodPosition)):
            self.__mListFishingRodPosition[i] = [
                self.__mListFishingRodPosition[i][0] * self.__mEmulatorSize[0] // DEFAULT_EMULATOR_SIZE[0],
                self.__mListFishingRodPosition[i][1] * self.__mEmulatorSize[1] // DEFAULT_EMULATOR_SIZE[1]]

        self.mPreservationRec = [PRESERVATION_REC[0] * self.__mEmulatorSize[0] // DEFAULT_EMULATOR_SIZE[0],
                                 PRESERVATION_REC[1] * self.__mEmulatorSize[1] // DEFAULT_EMULATOR_SIZE[1]]
        self.mBackpackRec = [BACKPACK_REC[0] * self.__mEmulatorSize[0] // DEFAULT_EMULATOR_SIZE[0],
                             BACKPACK_REC[1] * self.__mEmulatorSize[1] // DEFAULT_EMULATOR_SIZE[1]]
        self.mCheckTypeFishPos = [CHECK_TYPE_FISH_POS[0] * self.__mEmulatorSize[0] // DEFAULT_EMULATOR_SIZE[0],
                                  CHECK_TYPE_FISH_POS[1] * self.__mEmulatorSize[1] // DEFAULT_EMULATOR_SIZE[1]]
        self.mFishImgRegion = [FISH_IMG_REGION[0] * self.__mEmulatorSize[0] // DEFAULT_EMULATOR_SIZE[0],
                               FISH_IMG_REGION[1] * self.__mEmulatorSize[1] // DEFAULT_EMULATOR_SIZE[1],
                               FISH_IMG_REGION[2] * self.__mEmulatorSize[0] // DEFAULT_EMULATOR_SIZE[0],
                               FISH_IMG_REGION[3] * self.__mEmulatorSize[1] // DEFAULT_EMULATOR_SIZE[1]]
        self.mFontScale = FONT_SCALE_DEFAULT * self.__mEmulatorSize[0] / DEFAULT_EMULATOR_SIZE[0]
        self.mMaxContour = MAX_CONTOUR * self.__mEmulatorSize[0] // DEFAULT_EMULATOR_SIZE[0]

    def __del__(self):
        pass

    def GetDelayTime(self):
        return self.__mDelayTime

    def SetDelayTime(self, mDelayTime: float):
        self.__mMutex.acquire()
        self.__mDelayTime = mDelayTime
        self.__mMutex.release()

    def GetFacebook(self):
        return self.__mFacebook

    def GetYoutube(self):
        return self.__mYoutube

    def GetAdbAddress(self):
        return self.__mAdbAddress

    def SetAdbAddress(self, mAdbAddress: str):
        self.__mMutex.acquire()
        self.__mAdbAddress = mAdbAddress
        self.__mMutex.release()

    def GetMinContour(self):
        return self.__mMinContour

    def SetMinContour(self, mMinContour: int):
        self.__mMutex.acquire()
        self.__mMinContour = mMinContour
        self.__mMutex.release()

    def GetShutdownCheckBox(self):
        return self.__mShutdownCheckBox

    def SetShutdownCheckBox(self, mShutdownCheckBox: bool):
        self.__mMutex.acquire()
        self.__mShutdownCheckBox = mShutdownCheckBox
        self.__mMutex.release()

    def GetShutdownTime(self):
        return self.__mShutdownTime

    def SetShutdownTime(self, mShutdownTime: int):
        self.__mMutex.acquire()
        self.__mShutdownTime = mShutdownTime
        self.__mMutex.release()

    def GetEmulatorSize(self):
        return self.__mEmulatorSize

    def GetCurrentPath(self):
        return self.__mCurrentPath

    def GetDataPath(self):
        return self.__mDataPath

    def GetAdbHost(self):
        return self.__mAdbHost

    def GetAdbPort(self):
        return self.__mAdbPort

    def GetWindowName(self):
        return self.__mWindowName

    def SetWindowName(self, mWindowName: str):
        self.__mMutex.acquire()
        self.__mWindowName = mWindowName
        self.__mMutex.release()

    def GetFreeMouse(self):
        return self.__mFreeMouse

    def SetFreeMouse(self, mFreeMouse: bool):
        self.__mMutex.acquire()
        self.__mFreeMouse = mFreeMouse
        self.__mMutex.release()

    def GetDifferentColor(self):
        return self.__mDifferentColor

    def GetConfidence(self):
        return self.__mConfidence

    def GetWaitingFishTime(self):
        return self.__mWaitingFishTime

    def SetWaitingFishTime(self, mWaitingFishTime: int):
        self.__mMutex.acquire()
        self.__mWaitingFishTime = mWaitingFishTime
        self.__mMutex.release()

    def GetPullingFishTime(self):
        return self.__mPullingFishTime

    def SetPullingFishTime(self, mPullingFishTime: int):
        self.__mMutex.acquire()
        self.__mPullingFishTime = mPullingFishTime
        self.__mMutex.release()

    def GetFishDetection(self):
        return self.__mFishShadow

    def SetFishDetection(self, mFishShadow: bool):
        self.__mMutex.acquire()
        self.__mFishShadow = mFishShadow
        self.__mMutex.release()

    def GetShowFishShadow(self):
        return self.__mShowFishShadow

    def SetShowFishShadow(self, mShowFishShadow: bool):
        self.__mMutex.acquire()
        self.__mShowFishShadow = mShowFishShadow
        self.__mMutex.release()

    def GetFishSize(self):
        return self.__mFishSize

    def SetFishSize(self, mFishSize: int):
        self.__mMutex.acquire()
        self.__mFishSize = mFishSize
        self.__mMutex.release()

    def GetRadiusFishingRegion(self):
        return self.__mRadiusFishingRegion

    def GetOpenBackPack(self):
        return self.__mOpenBackPack

    def GetCloseBackPack(self):
        return self.__mCloseBackPack

    def GetTools(self):
        return self.__mTools

    def GetCastingRod(self):
        return self.__mCastingRod

    def GetPullingRod(self):
        return self.__mPullingRod

    def GetPreservation(self):
        return self.__mPreservation

    def GetConfirm(self):
        return self.__mConfirm

    def GetOKButton(self):
        return self.__mOKButton

    def GetFishingRod(self):
        return self.__mFishingRod

    def SetFishingRod(self, mFishingRod: int):
        self.__mMutex.acquire()
        self.__mFishingRod = mFishingRod
        self.__mMutex.release()

    def GetFishingRodPosition(self):
        return self.__mListFishingRodPosition[self.__mFishingRod - 1]

    def SaveConfig(self):
        mNewConfig = configparser.ConfigParser()
        mNewConfig['CONFIG'] = {}
        mNewConfig['CONFIG']['adb_host'] = self.__mConfig['adb_host']
        mNewConfig['CONFIG']['adb_port'] = self.__mConfig['adb_port']
        mNewConfig['CONFIG']['window_name'] = self.__mWindowName
        mNewConfig['CONFIG']['emulator_size'] = self.__mConfig['emulator_size']
        mNewConfig['CONFIG']['free_mouse'] = str(self.__mFreeMouse)
        mNewConfig['CONFIG']['different_color'] = self.__mConfig['different_color']
        mNewConfig['CONFIG']['confidence'] = self.__mConfig['confidence']
        mNewConfig['CONFIG']['waiting_fish_time'] = str(self.__mWaitingFishTime)
        mNewConfig['CONFIG']['pulling_fish_time'] = str(self.__mPullingFishTime)
        mNewConfig['CONFIG']['fish_shadow'] = str(self.__mFishShadow)
        mNewConfig['CONFIG']['show_fish_shadow'] = str(self.__mShowFishShadow)
        mNewConfig['CONFIG']['fish_size'] = str(self.__mFishSize)
        mNewConfig['CONFIG']['min_contour'] = str(self.__mMinContour)
        mNewConfig['CONFIG']['radius_fishing_region'] = self.__mConfig['radius_fishing_region']
        mNewConfig['CONFIG']['open_back_back'] = self.__mConfig['open_back_back']
        mNewConfig['CONFIG']['close_back_pack'] = self.__mConfig['close_back_pack']
        mNewConfig['CONFIG']['tools'] = self.__mConfig['tools']
        mNewConfig['CONFIG']['casting_rod'] = self.__mConfig['casting_rod']
        mNewConfig['CONFIG']['pulling_rod'] = self.__mConfig['pulling_rod']
        mNewConfig['CONFIG']['preservation'] = self.__mConfig['preservation']
        mNewConfig['CONFIG']['confirm'] = self.__mConfig['confirm']
        mNewConfig['CONFIG']['ok_button'] = self.__mConfig['ok_button']
        mNewConfig['CONFIG']['fishing_rod'] = str(self.__mFishingRod)
        mNewConfig['CONFIG']['fishing_rod_position1'] = self.__mConfig['fishing_rod_position1']
        mNewConfig['CONFIG']['fishing_rod_position2'] = self.__mConfig['fishing_rod_position2']
        mNewConfig['CONFIG']['fishing_rod_position3'] = self.__mConfig['fishing_rod_position3']
        mNewConfig['CONFIG']['fishing_rod_position4'] = self.__mConfig['fishing_rod_position4']
        mNewConfig['CONFIG']['fishing_rod_position5'] = self.__mConfig['fishing_rod_position5']
        mNewConfig['CONFIG']['fishing_rod_position6'] = self.__mConfig['fishing_rod_position6']
        mNewConfig['CONFIG']['delay_time'] = str(self.__mDelayTime)
        mNewConfig['CONFIG']['facebook'] = self.__mConfig['facebook']
        mNewConfig['CONFIG']['youtube'] = self.__mConfig['youtube']

        with open(self.__mConfigPath, 'w') as mConfigFile:
            mNewConfig.write(mConfigFile)
