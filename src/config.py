import configparser
import os
from threading import Lock


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

        self.__mAdbHost = self.__mConfig['adb_host']
        self.__mAdbPort = int(self.__mConfig['adb_port'])

        self.__mWindowNames = list(self.__mConfig['window_names'].split(','))

        self.__mWindowName = self.__mWindowNames[0]

        self.__mFreeMouse = self.__mConfig.getboolean('free_mouse')

        self.__mDifferentColor = int(self.__mConfig['different_color'])

        self.__mConfidence = float(self.__mConfig['confidence'])

        self.__mWaitingFishTime = int(self.__mConfig['waiting_fish_time'])

        self.__mPullingFishTime = int(self.__mConfig['pulling_fish_time'])

        self.__mFishShadow = self.__mConfig.getboolean('fish_shadow')

        self.__mShowFishShadow = self.__mConfig.getboolean('show_fish_shadow')

        self.__mFishSize = int(self.__mConfig['fish_size'])

        self.__mRadiusFishingRegion = int(self.__mConfig['radius_fishing_region'])

        self.__mOpenBackPack = list(map(int, self.__mConfig['open_back_back'].split(',')))

        self.__mCloseBackPack = list(map(int, self.__mConfig['close_back_pack'].split(',')))

        self.__mTools = list(map(int, self.__mConfig['tools'].split(',')))

        self.__mCastingRod = list(map(int, self.__mConfig['casting_rod'].split(',')))

        self.__mPullingRod = list(map(int, self.__mConfig['pulling_rod'].split(',')))

        self.__mPreservation = list(map(int, self.__mConfig['preservation'].split(',')))

        self.__mConfirm = list(map(int, self.__mConfig['confirm'].split(',')))

        self.__mOKButton = list(map(int, self.__mConfig['ok_button'].split(',')))

        self.__mFishingRod = int(self.__mConfig['fishing_rod'])

        self.__mListFishingRodPosition = []
        self.__mListFishingRodPosition.append(list(map(int, self.__mConfig['fishing_rod_position1'].split(','))))
        self.__mListFishingRodPosition.append(list(map(int, self.__mConfig['fishing_rod_position2'].split(','))))
        self.__mListFishingRodPosition.append(list(map(int, self.__mConfig['fishing_rod_position3'].split(','))))
        self.__mListFishingRodPosition.append(list(map(int, self.__mConfig['fishing_rod_position4'].split(','))))
        self.__mListFishingRodPosition.append(list(map(int, self.__mConfig['fishing_rod_position5'].split(','))))
        self.__mListFishingRodPosition.append(list(map(int, self.__mConfig['fishing_rod_position6'].split(','))))

    def __del__(self):
        pass

    def GetCurrentPath(self):
        return self.__mCurrentPath

    def GetDataPath(self):
        return self.__mDataPath

    def GetAdbHost(self):
        return self.__mAdbHost

    def GetAdbPort(self):
        return self.__mAdbPort

    def GetWindowNames(self):
        return self.__mWindowNames

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

    def SetFishSize(self, mFishSize):
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
        mNewConfig['CONFIG']['window_names'] = self.__mConfig['window_names']
        mNewConfig['CONFIG']['free_mouse'] = str(self.__mFreeMouse)
        mNewConfig['CONFIG']['different_color'] = self.__mConfig['different_color']
        mNewConfig['CONFIG']['confidence'] = self.__mConfig['confidence']
        mNewConfig['CONFIG']['waiting_fish_time'] = str(self.__mWaitingFishTime)
        mNewConfig['CONFIG']['pulling_fish_time'] = str(self.__mPullingFishTime)
        mNewConfig['CONFIG']['fish_shadow'] = str(self.__mFishShadow)
        mNewConfig['CONFIG']['show_fish_shadow'] = str(self.__mShowFishShadow)
        mNewConfig['CONFIG']['fish_size'] = str(self.__mFishSize)
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

        with open(self.__mConfigPath, 'w') as mConfigFile:
            mNewConfig.write(mConfigFile)
