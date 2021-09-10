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
        self.mMutex = Lock()

        self.__mCurrentPath = os.getcwd()
        self.__mDataPath = self.__mCurrentPath + "\\data\\"

        self.__mConfig = configparser.ConfigParser()
        self.__mConfig.read(self.__mCurrentPath + "\\config\\config.ini")
        self.__mConfig = self.__mConfig['CONFIG']

        self.__mAdbHost = self.__mConfig['AdbHost']
        self.__mAdbPort = int(self.__mConfig['AdbPort'])

        self.__mWindowNames = list(self.__mConfig['WindowNames'].split(','))

        self.__mWindowName = self.__mWindowNames[0]

        self.__mFreeMouse = bool(self.__mConfig['FreeMouse'])

        self.__mDifferentColor = int(self.__mConfig['DifferentColor'])

        self.__mConfidence = float(self.__mConfig['Confidence'])

        self.__mWaitingFishTime = int(self.__mConfig['WaitingFishTime'])

        self.__mPullingFishTime = int(self.__mConfig['PullingFishTime'])

        self.__mFishShadow = bool(self.__mConfig['FishShadow'])

        self.__mShowFishShadow = bool(self.__mConfig['ShowFishShadow'])

        self.__mFishSize = int(self.__mConfig['FishSize'])

        self.__mRadiusFishingRegion = int(self.__mConfig['RadiusFishingRegion'])

        self.__mOpenBackPack = list(map(int, self.__mConfig['OpenBackPack'].split(',')))

        self.__mCloseBackPack = list(map(int, self.__mConfig['CloseBackPack'].split(',')))

        self.__mTools = list(map(int, self.__mConfig['Tools'].split(',')))

        self.__mCastingRod = list(map(int, self.__mConfig['CastingRod'].split(',')))

        self.__mPullingRod = list(map(int, self.__mConfig['PullingRod'].split(',')))

        self.__mPreservation = list(map(int, self.__mConfig['Preservation'].split(',')))

        self.__mConfirm = list(map(int, self.__mConfig['Confirm'].split(',')))

        self.__mOKButton = list(map(int, self.__mConfig['OKButton'].split(',')))

        self.__mFishingRod = int(self.__mConfig['FishingRod'])

        self.__mListFishingRodPosition = []
        self.__mListFishingRodPosition.append(list(map(int, self.__mConfig['FishingRodPosition1'].split(','))))
        self.__mListFishingRodPosition.append(list(map(int, self.__mConfig['FishingRodPosition2'].split(','))))
        self.__mListFishingRodPosition.append(list(map(int, self.__mConfig['FishingRodPosition3'].split(','))))
        self.__mListFishingRodPosition.append(list(map(int, self.__mConfig['FishingRodPosition4'].split(','))))
        self.__mListFishingRodPosition.append(list(map(int, self.__mConfig['FishingRodPosition5'].split(','))))
        self.__mListFishingRodPosition.append(list(map(int, self.__mConfig['FishingRodPosition6'].split(','))))

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
        self.mMutex.acquire()
        self.__mWindowName = mWindowName
        self.mMutex.release()

    def GetFreeMouse(self):
        return self.__mFreeMouse

    def SetFreeMouse(self, mFreeMouse: bool):
        self.mMutex.acquire()
        self.__mFreeMouse = mFreeMouse
        self.mMutex.release()

    def GetDifferentColor(self):
        return self.__mDifferentColor

    def GetConfidence(self):
        return self.__mConfidence

    def GetWaitingFishTime(self):
        return self.__mWaitingFishTime

    def SetWaitingFishTime(self, mWaitingFishTime: int):
        self.mMutex.acquire()
        self.__mWaitingFishTime = mWaitingFishTime
        self.mMutex.release()

    def GetPullingFishTime(self):
        return self.__mPullingFishTime

    def SetPullingFishTime(self, mPullingFishTime: int):
        self.mMutex.acquire()
        self.__mWaitingFishTime = mPullingFishTime
        self.mMutex.release()

    def GetFishDetection(self):
        return self.__mFishShadow

    def SetFishDetection(self, mFishShadow: bool):
        self.mMutex.acquire()
        self.__mFishShadow = mFishShadow
        self.mMutex.release()

    def GetShowFishShadow(self):
        return self.__mShowFishShadow

    def SetShowFishShadow(self, mShowFishShadow: bool):
        self.mMutex.acquire()
        self.__mShowFishShadow = mShowFishShadow
        self.mMutex.release()

    def GetFishSize(self):
        return self.__mFishSize

    def SetFishSize(self, mFishSize):
        self.mMutex.acquire()
        self.__mFishSize = mFishSize
        self.mMutex.release()

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
        self.mMutex.acquire()
        self.__mFishingRod = mFishingRod
        self.mMutex.release()

    def GetFishingRodPosition(self):
        return self.__mListFishingRodPosition[self.__mFishingRod - 1]

    def SaveConfig(self):
        with open(self.__mCurrentPath + "\\config\\config.ini", 'w') as mConfigFile:
            self.__mConfig.write(mConfigFile)
