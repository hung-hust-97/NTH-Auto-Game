import os
import win32con
import ctypes
import psutil
from ctypes import *
from threading import Lock

ROD_OFFSET = int('0x10', 16)
TEMP_PATH = 'C:\\AutoFishing\\config\\temp.txt'


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


class ReadMemory(metaclass=SingletonMeta):
    mProcessName = "MEmuHeadless.exe"
    mBufferLen = 4
    mProcess = None
    mProcessID = None
    mBaseAddress = 0
    mRodAddress = 0

    def __init__(self):
        pass

    def __del__(self):
        pass

    @staticmethod
    def GetPID(processName):
        for proc in psutil.process_iter():
            if proc.name() == processName:
                return proc.pid
        return None

    def OpenProcess(self):
        self.mProcessID = self.GetPID(self.mProcessName)
        if self.mProcessID is None:
            print("Process was not found")
            return False
        self.mProcess = windll.kernel32.OpenProcess(win32con.PROCESS_VM_READ, 0, self.mProcessID)
        return True

    def GetData(self, address):
        mAddressMemory = ctypes.c_ulong()
        mRead = windll.kernel32.ReadProcessMemory(self.mProcess,
                                                  address,
                                                  ctypes.byref(mAddressMemory),
                                                  self.mBufferLen, 0)
        if mRead:
            return mAddressMemory.value
        return None

    def SetAddress(self):
        self.mRodAddress = self.mBaseAddress + ROD_OFFSET

    def GetBaseAddress(self):
        listText = self.ReadTextFile()
        if not listText:
            print("txt empty")
            return False

        try:
            self.mBaseAddress = int(f'0x{listText[0]}', 16)
            print(f"self.mBaseAddress={self.mBaseAddress}")
        except (ValueError, Exception):
            print("cannot get base addr from txt")
            return False
        return True

    @staticmethod
    def WriteBaseAddress():
        try:
            os.system('cheat_engine\\scanner.exe')
        except (ValueError, Exception):
            print("write base addr fail")
            return False
        return True

    @staticmethod
    def ReadTextFile():
        f = open(TEMP_PATH, "r")
        lines = f.readlines()
        output = []
        for line in lines:
            text = line.split('\n')[0]
            output.append(text)
        return output

    @staticmethod
    def DeleteFile():
        if os.path.exists(TEMP_PATH):
            os.remove(TEMP_PATH)

    def ReadMemoryInit(self):
        checkOpenProc = self.OpenProcess()
        if checkOpenProc is False:
            return False
        checkWriteBase = self.WriteBaseAddress()
        if checkWriteBase is False:
            return False
        checkGetBase = self.GetBaseAddress()
        if checkGetBase is False:
            return False
        self.SetAddress()
        # self.DeleteFile()
