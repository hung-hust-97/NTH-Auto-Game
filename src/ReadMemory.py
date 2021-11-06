import os
import win32con
import ctypes
import psutil
from ctypes import *
from threading import Lock
import logging as log

# offset from control_base_addr
# +0x10
ROD_OFFSET = int('0x10', 16)
# -0x28
BACKPACK_OFFSET = int('0x28', 16)
# -0x24
ROD_ON_HAND_OFFSET = int('0x24', 16)
# +0xc4
FIX_ROD_OFFSET = int('0xc4', 16)


# offset from filter_base_addr
FISH_TYPE_OFFSET = int('0xAC', 16)
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
    def __init__(self):
        self.mProcessName = "MEmuHeadless.exe"
        self.mBufferLen = 4
        self.mProcess = None
        self.mProcessID = None
        self.mControlBaseAddress = 0
        self.hexControlBaseAddress = "Quét data lỗi"
        self.mFilterBaseAddress = 0
        self.hexFilterBaseAddress = "Quét data lỗi"
        self.mControlAddress = 0
        self.mFishTypeAddress = 0
        self.mBackpackAddress = 0
        self.mRodOnHandAddress = 0
        self.mFixRodAddress = 0

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
            log.info("Process was not found")
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
        log.info("Read memory error")
        return -1

    def SetAddress(self):
        # co cham than xuat hien = 4
        # dang quang can = 1
        # tha can ok  = 3
        # ve giao dien chinh, co ba lo = 0
        # cau thanh cong = 8
        # dang thu can = 7
        # dang keo ca = 5
        # 9 11 6 chua ro
        self.mControlAddress = self.mControlBaseAddress + ROD_OFFSET
        # < close backpack, 300 = open backpack
        self.mBackpackAddress = self.mControlBaseAddress - BACKPACK_OFFSET
        # 1 = ko cam gi ca, 103 = dang cam can cau, > 103 = dang cam linh tinh
        self.mRodOnHandAddress = self.mControlBaseAddress - ROD_ON_HAND_OFFSET
        # 6 = hong can, 4 vaf 9 chua ro
        self.mFixRodAddress = self.mControlBaseAddress + FIX_ROD_OFFSET

        self.mFishTypeAddress = self.mFilterBaseAddress + FISH_TYPE_OFFSET

    def GetBaseAddress(self):
        listText = self.ReadTextFile()
        if listText is None:
            self.hexControlBaseAddress = "Quét data lỗi"
            self.hexFilterBaseAddress = "Quét data lỗi"
            return False

        try:
            self.mControlBaseAddress = int(f'0x{listText[0]}', 16)
            self.hexControlBaseAddress = listText[0]
            self.mFilterBaseAddress = int(f'0x{listText[1]}', 16)
            self.hexFilterBaseAddress = listText[1]
        except (ValueError, Exception):
            self.hexControlBaseAddress = "Quét data lỗi"
            self.hexFilterBaseAddress = "Quét data lỗi"
            return False
        return True

    def WriteBaseAddress(self):
        self.DeleteFile()
        try:
            os.system('cheat_engine\\scanner.exe')
        except (ValueError, Exception):
            log.info("Write base address fail")
            return False
        return True

    @staticmethod
    def ReadTextFile():
        try:
            f = open(TEMP_PATH, "r")
        except (ValueError, Exception):
            log.info("Dont have temp file")
            return None
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
            log.info("checkOpenProc Error")
            return False
        checkWriteBase = self.WriteBaseAddress()
        if checkWriteBase is False:
            log.info("checkWriteBase Error")
            return False
        checkGetBase = self.GetBaseAddress()
        if checkGetBase is False:
            log.info("checkGetBase Error")
            return False
        self.SetAddress()
