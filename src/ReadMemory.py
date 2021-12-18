import os
import win32con
import ctypes
import psutil
from ctypes import *
from threading import Lock
import logging as log
# co the su dung thu vien pymem ho tro doc ram rat tot

# offset from control_base_addr
# +0x10
ROD_OFFSET = int('0x1C', 16)
# -0x28
BACKPACK_OFFSET = int('0x28', 16)
# -0x24
ROD_ON_HAND_OFFSET = int('0x24', 16)

# offset from filter_base_addr +10
FISH_TYPE_OFFSET = int('0x10', 16)

MEMU_MARK_SCANNER_PATH = "cheat_engine\\memu\\MarkScanner.exe"
MEMU_FISH_SCANNER_PATH = "cheat_engine\\memu\\FishScanner.exe"
NOX_MARK_SCANNER_PATH = "cheat_engine\\nox\\MarkScanner.exe"
NOX_FISH_SCANNER_PATH = "cheat_engine\\nox\\FishScanner.exe"
MEMU_PROCESS_NAME = "MEmuHeadless.exe"
NOX_PROCESS_NAME = "NoxVMHandle.exe"


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
        self.mProcessName = MEMU_PROCESS_NAME
        self.mProcess = None
        self.mProcessID = None
        self.mControlBaseAddress = 0
        self.hexControlBaseAddress = ""
        self.mFilterBaseAddress = 0
        self.hexFilterBaseAddress = ""
        self.mControlAddress = 0
        self.mFishTypeAddress = 0
        self.mBackpackAddress = 0
        self.mRodOnHandAddress = 0
        self.mFixRodAddress = 0
        self.mListPID = []
        self.mMarkScannerPath = MEMU_MARK_SCANNER_PATH
        self.mFishScannerPath = MEMU_FISH_SCANNER_PATH
        self.mMarkAddressPath = 'C:\\AutoFishing\\config\\mark.txt'
        self.mFishAddressPath = 'C:\\AutoFishing\\config\\fish.txt'

    def __del__(self):
        pass

    def GetPID(self):
        self.mListPID.clear()
        for proc in psutil.process_iter():
            if proc.name() == self.mProcessName:
                self.mListPID.append(proc.pid)

        if not self.mListPID:
            log.info("Process was not found")
            return False
        return True

    def OpenProcess(self):
        try:
            self.mProcess = windll.kernel32.OpenProcess(win32con.PROCESS_VM_READ, 0, self.mProcessID)
        except (ValueError, Exception):
            log.info(f"Cannot open PID {self.mProcessID}")
            return False
        return True

    def GetData(self, address):
        mBufferLen = 4
        mAddressMemory = ctypes.c_ulong()
        mRead = windll.kernel32.ReadProcessMemory(self.mProcess,
                                                  ctypes.c_void_p(address),
                                                  ctypes.byref(mAddressMemory),
                                                  mBufferLen, 0)
        if mRead:
            return mAddressMemory.value
        log.info("Read memory error")
        return -1

    def GetBaseAddress(self, path: str):
        listText = self.ReadTextFile(path)
        if listText is None:
            log.info("listText is None")
            return 0, "ERROR"
        if len(listText) > 1:
            log.info(f"len(listText) = {len(listText)}")
            return 0, "ERROR"

        try:
            intAddress = int(f'0x{listText[0]}', 16)
            hexAddress = listText[0]
        except (ValueError, Exception):
            log.info('')
            return 0, "ERROR"
        return intAddress, hexAddress

    @staticmethod
    def CheatEngine(path: str):
        try:
            os.popen(path).read()
        except (ValueError, Exception):
            log.info("Write base address fail")
            return False
        return True

    @staticmethod
    def ReadTextFile(path: str):
        try:
            f = open(path, "r")
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
    def DeleteFile(path: str):
        if os.path.exists(path):
            os.remove(path)

    def MarkScanner(self):
        self.DeleteFile(self.mMarkAddressPath)
        checkWrite = self.CheatEngine(self.mMarkScannerPath)
        if checkWrite is False:
            log.info("checkWriteBase Error")
            return False
        self.mControlBaseAddress, self.hexControlBaseAddress = self.GetBaseAddress(self.mMarkAddressPath)
        if self.mControlBaseAddress == 0:
            log.info("checkGetBase Error")
            return False

        # cham than xuat hien = 4, tha can ok  = 3, ve giao dien chinh, co ba lo = 0
        # cau thanh cong = 8, dang thu can = 7, dang keo ca = 5
        self.mControlAddress = self.mControlBaseAddress + ROD_OFFSET
        # < close backpack, 300 = open backpack
        self.mBackpackAddress = self.mControlBaseAddress - BACKPACK_OFFSET
        # 1 = ko cam gi ca, 103 = dang cam can cau, > 103 = dang cam linh tinh
        self.mRodOnHandAddress = self.mControlBaseAddress - ROD_ON_HAND_OFFSET

    def FishScanner(self):
        self.DeleteFile(self.mFishAddressPath)
        checkWrite = self.CheatEngine(self.mFishScannerPath)
        if checkWrite is False:
            log.info("checkWriteBase Error")
            return False
        self.mFilterBaseAddress, self.hexFilterBaseAddress = self.GetBaseAddress(self.mFishAddressPath)
        if self.mFilterBaseAddress == 0:
            log.info("checkGetBase Error")
            return False
        self.mFishTypeAddress = self.mFilterBaseAddress + FISH_TYPE_OFFSET
