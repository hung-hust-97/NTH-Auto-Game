import cv2
import numpy as np
import time
import win32gui
import win32ui
import win32con
from src.common import *
import logging as log
from src.config import Config
from PyQt5.QtCore import pyqtSignal, QObject


class ScreenHandle(QObject):
    mSignalFindImage = pyqtSignal()
    mImageShow = None

    def __init__(self):
        QObject.__init__(self, parent=None)
        self.mConfig = Config()
        self.hwndMain = None

        self.mLeftBar = 0
        self.mTopBar = 0

        self.mWindowBox = [0, 0, 0, 0]
        self.mFrameSize = [0, 0]

    def CheckWindowApplication(self, window_name):
        self.hwndMain = win32gui.FindWindow(None, window_name)
        if not self.hwndMain:
            return False
        return True

    def SetWindowApplication(self, width: int, height: int):
        # get the window box
        window_rect = win32gui.GetWindowRect(self.hwndMain)
        self.mWindowBox[0] = window_rect[0]
        self.mWindowBox[1] = window_rect[1]
        self.mWindowBox[2] = window_rect[2] - window_rect[0]
        self.mWindowBox[3] = window_rect[3] - window_rect[1]

        # Bo title bar
        self.mTopBar = self.mWindowBox[3] - height
        # print(self.mTopBar)
        # print(self.mWindowBox[2] - width)
        # self.mTopBar = 30

        self.mFrameSize[0] = width
        self.mFrameSize[1] = height
        # win32gui.MoveWindow(self.hwndMain,
        #                     self.mWindowBox[0],
        #                     self.mWindowBox[1],
        #                     width + 40,
        #                     height + 30,
        #                     0)
        # window_rect = win32gui.GetWindowRect(self.hwndMain)
        # self.mWindowBox[0] = window_rect[0]
        # self.mWindowBox[1] = window_rect[1]
        # self.mWindowBox[2] = window_rect[2] - window_rect[0]
        # self.mWindowBox[3] = window_rect[3] - window_rect[1]

    def WindowScreenShot(self, width: int, height: int, left_offset: int, top_offset: int):
        try:
            # get the window image data
            wDC = win32gui.GetWindowDC(self.hwndMain)
            dcObj = win32ui.CreateDCFromHandle(wDC)
            cDC = dcObj.CreateCompatibleDC()
            dataBitMap = win32ui.CreateBitmap()
            dataBitMap.CreateCompatibleBitmap(dcObj, width, height)
            cDC.SelectObject(dataBitMap)
            cDC.BitBlt((0, 0), (width, height), dcObj, (left_offset, top_offset), win32con.SRCCOPY)

            # convert the raw data into a format opencv can read
            signedIntsArray = dataBitMap.GetBitmapBits(True)
            img = np.frombuffer(signedIntsArray, dtype='uint8')
            img.shape = (height, width, 4)

            dcObj.DeleteDC()
            cDC.DeleteDC()
            win32gui.ReleaseDC(self.hwndMain, wDC)
            win32gui.DeleteObject(dataBitMap.GetHandle())
            img = img[..., :3]
            img = np.ascontiguousarray(img)
        except (ValueError, Exception):
            return None
        return img

    def Stream(self):
        while True:
            cv2.imshow("nth", self.WindowScreenShot(self.mFrameSize[0],
                                                    self.mFrameSize[1],
                                                    self.mLeftBar,
                                                    self.mTopBar))
            cv2.waitKey(1)

    def ActivateWindow(self):
        win32gui.ShowWindow(self.hwndMain, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(self.hwndMain)

    def GetWindowBox(self):
        return self.mWindowBox

    def RegionScreenshot(self, mRegion: list):
        frame = self.WindowScreenShot(self.mFrameSize[0],
                                      self.mFrameSize[1],
                                      self.mLeftBar,
                                      self.mTopBar)
        try:
            outFrame = frame[mRegion[1]:mRegion[1] + mRegion[3], mRegion[0]:mRegion[0] + mRegion[2]]
        except (ValueError, Exception):
            log.info(f"REGION_SCREEN_SHOT_ERROR {mRegion}")
            return None
        return outFrame

    def PixelScreenShot(self, mCoordinate: list):
        frame = self.WindowScreenShot(self.mFrameSize[0],
                                      self.mFrameSize[1],
                                      self.mLeftBar,
                                      self.mTopBar)
        try:
            pixel = frame[mCoordinate[1], mCoordinate[0]]
        except (ValueError, Exception):
            log.info(f'PIXEL_SCREEN_SHOT_ERROR {mCoordinate}')
            return None
        return pixel

    # input: gray image, region in windowAppFrame, confident
    def FindImage(self, image, region: list, confidence: float):
        mCroppedAppFrame = self.RegionScreenshot(region)
        if mCroppedAppFrame is None:
            return Flags.FIND_IMG_ERROR
        if self.mConfig.mDebugMode is True:
            self.mImageShow = mCroppedAppFrame
            self.mSignalFindImage.emit()
        mCroppedAppFrame = cv2.cvtColor(mCroppedAppFrame, cv2.COLOR_BGR2GRAY)
        # Store width and height of template in w and h
        # w, h = image.shape[::-1]
        # Perform match operations.
        res = cv2.matchTemplate(mCroppedAppFrame, image, cv2.TM_CCOEFF_NORMED)
        # Store the coordinates of matched area in a numpy array
        loc = np.where(res >= confidence)
        # Draw a rectangle around the matched region.
        for pt in zip(*loc[::-1]):
            # cv2.rectangle(big_img_gray, pt, (pt[0] + w, pt[1] + h), (255, 255, 255), 1)
            return Flags.TRUE
        return Flags.FALSE

    # input: big image, small image gray, confidence float
    @staticmethod
    def CompareImage(big_image, small_image, confidence: float):
        res = cv2.matchTemplate(big_image, small_image, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= confidence)
        for pt in zip(*loc[::-1]):
            return True
        return False

    # input key = VK_SPACE
    def SendKey(self):
        win32gui.SendMessage(self.hwndMain, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)
        win32gui.SendMessage(self.hwndMain, win32con.WM_KEYDOWN, win32con.VK_SPACE, 0)
        win32gui.SendMessage(self.hwndMain, win32con.WM_KEYUP, win32con.VK_SPACE, 0)

    def FindLogo(self):
        mRegionFindLogoScreen = self.WindowScreenShot(50, 50, 0, 0)
        if mRegionFindLogoScreen is None:
            return ''
        mRegionFindLogoScreen = cv2.cvtColor(mRegionFindLogoScreen, cv2.COLOR_BGR2GRAY)
        if self.CompareImage(mRegionFindLogoScreen, self.mConfig.mNoxLogo, 0.7) is True:
            return NOX
        if self.CompareImage(mRegionFindLogoScreen, self.mConfig.mMemuLogo, 0.7) is True:
            return MEMU
        return OTHER
