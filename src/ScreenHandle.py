import cv2
import numpy as np
import time
import win32gui
import win32ui
import win32con
from src.common import Flags
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

    @staticmethod
    def CheckWindowApplication(window_name):
        hwnd = win32gui.FindWindow(None, window_name)
        if not hwnd:
            return False
        return hwnd

    def SetWindowApplication(self, window_name, width, height):
        self.hwndMain = win32gui.FindWindow(None, window_name)
        if not self.hwndMain:
            return False

        # get the window box
        window_rect = win32gui.GetWindowRect(self.hwndMain)
        self.mWindowBox[0] = window_rect[0]
        self.mWindowBox[1] = window_rect[1]
        self.mWindowBox[2] = window_rect[2] - window_rect[0]
        self.mWindowBox[3] = window_rect[3] - window_rect[1]

        # Bo title bar
        self.mTopBar = self.mWindowBox[3] - height

        self.mFrameSize[0] = width
        self.mFrameSize[1] = height
        return True

    def WindowScreenShot(self):
        # get the window image data
        wDC = win32gui.GetWindowDC(self.hwndMain)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.mFrameSize[0], self.mFrameSize[1])
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.mFrameSize[0], self.mFrameSize[1]), dcObj, (self.mLeftBar, self.mTopBar),
                   win32con.SRCCOPY)

        # convert the raw data into a format opencv can read
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (self.mFrameSize[1], self.mFrameSize[0], 4)

        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwndMain, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())
        img = img[..., :3]
        img = np.ascontiguousarray(img)
        return img

    def Stream(self):
        while True:
            cv2.imshow("nth", self.WindowScreenShot())
            cv2.waitKey(1)

    @staticmethod
    def ActivateWindow(hwnd):
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)

    def GetWindowBox(self):
        return self.mWindowBox

    def RegionScreenshot(self, mRegion: list):
        frame = self.WindowScreenShot()
        try:
            outFrame = frame[mRegion[1]:mRegion[1] + mRegion[3], mRegion[0]:mRegion[0] + mRegion[2]]
        except (ValueError, Exception):
            log.info(f"REGION_SCREEN_SHOT_ERROR {mRegion}")
            return None
        return outFrame

    def PixelScreenShot(self, listCoordinate: list):
        frame = self.WindowScreenShot()
        listPixel = []
        try:
            for i in range(len(listCoordinate)):
                pixel = frame[listCoordinate[i][1], listCoordinate[i][0]]
                listPixel.append(pixel)
        except (ValueError, Exception):
            return None
        return listPixel

    # input: gray image, region in windowAppFrame, confident
    def FindImage(self, image, region: list, confidence: float):
        mCroppedAppFrame = self.RegionScreenshot(region)
        if mCroppedAppFrame is False:
            return Flags.FIND_IMG_ERROR

        if self.mConfig.mDebugMode is True:
            self.mImageShow = mCroppedAppFrame
            self.mSignalFindImage.emit()

        mCroppedAppFrame = cv2.cvtColor(mCroppedAppFrame, cv2.COLOR_BGR2GRAY)

        check = Flags.FALSE
        # Store width and height of template in w and h
        w, h = image.shape[::-1]
        # Perform match operations.
        res = cv2.matchTemplate(mCroppedAppFrame, image, cv2.TM_CCOEFF_NORMED)
        # Store the coordinates of matched area in a numpy array

        loc = np.where(res >= confidence)
        # Draw a rectangle around the matched region.
        for pt in zip(*loc[::-1]):
            # cv2.rectangle(big_img_gray, pt, (pt[0] + w, pt[1] + h), (255, 255, 255), 1)
            check = Flags.TRUE
            break
        return check

    # input key = VK_SPACE
    def SendKey(self):
        win32gui.SendMessage(self.hwndMain, win32con.WM_KEYDOWN, win32con.VK_SPACE, 0)
        time.sleep(0.01)
        win32gui.SendMessage(self.hwndMain, win32con.WM_KEYUP, win32con.VK_SPACE, 0)
