import math
import time
import pyautogui
# import os
import gc
import cv2
from ppadb.client import Client as AdbClient
import numpy
import win32api
from src.config import Config
from PyQt5.QtCore import pyqtSignal, QObject
import subprocess

NOISE_CONTOUR_SIZE = 30
CREATE_NO_WINDOW = 0x08000000


class AutoFishing(QObject):
    mSignalSetPixelPos = pyqtSignal(int, int)
    mSignalSetFishingBobberPos = pyqtSignal(int, int)
    mSignalUpdateFishingNum = pyqtSignal(int)
    mSignalUpdateFishNum = pyqtSignal(int)
    mSignalUpdateFishDetectionImage = pyqtSignal()
    mSignalMessage = pyqtSignal(str, bool)
    mSignalUpdateStatus = pyqtSignal(str)

    def __init__(self):
        QObject.__init__(self, parent=None)  # Kế thừa QObject
        self.mConfig = Config()
        self.mFishingNum = 0
        self.mFishNum = 0
        self.mAbsPullingRodPos = [0, 0]
        self.mMark = [0, 0]
        self.mFishingRegion = [0, 0, 0, 0]
        self.mAdbDevice = None
        self.mFishDetectionRunning = True
        self.mCheckMouseRunning = False
        self.mAutoFishRunning = False
        self.mCheckFish = False
        self.mEmulatorWindow = None
        self.mEmulatorBox = None
        self.mFishImage = None
        self.mCurrentTime = time.time()

    def __del__(self):
        self.mFishDetectionRunning = False
        self.mCheckMouseRunning = False
        self.mAutoFishRunning = False

    def MsgEmit(self, mText: str, mReturn: bool):
        self.mSignalMessage.emit(mText, mReturn)

    def StatusEmit(self, mText: str):
        self.mSignalUpdateStatus.emit(mText)

    @staticmethod
    def CheckLeftMouseClick():
        if win32api.GetKeyState(0x01) < 0:
            return True
        return False

    @staticmethod
    def CheckRightMouseClick():
        if win32api.GetKeyState(0x02) < 0:
            return True
        return False

    def CloseBackPack(self):
        self.AdbClick(self.mConfig.GetCloseBackPack()[0],
                      self.mConfig.GetCloseBackPack()[1])
        return True

    def OpenTools(self):
        self.AdbClick(self.mConfig.GetTools()[0],
                      self.mConfig.GetTools()[1])
        return True

    def OpenBackPack(self):
        self.AdbClick(self.mConfig.GetOpenBackPack()[0],
                      self.mConfig.GetOpenBackPack()[1])

    def FixClick(self):
        self.AdbClick(self.mConfig.GetFishingRodPosition()[0],
                      self.mConfig.GetFishingRodPosition()[1])
        self.StatusEmit("Đã bấm nút sửa cần")

    def FixConfirm(self):
        self.AdbClick(self.mConfig.GetConfirm()[0],
                      self.mConfig.GetConfirm()[1])
        self.StatusEmit("Đã xác nhận sửa cần")

    def ClickOk(self):
        self.AdbClick(self.mConfig.GetOKButton()[0],
                      self.mConfig.GetOKButton()[1])

    def CheckRod(self):
        time.sleep(3)
        if self.mAutoFishRunning is False:
            return False
        mCheck = 0
        while mCheck < 5:
            mBackpackPos = pyautogui.locateOnScreen(f'{self.mConfig.GetDataPath()}backpack.png',
                                                    grayscale=True,
                                                    region=(self.mEmulatorBox.left,
                                                            self.mEmulatorBox.top,
                                                            self.mEmulatorBox.width,
                                                            self.mEmulatorBox.height),
                                                    confidence=self.mConfig.GetConfidence())
            if mBackpackPos is not None:
                self.FixRod()
                return False
            mBackpackPos = pyautogui.locateOnScreen(f'{self.mConfig.GetDataPath()}backpack2.png',
                                                    grayscale=True,
                                                    region=(self.mEmulatorBox.left,
                                                            self.mEmulatorBox.top,
                                                            self.mEmulatorBox.width,
                                                            self.mEmulatorBox.height),
                                                    confidence=self.mConfig.GetConfidence())
            if mBackpackPos is not None:
                self.FixRod()
                return False
            time.sleep(0.2)
            mCheck += 1
            if self.mAutoFishRunning is False:
                return False
        return True

    def FixRod(self):
        self.OpenBackPack()
        if self.mAutoFishRunning is False:
            return
        time.sleep(1)
        if self.mAutoFishRunning is False:
            return
        self.OpenTools()
        if self.mAutoFishRunning is False:
            return
        time.sleep(0.5)
        if self.mAutoFishRunning is False:
            return
        self.FixClick()
        if self.mAutoFishRunning is False:
            return
        time.sleep(0.5)
        if self.mAutoFishRunning is False:
            return
        self.FixConfirm()
        if self.mAutoFishRunning is False:
            return
        time.sleep(0.5)
        if self.mAutoFishRunning is False:
            return
        self.ClickOk()
        if self.mAutoFishRunning is False:
            return
        time.sleep(0.5)
        if self.mAutoFishRunning is False:
            return
        self.CloseBackPack()
        if self.mAutoFishRunning is False:
            return
        time.sleep(0.5)
        if self.mAutoFishRunning is False:
            return
        mCheck = 0
        while mCheck < 5:
            mPreservationButtonPos = pyautogui.locateOnScreen(f'{self.mConfig.GetDataPath()}preservation.png',
                                                              grayscale=True,
                                                              region=(self.mEmulatorBox.left,
                                                                      self.mEmulatorBox.top,
                                                                      self.mEmulatorBox.width,
                                                                      self.mEmulatorBox.height),
                                                              confidence=self.mConfig.GetConfidence())
            if mPreservationButtonPos is not None:
                self.AdbClick(self.mConfig.GetPreservation()[0],
                              self.mConfig.GetPreservation()[1])
                return
            mCheck += 1
            time.sleep(0.2)
            if self.mAutoFishRunning is False:
                return
        return

    def CastFishingRod(self):
        mCheck = 0
        while mCheck < 5:
            mBackpackPos = pyautogui.locateOnScreen(f'{self.mConfig.GetDataPath()}backpack.png',
                                                    grayscale=True,
                                                    region=(self.mEmulatorBox.left,
                                                            self.mEmulatorBox.top,
                                                            self.mEmulatorBox.width,
                                                            self.mEmulatorBox.height),
                                                    confidence=self.mConfig.GetConfidence())
            if mBackpackPos is not None:
                self.AdbClick(self.mConfig.GetCastingRod()[0],
                              self.mConfig.GetCastingRod()[1])
                self.StatusEmit("Đã thả cần câu")
                return True
            mBackpackPos = pyautogui.locateOnScreen(f'{self.mConfig.GetDataPath()}backpack2.png',
                                                    grayscale=True,
                                                    region=(self.mEmulatorBox.left,
                                                            self.mEmulatorBox.top,
                                                            self.mEmulatorBox.width,
                                                            self.mEmulatorBox.height),
                                                    confidence=self.mConfig.GetConfidence())
            if mBackpackPos is not None:
                self.AdbClick(self.mConfig.GetCastingRod()[0],
                              self.mConfig.GetCastingRod()[1])
                self.StatusEmit("Đã thả cần câu")
                return True
            time.sleep(0.2)
            mCheck += 1
        self.StatusEmit("Không tìm được cần câu")
        self.CloseBackPack()
        mCheck = 0
        while mCheck < 5:
            mPreservationButtonPos = pyautogui.locateOnScreen(f'{self.mConfig.GetDataPath()}preservation.png',
                                                              grayscale=True,
                                                              region=(self.mEmulatorBox.left,
                                                                      self.mEmulatorBox.top,
                                                                      self.mEmulatorBox.width,
                                                                      self.mEmulatorBox.height),
                                                              confidence=self.mConfig.GetConfidence())
            if mPreservationButtonPos is not None:
                self.AdbClick(self.mConfig.GetPreservation()[0],
                              self.mConfig.GetPreservation()[1])
                return False
            mCheck += 1
            time.sleep(0.2)
            if self.mAutoFishRunning is False:
                return False
        return False

    def FishDetection(self, mPrevFrame, mCurrFrame):
        mBackGroundColor = mPrevFrame[self.mFishingRegion[3] // 2, self.mFishingRegion[2] // 4]
        # tối ở camp 49  # tối ở biển 57
        if mBackGroundColor <= 70:
            mMinThreshValue = 10
            mMaxThreshValue = 100
            mColor = (255, 255, 255)
        # buổi chiều nền biền 74, sáng ở camp 149, chiều ở cam 166
        elif 70 < mBackGroundColor < 170:
            mMinThreshValue = 30
            mMaxThreshValue = 100
            mColor = (255, 255, 255)
        # buổi sáng nền biển 174
        else:
            mMinThreshValue = 50
            mMaxThreshValue = 100
            mColor = (0, 0, 0)

        mCurrImgArrWidth, mCurrImgArrHeight = mCurrFrame.shape
        mImgCenterX = mCurrImgArrWidth // 2
        mImgCenterY = mCurrImgArrHeight // 2

        mPrevFrameBlur = cv2.GaussianBlur(mPrevFrame, (21, 21), 0)
        mCurrFrameBlur = cv2.GaussianBlur(mCurrFrame, (21, 21), 0)

        # so sánh 2 frame, tìm sai khác
        mFrameDelta = cv2.absdiff(mPrevFrameBlur, mCurrFrameBlur)
        mThresh = cv2.threshold(mFrameDelta, mMinThreshValue, mMaxThreshValue, cv2.THRESH_BINARY)[1]

        # Fill in holes via dilate()
        mThresh = cv2.dilate(mThresh, None, iterations=2)

        # Tìm đường biên contours, hierarchy
        mContours, mHierarchy = cv2.findContours(mThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Quét tất cả các đường biên
        mFishArea = 0
        mCurrFrame = cv2.circle(mCurrFrame, (mImgCenterX, mImgCenterY),
                                self.mConfig.GetRadiusFishingRegion() * 3 // 4, mColor, 1)
        mCurrFrame = cv2.circle(mCurrFrame, (mImgCenterX, mImgCenterY), self.mConfig.GetRadiusFishingRegion() * 1 // 4,
                                mColor, 1)
        cv2.putText(mCurrFrame, str(mBackGroundColor), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, mColor, 2)
        for mContour in mContours:
            # check coordinates of all found contours
            (x, y, w, h) = cv2.boundingRect(mContour)
            mContourCenterX = x + w // 2
            mContourCenterY = y + h // 2
            mRadius = math.sqrt(pow((mImgCenterX - mContourCenterX), 2) + pow((mImgCenterY - mContourCenterY), 2))
            # loại bỏ phao câu
            if mRadius < self.mConfig.GetRadiusFishingRegion() / 4:
                continue
            # loại box nhỏ tránh nhiễu
            if cv2.contourArea(mContour) < NOISE_CONTOUR_SIZE:
                continue
            # loại bỏ box xuất hiện ở viền
            if mRadius > self.mConfig.GetRadiusFishingRegion() * 3 / 4:
                continue
            mFishArea = int(cv2.contourArea(mContour))
            cv2.rectangle(mCurrFrame, (x, y), (x + w, y + h), mColor, 1)
            cv2.putText(mCurrFrame, str(mFishArea), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, mColor, 2)

            break
        mCurrFrameResize = cv2.resize(mCurrFrame, (200, 200), interpolation=cv2.INTER_AREA)
        self.mFishImage = mCurrFrameResize
        if self.mConfig.GetShowFishShadow() is True:
            self.mSignalUpdateFishDetectionImage.emit()
        return mFishArea

    def ScreenshotFishingRegion(self):
        mScreenShotPilImage = pyautogui.screenshot(region=(self.mFishingRegion[0],
                                                           self.mFishingRegion[1],
                                                           self.mFishingRegion[2],
                                                           self.mFishingRegion[3]))
        mScreenShotMat = numpy.array(mScreenShotPilImage)
        mScreenShotMatGray = cv2.cvtColor(mScreenShotMat, cv2.COLOR_RGB2GRAY)
        return mScreenShotMatGray

    def CheckMark(self):
        time.sleep(3)
        mStaticFrame = None
        if self.mConfig.GetFishDetection() is True:
            mStaticFrame = self.ScreenshotFishingRegion()
            self.mFishImage = cv2.resize(mStaticFrame, (200, 200), interpolation=cv2.INTER_AREA)
        self.StatusEmit("Đang đợi dấu chấm than")
        time1 = time.time()
        time2 = time.time()
        mCheck = False
        while (time2 - time1) < 5:
            try:
                mPixel = pyautogui.pixel(self.mMark[0], self.mMark[1])
            except:
                time.sleep(0.01)
                time2 = time.time()
                continue
            mCheck = True
            break

        if mCheck is False:
            self.StatusEmit("Lỗi hệ thống\nKhông xác định được màu nền tại vị trí dấu chấm than")
            return False
        time1 = time.time()
        time2 = time.time()
        mStopDetect = False
        mSkipFrame = 0
        while (time2 - time1) < self.mConfig.GetWaitingFishTime():
            if self.mConfig.GetFishDetection() is True and mStopDetect is False:
                mCurrentFrame = self.ScreenshotFishingRegion()
                mSizeFish = self.FishDetection(mStaticFrame, mCurrentFrame)
                if mSizeFish != 0:
                    mSkipFrame += 1
                if mSkipFrame == 10:
                    mStopDetect = True
                    if mSizeFish < int(self.mConfig.GetFishSize()):
                        return True
            try:
                mPixelCurrent = pyautogui.pixel(self.mMark[0], self.mMark[1])
            except:
                time2 = time.time()
                continue

            mDiffR = abs(mPixelCurrent[0] - mPixel[0])
            mDiffG = abs(mPixelCurrent[1] - mPixel[1])
            mDiffB = abs(mPixelCurrent[2] - mPixel[2])
            mDiffRgb = (mDiffR + mDiffG + mDiffB) // 3

            if mDiffRgb > self.mConfig.GetDifferentColor():
                return True

            time2 = time.time()
            time.sleep(0.01)
            if self.mAutoFishRunning is False:
                return False

        self.StatusEmit("Không phát hiện dấu chấm than")
        return False

    def PullFishingRod(self):
        if self.mConfig.GetFreeMouse() is False:
            pyautogui.click(x=self.mAbsPullingRodPos[0],
                            y=self.mAbsPullingRodPos[1],
                            clicks=2,
                            interval=0.1,
                            button='left')
            self.StatusEmit("Đang kéo cần câu")
            return True
        else:
            self.StatusEmit("Đang kéo cần câu")
            time1 = time.time()
            self.AdbClick(self.mConfig.GetPullingRod()[0],
                          self.mConfig.GetPullingRod()[1])
            time2 = time.time()
            self.StatusEmit(f'Độ trễ giật cần {round(time2 - time1, 2)} giây')
            return True

    def FishPreservation(self):
        time.sleep(0.1)
        if self.mAutoFishRunning is False:
            return False
        time1 = time.time()
        time2 = time.time()
        while (time2 - time1) < int(self.mConfig.GetPullingFishTime()):
            mBackpackPos = pyautogui.locateOnScreen(f'{self.mConfig.GetDataPath()}backpack.png',
                                                    region=(self.mEmulatorBox.left,
                                                            self.mEmulatorBox.top,
                                                            self.mEmulatorBox.width,
                                                            self.mEmulatorBox.height),
                                                    grayscale=True,
                                                    confidence=self.mConfig.GetConfidence())
            if mBackpackPos is not None:
                self.StatusEmit("Câu thất bại")
                return True

            mBackpackPos = pyautogui.locateOnScreen(f'{self.mConfig.GetDataPath()}backpack2.png',
                                                    region=(self.mEmulatorBox.left,
                                                            self.mEmulatorBox.top,
                                                            self.mEmulatorBox.width,
                                                            self.mEmulatorBox.height),
                                                    grayscale=True,
                                                    confidence=self.mConfig.GetConfidence())
            if mBackpackPos is not None:
                self.StatusEmit("Câu thất bại")
                return True
            mPreservationPos = pyautogui.locateOnScreen(f'{self.mConfig.GetDataPath()}preservation.png',
                                                        region=(self.mEmulatorBox.left,
                                                                self.mEmulatorBox.top,
                                                                self.mEmulatorBox.width,
                                                                self.mEmulatorBox.height),
                                                        grayscale=True,
                                                        confidence=self.mConfig.GetConfidence())
            if mPreservationPos is not None:
                self.StatusEmit("Câu thành công")
                self.mFishNum += 1
                self.AdbClick(self.mConfig.GetPreservation()[0],
                              self.mConfig.GetPreservation()[1])
                return True

            time.sleep(0.1)
            time2 = time.time()
            if self.mAutoFishRunning is False:
                return False
        self.StatusEmit("Kiểm tra kết quả bị lỗi")
        return False

    def SetPixelPos(self):
        self.mMark = [0, 0]
        time.sleep(0.1)
        mMousePos = pyautogui.position()
        self.StatusEmit("Di chuyển chuột đến đầu của dấu chấm than và Click")
        self.mCheckMouseRunning = True
        while self.mCheckMouseRunning is True:
            mMousePos = pyautogui.position()
            self.mSignalSetPixelPos.emit(int(mMousePos.x), int(mMousePos.y))
            if self.CheckLeftMouseClick() is True:
                self.mCheckMouseRunning = False
            time.sleep(0.01)
        self.mMark[0] = int(mMousePos.x)
        self.mMark[1] = int(mMousePos.y)
        self.StatusEmit(f'Vị trí dấu chấm than đã cài đặt:\n{mMousePos}')

    def SetFishingBobberPos(self):
        self.mFishingRegion = [0, 0, 0, 0]
        time.sleep(0.1)
        mScreenSize = pyautogui.size()
        mMousePos = pyautogui.position()
        self.StatusEmit("Di chuyển chuột đến phao câu và Click")
        self.mCheckMouseRunning = True
        while self.mCheckMouseRunning is True:
            mMousePos = pyautogui.position()
            self.mSignalSetFishingBobberPos.emit(int(mMousePos.x), int(mMousePos.y))
            if self.CheckLeftMouseClick() is True:
                self.mCheckMouseRunning = False
            time.sleep(0.01)
        self.mFishingRegion[0] = mMousePos.x - self.mConfig.GetRadiusFishingRegion()
        self.mFishingRegion[1] = mMousePos.y - self.mConfig.GetRadiusFishingRegion()
        self.mFishingRegion[2] = self.mConfig.GetRadiusFishingRegion() * 2
        self.mFishingRegion[3] = self.mConfig.GetRadiusFishingRegion() * 2
        if self.mFishingRegion[0] < self.mConfig.GetRadiusFishingRegion() \
                or self.mFishingRegion[1] < self.mConfig.GetRadiusFishingRegion() \
                or self.mFishingRegion[0] + self.mFishingRegion[2] > mScreenSize.width \
                or self.mFishingRegion[1] + self.mFishingRegion[3] > mScreenSize.height:
            self.MsgEmit("Vị trí phao câu quá gần viền màn hình", False)
        self.StatusEmit(f'Vị trí phao câu đã cài đặt:\n{mMousePos}')

    def CheckRegionEmulator(self):
        mScreenSize = pyautogui.size()
        self.mEmulatorBox = None
        self.mEmulatorWindow = None
        mEmulatorWindows = []
        self.StatusEmit(f'Kích thước màn hình =\n{mScreenSize}')
        try:
            mEmulatorWindows = pyautogui.getWindowsWithTitle(self.mConfig.GetWindowName())
        except:
            self.MsgEmit(f'Không tìm thấy cửa sổ {self.mConfig.GetWindowName()}', False)
            return False
        if len(mEmulatorWindows) > 0:
            self.mEmulatorWindow = mEmulatorWindows[0]
        else:
            self.MsgEmit(f'Không tìm thấy cửa sổ {self.mConfig.GetWindowName()}', False)
            return False
        self.mEmulatorBox = self.mEmulatorWindow.box
        self.StatusEmit(f'Đã tìm thấy cửa sổ giả lập\n{self.mEmulatorBox}')
        if self.mEmulatorBox.width < self.mConfig.GetEmulatorSize()[0] or self.mEmulatorBox.height < \
                self.mConfig.GetEmulatorSize()[1]:
            self.MsgEmit("Cửa sổ giả lập bị ẩn hoặc độ phân giải không phù hợp", False)
            return False
        if self.mEmulatorBox.top < 0:
            self.mEmulatorWindow.activate()
            self.mEmulatorWindow.move(0, abs(self.mEmulatorBox.top))
            self.StatusEmit("Cửa sổ giả lập bị khuất về bên trên\nTự động di chuyển")
        if self.mEmulatorBox.left < 0:
            self.mEmulatorWindow.activate()
            self.mEmulatorWindow.move(abs(self.mEmulatorBox.left), 0)
            self.StatusEmit("Cửa sổ giả lập bị khuất về bên trái\nTự động di chuyển")
        if self.mEmulatorBox.top + self.mEmulatorBox.height > mScreenSize.height:
            self.mEmulatorWindow.activate()
            self.mEmulatorWindow.move(0, 0 - self.mEmulatorBox.top)
            self.StatusEmit("Cửa sổ giả lập bị khuất về bên dưới\nTự động di chuyển")
        if self.mEmulatorBox.left + self.mEmulatorBox.width > mScreenSize.width:
            self.mEmulatorWindow.activate()
            self.mEmulatorWindow.move(0 - self.mEmulatorBox.left, 0)
            self.StatusEmit("Cửa sổ giả lập bị khuất về bên phải\nTự động di chuyển")
        self.mEmulatorBox = self.mEmulatorWindow.box

        # Tọa độ tuyệt đối của nút kéo cần trên màn hình
        self.mAbsPullingRodPos[0] = self.mEmulatorBox.left + self.mConfig.GetPullingRod()[0]
        self.mAbsPullingRodPos[1] = self.mEmulatorBox.top + self.mEmulatorBox.height \
                                    + self.mConfig.GetPullingRod()[1] - self.mConfig.GetEmulatorSize()[1]
        return True

    def StartAdbServer(self):
        self.StatusEmit("Đang khởi tạo adb-server")
        try:
            subprocess.call(f'{self.mConfig.GetCurrentPath()}\\adb\\adb.exe devices', creationflags=CREATE_NO_WINDOW)
        except:
            self.StatusEmit('Khởi tạo adb-server thất bại')
            return False
        self.StatusEmit('Khởi tạo adb-server thành công')
        return True

    def AdbConnect(self):
        mDevices = None
        try:
            mAdbClient = AdbClient(self.mConfig.GetAdbHost(),
                                   self.mConfig.GetAdbPort())
            mDevices = mAdbClient.devices()
        except:
            mCheckStartServer = self.StartAdbServer()
            if mCheckStartServer is False:
                self.MsgEmit('Không tìm thấy adb-server', False)
                return False
            else:
                mAdbClient = AdbClient(self.mConfig.GetAdbHost(),
                                       self.mConfig.GetAdbPort())
                mDevices = mAdbClient.devices()
        if mDevices is None:
            self.MsgEmit('Không kết nối được giả lập qua adb-server\nRestart lại giả lập', False)
            return False

        if len(mDevices) == 0:
            self.MsgEmit('Không kết nối được giả lập qua adb-server\nRestart lại giả lập', False)
            return False
        elif len(mDevices) == 1:
            self.mAdbDevice = mDevices[0]
            self.StatusEmit(f'Kết nối giả lập qua adb-server thành công\nĐịa chỉ giả lập {self.mAdbDevice.serial}')
        else:
            self.MsgEmit("Hãy tắt bớt thiết bị kết nối Adb Server:")
            return False
        return True

    def AdbClick(self, mCoordinateX, mCoordinateY):
        self.mAdbDevice.shell(f'input tap {str(mCoordinateX)} {str(mCoordinateY)}')

    def AdbDoubleClick(self, mCoordinateX, mCoordinateY):
        self.mAdbDevice.shell(
            f'input tap {str(mCoordinateX)} {str(mCoordinateY)} & sleep 0.1; input tap {str(mCoordinateX)} {str(mCoordinateY)}')

    def AdbHoldClick(self, mCoordinateX, mCoordinateY, mTime):
        self.mAdbDevice.shell(
            f'input swipe {str(mCoordinateX)} {str(mCoordinateY)} {str(mCoordinateX)} {str(mCoordinateY)} {str(mTime)}')

    def StartAuto(self):
        self.mAutoFishRunning = True
        if self.mEmulatorBox is None:
            self.MsgEmit("Chưa kết nối phần mềm giả lập", True)
            return

        if self.mMark[0] == 0:
            self.MsgEmit('Chưa xác định vị trí dấu chấm than', False)
            return False

        if self.mConfig.GetFishDetection() is True:
            if self.mFishingRegion[0] == 0:
                self.MsgEmit('Chưa xác định vùng câu', False)
                return False

        time.sleep(1)
        while self.mAutoFishRunning is True:
            time.sleep(1)
            if self.mAutoFishRunning is False:
                break
            self.mFishingNum += 1
            self.mSignalUpdateFishingNum.emit(self.mFishingNum)

            mOutputCastRod = self.CastFishingRod()
            if mOutputCastRod is False:
                continue
            if self.mAutoFishRunning is False:
                break
            mOutPutCheckRod = self.CheckRod()
            if self.mAutoFishRunning is False:
                break
            if mOutPutCheckRod is True:
                mCheckMarkRgb = self.CheckMark()
                if self.mAutoFishRunning is False:
                    break
                if mCheckMarkRgb is True:
                    mPullingRod = self.PullFishingRod()
                    if self.mAutoFishRunning is False:
                        break
                    if mPullingRod is True:
                        self.FishPreservation()
                        if self.mAutoFishRunning is False:
                            break
                        self.mSignalUpdateFishNum.emit(self.mFishNum)
            gc.collect()
        return False
