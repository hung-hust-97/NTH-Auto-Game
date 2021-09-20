import math
import time
import pyautogui
import gc
import cv2
from ppadb.client import Client as AdbClient
import numpy
import win32api
from src.config import *
from PyQt5.QtCore import pyqtSignal, QObject
import subprocess


class AutoFishing(QObject):
    mSignalSetPixelPos = pyqtSignal(int, int)
    mSignalSetFishingBobberPos = pyqtSignal(int, int)
    mSignalUpdateFishingNum = pyqtSignal()
    mSignalUpdateFishNum = pyqtSignal()
    mSignalUpdateFishDetectionImage = pyqtSignal(FishImageColor)
    mSignalMessage = pyqtSignal(str, bool)
    mSignalUpdateStatus = pyqtSignal(str)

    def __init__(self):
        QObject.__init__(self, parent=None)  # Kế thừa QObject
        self.mConfig = Config()
        self.mFishingNum = 0
        self.mAbsPullingRodPos = [0, 0]
        self.mAbsBackPackRegion = [0, 0, 0, 0]
        self.mAbsPreservationRegion = [0, 0, 0, 0]
        self.mMark = [0, 0]
        self.mFishingRegion = [0, 0, 0, 0]
        self.mAdbClient = None
        self.mAdbDevice = None
        self.mAdbDevices = []
        self.mListAdbDevicesSerial = []
        self.mFishDetectionRunning = True
        self.mCheckMouseRunning = False
        self.mAutoFishRunning = False
        self.mCheckFish = False
        self.mEmulatorWindow = None
        self.mEmulatorBox = None
        self.mFishImage = None
        self.mLeu = None
        self.mCheckAdbDelay = 0

        # Khai báo số cá các loại
        self.mAllFish = 0
        self.mVioletFish = 0
        self.mBlueFish = 0
        self.mGreenFish = 0
        self.mGrayFish = 0

        self.mStartTime = time.time()
        self.mSaveTime = 0

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

    @staticmethod
    def ScreenshotWindowRegion(mRegion):
        try:
            mScreenShotPilImage = pyautogui.screenshot(region=(mRegion[0],
                                                               mRegion[1],
                                                               mRegion[2],
                                                               mRegion[3]))
        except:
            return False
        mScreenShotMat = numpy.array(mScreenShotPilImage)
        mScreenShotMat = cv2.cvtColor(mScreenShotMat, cv2.COLOR_BGR2RGB)
        return mScreenShotMat

    @staticmethod
    def ComparePixel(mPixel1: list, mPixel2: list):
        mDiffTotal = 0
        for i in range(3):
            mDiffTotal += abs(int(mPixel1[i]) - int(mPixel2[i]))
        return mDiffTotal / len(mPixel1)

    @staticmethod
    def FindImage(mImagePath: str, mRegion: list, mConfidence: float):
        try:
            mLocate = pyautogui.locateOnScreen(mImagePath, grayscale=True,
                                               region=(mRegion[0],
                                                       mRegion[1],
                                                       mRegion[2],
                                                       mRegion[3]),
                                               confidence=mConfidence)
        except:
            return False
        return mLocate

    # Convert tọa độ tương đối trên giả lập thành tọa độ tuyệt đối trên màn hình PC
    def ConvertCoordinates(self, mRelativePos: list):
        mAbsPos = [0, 0]
        mAbsPos[0] = self.mEmulatorBox.left + mRelativePos[0]
        mAbsPos[1] = self.mEmulatorBox.top + self.mEmulatorBox.height + mRelativePos[1] - \
                     self.mConfig.GetEmulatorSize()[1]
        return mAbsPos

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

    def FixConfirm(self):
        self.AdbClick(self.mConfig.GetConfirm()[0],
                      self.mConfig.GetConfirm()[1])

    def ClickOk(self):
        self.AdbClick(self.mConfig.GetOKButton()[0],
                      self.mConfig.GetOKButton()[1])

    def CheckRod(self):
        time.sleep(self.mConfig.GetDelayTime() + 1.5)
        mCheck = 0
        while mCheck < 5:
            # break point thread auto fishing
            if self.mAutoFishRunning is False:
                return False
            mBackpackPos = self.FindImage(f'{self.mConfig.GetDataPath()}backpack.png',
                                          self.mAbsBackPackRegion,
                                          self.mConfig.GetConfidence())
            if mBackpackPos is False:
                time.sleep(0.2)
                mCheck += 1
                continue
            if mBackpackPos is not None:
                self.FixRod()
                return False
            time.sleep(0.2)
            mCheck += 1
        return True

    def FixRod(self):
        self.StatusEmit("Cần câu bị hỏng. Auto đang sửa cần\n"
                        "Nếu không đúng hỏng cần, hãy cài đặt độ trễ từ 1 giây trở lên")
        self.OpenBackPack()
        if self.mAutoFishRunning is False:
            return
        time.sleep(self.mConfig.GetDelayTime() + 0.5)
        if self.mAutoFishRunning is False:
            return
        self.OpenTools()
        if self.mAutoFishRunning is False:
            return
        time.sleep(self.mConfig.GetDelayTime())
        if self.mAutoFishRunning is False:
            return
        self.FixClick()
        if self.mAutoFishRunning is False:
            return
        time.sleep(self.mConfig.GetDelayTime() + 0.5)
        if self.mAutoFishRunning is False:
            return
        self.FixConfirm()
        if self.mAutoFishRunning is False:
            return
        time.sleep(self.mConfig.GetDelayTime() + 0.5)
        if self.mAutoFishRunning is False:
            return
        self.ClickOk()
        if self.mAutoFishRunning is False:
            return
        time.sleep(self.mConfig.GetDelayTime())
        if self.mAutoFishRunning is False:
            return
        self.CloseBackPack()
        if self.mAutoFishRunning is False:
            return
        time.sleep(self.mConfig.GetDelayTime())
        mCheck = 0
        while mCheck < 5:
            # check point break auto fishing thread
            if self.mAutoFishRunning is False:
                return
            mPreservationButtonPos = self.FindImage(f'{self.mConfig.GetDataPath()}preservation.png',
                                                    self.mAbsPreservationRegion,
                                                    self.mConfig.GetConfidence())
            if mPreservationButtonPos is False:
                mCheck += 1
                time.sleep(0.2)
                continue
            if mPreservationButtonPos is not None:
                self.AdbClick(self.mConfig.GetPreservation()[0],
                              self.mConfig.GetPreservation()[1])
                return
            mCheck += 1
            time.sleep(0.2)
        return

    def CastFishingRod(self):
        mCheck = 0
        while mCheck < 5:
            # break point thread
            if self.mAutoFishRunning is False:
                return False
            mBackpackPos = self.FindImage(f'{self.mConfig.GetDataPath()}backpack.png',
                                          self.mAbsBackPackRegion,
                                          self.mConfig.GetConfidence())
            if mBackpackPos is False:
                mCheck += 1
                time.sleep(0.2)
                continue
            if mBackpackPos is not None:
                self.AdbClick(self.mConfig.GetCastingRod()[0],
                              self.mConfig.GetCastingRod()[1])
                self.StatusEmit("Đã thả cần câu")
                return True
            mCheck += 1
            time.sleep(0.2)

        self.StatusEmit("Không tìm thấy ba lô. Hãy thu cần về và thử lại")
        self.CloseBackPack()

        mCheck = 0
        while mCheck < 5:
            # break point thread
            if self.mAutoFishRunning is False:
                return False
            mPreservationButtonPos = self.FindImage(f'{self.mConfig.GetDataPath()}preservation.png',
                                                    [self.mAbsPreservationRegion[0],
                                                     self.mAbsPreservationRegion[1],
                                                     self.mConfig.mPreservationRec[0],
                                                     self.mConfig.mPreservationRec[1]],
                                                    self.mConfig.GetConfidence())
            if mPreservationButtonPos is False:
                mCheck += 1
                time.sleep(0.2)
                continue

            if mPreservationButtonPos is not None:
                self.AdbClick(self.mConfig.GetPreservation()[0],
                              self.mConfig.GetPreservation()[1])
                return False
            mCheck += 1
            time.sleep(0.2)

        self.FixConfirm()
        time.sleep(self.mConfig.GetDelayTime() + 0.5)
        self.ClickOk()
        return False

    def FishDetection(self, mPrevFrameGray, mCurrFrameGray, mCurrFrameRGB):
        mBackGroundColor = mPrevFrameGray[self.mFishingRegion[3] // 2, self.mFishingRegion[2] // 4]
        # tối ở camp 49  # tối ở biển 57
        if mBackGroundColor <= 70:
            mMinThreshValue = 10
            mMaxThreshValue = 100
        # buổi chiều nền biền 74, sáng ở camp 149, chiều ở cam 166
        elif 70 < mBackGroundColor < 170:
            mMinThreshValue = 30
            mMaxThreshValue = 100
        # buổi sáng nền biển 174
        else:
            mMinThreshValue = 50
            mMaxThreshValue = 100

        mCurrImgArrWidth, mCurrImgArrHeight = mCurrFrameGray.shape
        mImgCenterX = mCurrImgArrWidth // 2
        mImgCenterY = mCurrImgArrHeight // 2

        mPrevFrameBlur = cv2.GaussianBlur(mPrevFrameGray, (21, 21), 0)
        mCurrFrameBlur = cv2.GaussianBlur(mCurrFrameGray, (21, 21), 0)

        # so sánh 2 frame, tìm sai khác
        mFrameDelta = cv2.absdiff(mPrevFrameBlur, mCurrFrameBlur)
        mThresh = cv2.threshold(mFrameDelta, mMinThreshValue, mMaxThreshValue, cv2.THRESH_BINARY)[1]

        # Fill in holes via dilate()
        mThresh = cv2.dilate(mThresh, None, iterations=2)

        # Tìm đường biên contours, hierarchy
        mContours, mHierarchy = cv2.findContours(mThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Quét tất cả các đường biên
        mFishArea = 0
        mCurrFrameRGB = cv2.circle(mCurrFrameRGB, (mImgCenterX, mImgCenterY),
                                   int(self.mConfig.GetRadiusFishingRegion() * 3 // 4), TEXT_COLOR, 1)
        mCurrFrameRGB = cv2.circle(mCurrFrameRGB, (mImgCenterX, mImgCenterY),
                                   self.mConfig.GetRadiusFishingRegion() * 1 // 4,
                                   TEXT_COLOR, 1)
        cv2.putText(mCurrFrameRGB, str(mBackGroundColor),
                    (int(10 * self.mConfig.mFontScale),
                     int(40 * self.mConfig.mFontScale)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.mConfig.mFontScale, TEXT_COLOR, 1)
        for mContour in mContours:
            # check coordinates of all found contours
            (x, y, w, h) = cv2.boundingRect(mContour)
            mContourCenterX = x + w // 2
            mContourCenterY = y + h // 2
            mRadius = math.sqrt(pow((mImgCenterX - mContourCenterX), 2) + pow((mImgCenterY - mContourCenterY), 2))
            # loại bỏ phao câu
            if mRadius < self.mConfig.GetRadiusFishingRegion() / 4:
                continue
            # loại bỏ box xuất hiện ở viền
            if mRadius > self.mConfig.GetRadiusFishingRegion() * 3 / 4:
                continue

            # loại box nhỏ tránh nhiễu
            if cv2.contourArea(mContour) < self.mConfig.GetMinContour():
                continue

            # loai box qua to
            if cv2.contourArea(mContour) > self.mConfig.mMaxContour:
                continue

            mFishArea = int(cv2.contourArea(mContour))
            cv2.rectangle(mCurrFrameRGB, (x, y), (x + w, y + h), TEXT_COLOR, 1)
            cv2.putText(mCurrFrameRGB, str(mFishArea), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, self.mConfig.mFontScale, TEXT_COLOR, 1)

            break
        self.mFishImage = mCurrFrameRGB.copy()
        if self.mConfig.GetShowFishShadow() is True:
            self.mSignalUpdateFishDetectionImage.emit(FishImageColor.RGB)
        return mFishArea

    def ScreenshotFishingRegion(self):
        try:
            mScreenShotPilImage = pyautogui.screenshot(region=(self.mFishingRegion[0],
                                                               self.mFishingRegion[1],
                                                               self.mFishingRegion[2],
                                                               self.mFishingRegion[3]))
        except:
            return False, False
        mScreenShotMat = numpy.array(mScreenShotPilImage)
        mScreenShotMatRGB = cv2.cvtColor(mScreenShotMat, cv2.COLOR_BGR2RGB)
        mScreenShotMatGray = cv2.cvtColor(mScreenShotMat, cv2.COLOR_RGB2GRAY)
        return mScreenShotMatGray, mScreenShotMatRGB

    def CheckMark(self):
        time.sleep(3)
        mStaticFrameGray = None
        if self.mConfig.GetFishDetection() is True:
            mStaticFrameGray, mStaticFrameRGB = self.ScreenshotFishingRegion()
            if mStaticFrameRGB is False:
                return False
            self.mFishImage = mStaticFrameRGB
        self.StatusEmit("Đang đợi cá")
        # Chụp ảnh màn hình vùng chứa pixel chấm than
        mMarkRGB = self.ScreenshotWindowRegion([self.mMark[0] - 1, self.mMark[1] - 1, 3, 3])
        if mMarkRGB is False:
            return False
        mPixelBase = mMarkRGB[1, 1]
        time1 = time.time()
        time2 = time.time()
        mStopDetect = False
        mSkipFrame = 0
        while (time2 - time1) < self.mConfig.GetWaitingFishTime():
            # break point thread
            if self.mAutoFishRunning is False:
                return False

            if self.mConfig.GetFishDetection() is True and mStopDetect is False:
                mCurrentFrameGray, mCurrentFrameRGB = self.ScreenshotFishingRegion()
                if mCurrentFrameGray is False:
                    return False
                mSizeFish = self.FishDetection(mStaticFrameGray, mCurrentFrameGray, mCurrentFrameRGB)
                if mSizeFish != 0:
                    mSkipFrame += 1
                if mSkipFrame == 10:
                    mStopDetect = True
                    if mSizeFish < int(self.mConfig.GetFishSize()):
                        return True
            # Chụp ảnh màn hình vùng chứa pixel chấm than
            mMarkRGBCurr = self.ScreenshotWindowRegion([self.mMark[0] - 1, self.mMark[1] - 1, 3, 3])
            if mMarkRGBCurr is False:
                time2 = time.time()
                time.sleep(0.02)
                continue
            mPixelCurr = mMarkRGBCurr[1, 1]
            mDiffRgb = self.ComparePixel(mPixelCurr, mPixelBase)
            if mDiffRgb > self.mConfig.GetDifferentColor():
                return True
            time2 = time.time()
            time.sleep(0.03)

        self.StatusEmit("Không phát hiện dấu chấm than")
        return False

    def PullFishingRod(self):
        if self.mConfig.GetFreeMouse() is False:
            try:
                pyautogui.click(x=self.mAbsPullingRodPos[0],
                                y=self.mAbsPullingRodPos[1],
                                clicks=2,
                                interval=0.1,
                                button='left')
            except:
                return False
            self.StatusEmit("Đang kéo cần câu")
            return True
        else:
            self.StatusEmit("Đang kéo cần câu")
            time1 = time.time()
            self.AdbClick(self.mConfig.GetPullingRod()[0],
                          self.mConfig.GetPullingRod()[1])
            timeDelay = time.time() - time1
            self.StatusEmit(f'Độ trễ giật cần {round(timeDelay, 2)} giây')
            if timeDelay > 0.5:
                self.mCheckAdbDelay += 1
                if self.mCheckAdbDelay <= 3:
                    return True
                self.mAutoFishRunning = False
                self.MsgEmit(
                    f'Độ trễ truyền lệnh điều khiển giả lập qua Adb Server quá cao trên 0.5 giây\nTắt chế độ "Chuột tự do" để không bị kéo hụt cá',
                    False)
                self.mCheckAdbDelay = 0
            return True

    def FishPreservation(self):
        time.sleep(0.1)
        if self.mAutoFishRunning is False:
            return False
        time1 = time.time()
        time2 = time.time()
        while (time2 - time1) < int(self.mConfig.GetPullingFishTime()):
            # check point break auto fishing thread
            if self.mAutoFishRunning is False:
                return
            mBackpackPos = self.FindImage(f'{self.mConfig.GetDataPath()}backpack.png',
                                          self.mAbsBackPackRegion,
                                          self.mConfig.GetConfidence())
            if mBackpackPos is False:
                time.sleep(0.1)
                continue

            if mBackpackPos is not None:
                self.StatusEmit("Câu thất bại")
                # Hiện ảnh cá câu được lên app auto
                # if self.mConfig.GetShowFishShadow() is True:
                #     self.mSignalUpdateFishDetectionImage.emit(FishImageColor.LEU)
                return True

            mPreservationPos = self.FindImage(f'{self.mConfig.GetDataPath()}preservation.png',
                                              [self.mAbsPreservationRegion[0],
                                               self.mAbsPreservationRegion[1],
                                               self.mConfig.mPreservationRec[0], self.mConfig.mPreservationRec[1]],
                                              self.mConfig.GetConfidence())
            if mPreservationPos is False:
                time.sleep(0.1)
                continue

            if mPreservationPos is not None:
                self.StatusEmit("Câu thành công")
                self.FishCount()
                self.AdbClick(self.mConfig.GetPreservation()[0],
                              self.mConfig.GetPreservation()[1])
                return True
            time.sleep(0.1)
            time2 = time.time()
            if self.mAutoFishRunning is False:
                return False
        self.StatusEmit("Kiểm tra kết quả bị lỗi")
        return False

    def FishCount(self):
        mFishImagePos = self.ConvertCoordinates([self.mConfig.mFishImgRegion[0], self.mConfig.mFishImgRegion[1]])
        mFishImageRegion = [mFishImagePos[0], mFishImagePos[1], self.mConfig.mFishImgRegion[2],
                            self.mConfig.mFishImgRegion[3]]
        mFishImage = self.ScreenshotWindowRegion(mFishImageRegion)
        if mFishImage is False:
            return False
        self.mAllFish += 1
        mPixelCheckTypeFishPosition = [self.mConfig.mCheckTypeFishPos[0] - self.mConfig.mFishImgRegion[0],
                                       self.mConfig.mCheckTypeFishPos[1] - self.mConfig.mFishImgRegion[1]]
        mPixelCheckTypeFish = mFishImage[mPixelCheckTypeFishPosition[1],
                                         mPixelCheckTypeFishPosition[0]]
        if self.ComparePixel(mPixelCheckTypeFish, VIOLET_FISH_COLOR_BGR) < 10:
            self.mVioletFish += 1
        elif self.ComparePixel(mPixelCheckTypeFish, BLUE_FISH_COLOR_BGR) < 10:
            self.mBlueFish += 1
        elif self.ComparePixel(mPixelCheckTypeFish, GREEN_FISH_COLOR_BGR) < 10:
            self.mGreenFish += 1
        elif self.ComparePixel(mPixelCheckTypeFish, GRAY_FISH_COLOR_BGR) < 10:
            self.mGrayFish += 1
        else:
            pass

        # Debug vi tri xac dinh mau sac ca
        # mFishImage = cv2.circle(mFishImage, (mPixelCheckTypeFishPosition[0], mPixelCheckTypeFishPosition[1]), 3,
        #                         TEXT_COLOR, 1, cv2.LINE_AA)
        self.mFishImage = mFishImage.copy()

        # Hiện ảnh cá câu được lên app auto
        if self.mConfig.GetShowFishShadow() is True:
            self.mSignalUpdateFishDetectionImage.emit(FishImageColor.RGB)

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
        if self.mFishingRegion[0] < 0 or self.mFishingRegion[1] < 0 \
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
        self.mEmulatorWindow.activate()
        self.StatusEmit(f'Đã tìm thấy cửa sổ giả lập\n{self.mEmulatorBox}')
        mEmulatorSize = self.mConfig.GetEmulatorSize()
        if mEmulatorSize[0] * 0.9 > self.mEmulatorBox.width or \
                self.mEmulatorBox.width > mEmulatorSize[0] * 1.1 or \
                mEmulatorSize[1] * 0.9 > self.mEmulatorBox.height or \
                self.mEmulatorBox.height > mEmulatorSize[1] * 1.1:
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
            self.mEmulatorWindow.move(0, mScreenSize.height - (self.mEmulatorBox.top + self.mEmulatorBox.height))
            self.StatusEmit("Cửa sổ giả lập bị khuất về bên dưới\nTự động di chuyển")
        if self.mEmulatorBox.left + self.mEmulatorBox.width > mScreenSize.width:
            self.mEmulatorWindow.activate()
            self.mEmulatorWindow.move(mScreenSize.width - (self.mEmulatorBox.left + self.mEmulatorBox.width), 0)
            self.StatusEmit("Cửa sổ giả lập bị khuất về bên phải\nTự động di chuyển")
        self.mEmulatorBox = self.mEmulatorWindow.box

        # Tọa độ tuyệt đối
        self.mAbsPullingRodPos = self.ConvertCoordinates(self.mConfig.GetPullingRod())
        mAbsBackpackPos = self.ConvertCoordinates(self.mConfig.GetOpenBackPack())
        self.mAbsBackPackRegion[0] = mAbsBackpackPos[0] - self.mConfig.mBackpackRec[0] // 2
        self.mAbsBackPackRegion[1] = mAbsBackpackPos[1] - self.mConfig.mBackpackRec[1] // 2
        self.mAbsBackPackRegion[2] = self.mConfig.mBackpackRec[0]
        self.mAbsBackPackRegion[3] = self.mConfig.mBackpackRec[1]

        mAbsPreservationPos = self.ConvertCoordinates(self.mConfig.GetPreservation())
        self.mAbsPreservationRegion[2] = self.mConfig.mPreservationRec[0]
        self.mAbsPreservationRegion[3] = self.mConfig.mPreservationRec[1]
        self.mAbsPreservationRegion[0] = mAbsPreservationPos[0] - self.mConfig.mPreservationRec[0] // 2
        self.mAbsPreservationRegion[1] = mAbsPreservationPos[1] - self.mConfig.mPreservationRec[1] // 2

        # Troll
        self.mLeu = cv2.imread(f'{self.mConfig.GetDataPath()}{LEULEU}', cv2.IMREAD_COLOR)
        self.mLeu = cv2.resize(self.mLeu, (200, 200), interpolation=cv2.INTER_AREA)

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

    def AdbServerConnect(self):
        self.mAdbDevices = []
        self.mListAdbDevicesSerial = []
        try:
            self.mAdbClient = AdbClient(self.mConfig.GetAdbHost(),
                                        self.mConfig.GetAdbPort())
            self.mAdbDevices = self.mAdbClient.devices()
        except:
            mCheckStartServer = self.StartAdbServer()
            if mCheckStartServer is False:
                self.MsgEmit('Không tìm thấy adb-server', False)
                return False
            else:
                self.mAdbClient = AdbClient(self.mConfig.GetAdbHost(),
                                            self.mConfig.GetAdbPort())
                self.mAdbDevices = self.mAdbClient.devices()
        if len(self.mAdbDevices) == 0:
            self.mAdbClient = None
            self.MsgEmit('Không kết nối được giả lập qua adb-server\nRestart lại giả lập', False)
            return False
        else:
            for tempDevice in self.mAdbDevices:
                self.mListAdbDevicesSerial.append(tempDevice.serial)
        return True

    def AdbDeviceConnect(self):
        for index in range(len(self.mAdbDevices)):
            if self.mAdbDevices[index].serial == self.mConfig.GetAdbAddress():
                self.mAdbDevice = self.mAdbDevices[index]
                break
        if self.mAdbDevice is None:
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
            self.MsgEmit("Chưa kết nối cửa sổ giả lập", True)
            return

        if self.mAdbDevice is None:
            self.MsgEmit("Chưa kết nối địa chỉ Adb của thiết bị", True)
            return

        if self.mMark[0] == 0:
            self.MsgEmit('Chưa xác định vị trí dấu chấm than', False)
            return False

        if self.mConfig.GetFishDetection() is True:
            if self.mFishingRegion[0] == 0:
                self.MsgEmit('Chưa xác định vùng câu', False)
                return False

        time.sleep(self.mConfig.GetDelayTime())
        while self.mAutoFishRunning is True:
            time.sleep(self.mConfig.GetDelayTime())
            if self.mAutoFishRunning is False:
                break
            mOutputCastRod = self.CastFishingRod()
            if mOutputCastRod is False:
                continue

            self.mFishingNum += 1
            self.mSignalUpdateFishingNum.emit()

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
                        self.mSignalUpdateFishNum.emit()
            gc.collect()
        return False
