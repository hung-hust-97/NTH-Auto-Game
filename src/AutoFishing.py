import math
import time
import pyautogui
import gc
import cv2
from ppadb.client import Client as AdbClient
import numpy
import win32api
from src.config import Config
from PyQt5.QtCore import pyqtSignal, QObject
import subprocess


class AutoFishing(QObject):
    mSignalSetPixelPos = pyqtSignal(int, int)
    mSignalSetFishingBobberPos = pyqtSignal(int, int)
    mSignalUpdateFishingNum = pyqtSignal()
    mSignalUpdateFishNum = pyqtSignal()
    mSignalUpdateImageShow = pyqtSignal()
    mSignalMessage = pyqtSignal(str)
    mSignalUpdateStatus = pyqtSignal(str)

    def __init__(self):
        QObject.__init__(self, parent=None)  # Kế thừa QObject
        self.mConfig = Config()
        self.mFishingNum = 0
        self.mAbsPullingRodPos = [0, 0]
        self.mAbsBackPackRegion = [0, 0, 0, 0]
        self.mAbsPreservationRegion = [0, 0, 0, 0]
        self.mMark = [0, 0]
        self.mScreenSize = [0, 0]
        self.mFishingRegion = [0, 0, 0, 0]
        self.mAdbClient = None
        self.mAdbDevice = None
        self.mAdbDevices = []
        self.mListAdbDevicesSerial = []
        self.mCheckMouseRunning = False
        self.mAutoFishRunning = False
        self.mCheckFish = False
        self.mEmulatorWindow = None
        self.mEmulatorBox = None
        self.mImageShow = None
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
        print(59, self.mAutoFishRunning)
        self.mCheckMouseRunning = False
        self.mAutoFishRunning = False

    def MsgEmit(self, mText: str):
        self.mSignalMessage.emit(mText)

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
        except (ValueError, Exception):
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
        except (ValueError, Exception):
            return False
        return mLocate

    # Convert tọa độ tương đối trên giả lập thành tọa độ tuyệt đối trên màn hình PC
    def ConvertCoordinates(self, mRelativePos: list):
        mAbsPos = [0, 0]
        mAbsPos[0] = self.mEmulatorBox.left + mRelativePos[0]
        mAbsPos[1] = self.mEmulatorBox.top + self.mEmulatorBox.height + mRelativePos[1] - \
                     self.mConfig.mEmulatorSize[1]
        return mAbsPos

    def CloseBackPack(self):
        self.AdbClick(self.mConfig.mCloseBackPack[0],
                      self.mConfig.mCloseBackPack[1])
        return True

    def OpenTools(self):
        self.AdbClick(self.mConfig.mTools[0],
                      self.mConfig.mTools[1])
        return True

    def OpenBackPack(self):
        self.AdbClick(self.mConfig.mOpenBackPack[0],
                      self.mConfig.mOpenBackPack[1])

    def FixClick(self):
        self.AdbClick(self.mConfig.mListFishingRodPosition[self.mConfig.mFishingRodIndex][0],
                      self.mConfig.mListFishingRodPosition[self.mConfig.mFishingRodIndex][1])

    def FixConfirm(self):
        self.AdbClick(self.mConfig.mConfirm[0],
                      self.mConfig.mConfirm[1])

    def ClickOk(self):
        self.AdbClick(self.mConfig.mOKButton[0],
                      self.mConfig.mOKButton[1])

    def CheckRod(self):
        time.sleep(self.mConfig.mDelayTime + 1.5)
        mCheck = 0
        while mCheck < 5:
            # break point thread auto fishing
            if self.mAutoFishRunning is False:
                return False
            mBackpackPos = self.FindImage(self.mConfig.mBackpackImgPath,
                                          self.mAbsBackPackRegion,
                                          self.mConfig.mConfidence)
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
        time.sleep(self.mConfig.mDelayTime + 0.5)
        if self.mAutoFishRunning is False:
            return
        self.OpenTools()
        if self.mAutoFishRunning is False:
            return
        time.sleep(self.mConfig.mDelayTime)
        if self.mAutoFishRunning is False:
            return
        self.FixClick()
        if self.mAutoFishRunning is False:
            return
        time.sleep(self.mConfig.mDelayTime + 0.5)
        if self.mAutoFishRunning is False:
            return
        self.FixConfirm()
        if self.mAutoFishRunning is False:
            return
        time.sleep(self.mConfig.mDelayTime + 0.5)
        if self.mAutoFishRunning is False:
            return
        self.ClickOk()
        if self.mAutoFishRunning is False:
            return
        time.sleep(self.mConfig.mDelayTime)
        if self.mAutoFishRunning is False:
            return
        self.CloseBackPack()
        if self.mAutoFishRunning is False:
            return
        time.sleep(self.mConfig.mDelayTime)
        mCheck = 0
        while mCheck < 5:
            # check point break auto fishing thread
            if self.mAutoFishRunning is False:
                return
            mPreservationButtonPos = self.FindImage(self.mConfig.mPreservationImgPath,
                                                    self.mAbsPreservationRegion,
                                                    self.mConfig.mConfidence)
            if mPreservationButtonPos is False:
                mCheck += 1
                time.sleep(0.2)
                continue
            if mPreservationButtonPos is not None:
                self.AdbClick(self.mConfig.mPreservationPos[0],
                              self.mConfig.mPreservationPos[1])
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
            mBackpackPos = self.FindImage(self.mConfig.mBackpackImgPath,
                                          self.mAbsBackPackRegion,
                                          self.mConfig.mConfidence)
            if mBackpackPos is False:
                mCheck += 1
                time.sleep(0.2)
                continue
            if mBackpackPos is not None:
                self.AdbClick(self.mConfig.mCastingRodPos[0],
                              self.mConfig.mCastingRodPos[1])
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
            mPreservationButtonPos = self.FindImage(self.mConfig.mPreservationImgPath,
                                                    self.mAbsPreservationRegion,
                                                    self.mConfig.mConfidence)
            if mPreservationButtonPos is False:
                mCheck += 1
                time.sleep(0.2)
                continue

            if mPreservationButtonPos is not None:
                self.AdbClick(self.mConfig.mPreservationPos[0],
                              self.mConfig.mPreservationPos[1])
                return False
            mCheck += 1
            time.sleep(0.2)

        self.FixConfirm()
        time.sleep(self.mConfig.mDelayTime + 0.5)
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

        mPrevFrameBlur = cv2.GaussianBlur(mPrevFrameGray, (self.mConfig.mBlur, self.mConfig.mBlur), 0)
        mCurrFrameBlur = cv2.GaussianBlur(mCurrFrameGray, (self.mConfig.mBlur, self.mConfig.mBlur), 0)

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
                                   int(self.mConfig.mRadiusFishingRegion * 3 // 4),
                                   self.mConfig.mTextColor, 1)
        mCurrFrameRGB = cv2.circle(mCurrFrameRGB, (mImgCenterX, mImgCenterY),
                                   self.mConfig.mRadiusFishingRegion * 1 // 4,
                                   self.mConfig.mTextColor, 1)
        cv2.putText(mCurrFrameRGB, str(mBackGroundColor),
                    (int(10 * self.mConfig.mFontScale),
                     int(40 * self.mConfig.mFontScale)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.mConfig.mFontScale,
                    self.mConfig.mTextColor, 1)
        for mContour in mContours:
            # break point thread
            if self.mAutoFishRunning is False:
                return False

            # check coordinates of all found contours
            (x, y, w, h) = cv2.boundingRect(mContour)
            mContourCenterX = x + w // 2
            mContourCenterY = y + h // 2
            mRadius = math.sqrt(pow((mImgCenterX - mContourCenterX), 2) + pow((mImgCenterY - mContourCenterY), 2))
            # loại bỏ phao câu
            if mRadius < self.mConfig.mRadiusFishingRegion / 4:
                continue
            # loại bỏ box xuất hiện ở viền
            if mRadius > self.mConfig.mRadiusFishingRegion * 3 / 4:
                continue
            # loại box nhỏ tránh nhiễu
            if cv2.contourArea(mContour) < self.mConfig.mMinContour * self.mConfig.mWindowRatio:
                continue
            # loai box qua to
            if cv2.contourArea(mContour) > self.mConfig.mMaxContour * self.mConfig.mWindowRatio:
                continue
            mFishArea = int(cv2.contourArea(mContour))
            cv2.rectangle(mCurrFrameRGB, (x, y), (x + w, y + h), self.mConfig.mTextColor, 1)
            cv2.putText(mCurrFrameRGB, str(mFishArea), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, self.mConfig.mFontScale, self.mConfig.mTextColor, 1)
            break
        self.mImageShow = mCurrFrameRGB.copy()
        if self.mConfig.mShowFishCheck is True:
            self.mSignalUpdateImageShow.emit()
        return mFishArea

    def ScreenshotFishingRegion(self):
        try:
            mScreenShotPilImage = pyautogui.screenshot(region=(self.mFishingRegion[0],
                                                               self.mFishingRegion[1],
                                                               self.mFishingRegion[2],
                                                               self.mFishingRegion[3]))
        except (ValueError, Exception):
            return False, False
        mScreenShotMat = numpy.array(mScreenShotPilImage)
        mScreenShotMatRGB = cv2.cvtColor(mScreenShotMat, cv2.COLOR_BGR2RGB)
        mScreenShotMatGray = cv2.cvtColor(mScreenShotMat, cv2.COLOR_RGB2GRAY)
        return mScreenShotMatGray, mScreenShotMatRGB

    def CheckMark(self):
        for i in range(10):
            time.sleep(0.5)
            # break point thread
            if self.mAutoFishRunning is False:
                return False

        mStaticFrameGray = None
        if self.mConfig.mFishDetectionCheck is True:
            mStaticFrameGray, mStaticFrameRGB = self.ScreenshotFishingRegion()
            if mStaticFrameRGB is False:
                return False
            self.mImageShow = mStaticFrameRGB
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

        while (time2 - time1) < self.mConfig.mWaitingFishTime:
            # break point thread
            if self.mAutoFishRunning is False:
                return False

            if self.mConfig.mFishDetectionCheck is True and mStopDetect is False:
                mCurrentFrameGray, mCurrentFrameRGB = self.ScreenshotFishingRegion()
                if mCurrentFrameGray is False:
                    return False
                mSizeFish = self.FishDetection(mStaticFrameGray, mCurrentFrameGray, mCurrentFrameRGB)
                if mSizeFish != 0:
                    mSkipFrame += 1
                if mSkipFrame == 5:
                    mStopDetect = True
                    if mSizeFish < self.mConfig.mFishSize:
                        return True
            # Chụp ảnh màn hình vùng chứa pixel chấm than
            mMarkRGBCurr = self.ScreenshotWindowRegion([self.mMark[0] - 1, self.mMark[1] - 1, 3, 3])
            if mMarkRGBCurr is False:
                time2 = time.time()
                time.sleep(0.005)
                continue
            mPixelCurr = mMarkRGBCurr[1, 1]
            mDiffRgb = self.ComparePixel(mPixelCurr, mPixelBase)

            if self.mConfig.mShowFishCheck is True:
                if self.mConfig.mFishDetectionCheck is False:
                    mImageW = self.mImageShow.shape[0]
                    mImageH = self.mImageShow.shape[1]
                    mRegion = [self.mMark[0] - 20, self.mMark[1] - 20, 40, 40]
                    mTempImage = self.ScreenshotWindowRegion(mRegion)
                    mTempImage = cv2.circle(mTempImage, (20, 20), 1, (0, 0, 255), 2, cv2.LINE_AA)
                    mTempImage = cv2.rectangle(mTempImage, (0, 0), (mTempImage.shape[0] - 1, mTempImage.shape[1] - 1),
                                               self.mConfig.mTextColor, 1, cv2.LINE_AA)
                    mTempImage = cv2.resize(mTempImage, (mImageW // 4, mImageH // 4), interpolation=cv2.INTER_AREA)
                    self.mImageShow[0:mTempImage.shape[0], 0: mTempImage.shape[1]] = mTempImage
                else:
                    if mStopDetect is True:
                        mImageW = self.mImageShow.shape[0]
                        mImageH = self.mImageShow.shape[1]
                        mRegion = [self.mMark[0] - 20, self.mMark[1] - 20, 40, 40]
                        mTempImage = self.ScreenshotWindowRegion(mRegion)
                        mTempImage = cv2.circle(mTempImage, (20, 20), 1, (0, 0, 255), 2, cv2.LINE_AA)
                        mTempImage = cv2.rectangle(mTempImage, (0, 0),
                                                   (mTempImage.shape[0] - 1, mTempImage.shape[1] - 1),
                                                   self.mConfig.mTextColor, 1, cv2.LINE_AA)
                        mTempImage = cv2.resize(mTempImage, (mImageW // 4, mImageH // 4), interpolation=cv2.INTER_AREA)
                        self.mImageShow[0:mTempImage.shape[0], 0: mTempImage.shape[1]] = mTempImage
                self.mSignalUpdateImageShow.emit()

            if mDiffRgb > self.mConfig.mDifferentColor:
                return True
            time2 = time.time()
            time.sleep(0.005)

        self.StatusEmit("Không phát hiện dấu chấm than")
        return False

    def PullFishingRod(self):
        if self.mConfig.mFreeMouseCheck is False:
            try:
                pyautogui.click(x=self.mAbsPullingRodPos[0],
                                y=self.mAbsPullingRodPos[1],
                                clicks=2,
                                interval=0.1,
                                button='left')
            except (ValueError, Exception):
                return False
            self.StatusEmit("Đang kéo cần câu")
            return True
        else:
            time1 = time.time()
            self.AdbClick(self.mConfig.mPullingRodPos[0],
                          self.mConfig.mPullingRodPos[1])
            timeDelay = time.time() - time1
            self.StatusEmit(f'Đang kéo cần câu. Độ trễ giật cần {round(timeDelay, 2)} giây')
            if timeDelay > 0.5:
                self.mCheckAdbDelay += 1
                if self.mCheckAdbDelay <= 3:
                    return True
                self.mAutoFishRunning = False
                self.MsgEmit(
                    f'Độ trễ truyền lệnh điều khiển giả lập qua Adb Server quá cao trên 0.5 giây\nTắt chế độ "Chuột tự do" để không bị kéo hụt cá')
                self.mCheckAdbDelay = 0
            return True

    def FishPreservation(self):
        time.sleep(0.1)
        if self.mAutoFishRunning is False:
            return False
        time1 = time.time()
        time2 = time.time()
        while (time2 - time1) < int(self.mConfig.mPullingFishTime):
            # check point break auto fishing thread
            if self.mAutoFishRunning is False:
                return
            mBackpackPos = self.FindImage(self.mConfig.mBackpackImgPath,
                                          self.mAbsBackPackRegion,
                                          self.mConfig.mConfidence)
            if mBackpackPos is False:
                time.sleep(0.1)
                continue

            if mBackpackPos is not None:
                self.StatusEmit("Câu thất bại")
                return True

            mPreservationPos = self.FindImage(self.mConfig.mPreservationImgPath,
                                              self.mAbsPreservationRegion,
                                              self.mConfig.mConfidence)
            if mPreservationPos is False:
                time.sleep(0.1)
                continue

            if mPreservationPos is not None:
                self.StatusEmit("Câu thành công")
                self.FishCount()
                self.AdbClick(self.mConfig.mPreservationPos[0],
                              self.mConfig.mPreservationPos[1])
                return True
            time.sleep(0.1)
            time2 = time.time()
            if self.mAutoFishRunning is False:
                return False
        self.StatusEmit("Kiểm tra kết quả bị lỗi")
        return False

    def FishCount(self):
        time.sleep(0.2)
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
        if self.ComparePixel(mPixelCheckTypeFish, self.mConfig.mVioletColorBGR) < 10:
            self.mVioletFish += 1
        elif self.ComparePixel(mPixelCheckTypeFish, self.mConfig.mBlueColorBGR) < 10:
            self.mBlueFish += 1
        elif self.ComparePixel(mPixelCheckTypeFish, self.mConfig.mGreenColorBGR) < 10:
            self.mGreenFish += 1
        elif self.ComparePixel(mPixelCheckTypeFish, self.mConfig.mGrayColorBGR) < 10:
            self.mGrayFish += 1
        else:
            pass

        # Debug vi tri xac dinh mau sac ca
        # mImageShow = cv2.circle(mImageShow, (mPixelCheckTypeFishPosition[0], mPixelCheckTypeFishPosition[1]), 3,
        #                         TEXT_COLOR, 1, cv2.LINE_AA)
        self.mImageShow = mFishImage.copy()

        # Hiện ảnh cá câu được lên app auto
        if self.mConfig.mShowFishCheck is True:
            self.mSignalUpdateImageShow.emit()

    def SetPixelPos(self):
        if self.mEmulatorBox is None:
            self.MsgEmit("Chưa kết nối cửa sổ giả lập")
            return
        if self.mAdbDevice is None:
            self.MsgEmit("Chưa kết nối địa chỉ Adb của thiết bị")
            return
        self.mMark = [0, 0]
        time.sleep(0.1)
        mMousePos = pyautogui.position()
        self.StatusEmit("Di chuyển chuột đến đầu của dấu chấm than và Click")
        self.mCheckMouseRunning = True
        while self.mCheckMouseRunning is True:
            mMousePos = pyautogui.position()
            self.mSignalSetPixelPos.emit(mMousePos.x, mMousePos.y)
            if self.mScreenSize[0] - 50 > mMousePos.x > 50 and \
                    self.mScreenSize[1] - 50 > mMousePos.y > 50:
                mRegion = [mMousePos.x - 50, mMousePos.y - 50, 100, 100]
                mTempImage = self.ScreenshotWindowRegion(mRegion)
                mTempImage = cv2.circle(mTempImage, (50, 50), 1, (0, 0, 255), 2, cv2.LINE_AA)
                self.mImageShow = mTempImage.copy()
                self.mSignalUpdateImageShow.emit()
            if self.CheckLeftMouseClick() is True:
                self.mCheckMouseRunning = False
            time.sleep(0.01)
        self.mMark[0] = mMousePos.x
        self.mMark[1] = mMousePos.y
        self.StatusEmit(f'Vị trí dấu chấm than đã cài đặt:\n{mMousePos}')

    def SetFishingBobberPos(self):
        if self.mEmulatorBox is None:
            self.MsgEmit("Chưa kết nối cửa sổ giả lập")
            return

        if self.mAdbDevice is None:
            self.MsgEmit("Chưa kết nối địa chỉ Adb của thiết bị")
            return
        self.mFishingRegion = [0, 0, 0, 0]
        time.sleep(0.1)
        self.mScreenSize = pyautogui.size()
        mMousePos = pyautogui.position()
        self.StatusEmit("Di chuyển chuột đến phao câu và Click")
        self.mCheckMouseRunning = True
        while self.mCheckMouseRunning is True:
            mMousePos = pyautogui.position()
            self.mSignalSetFishingBobberPos.emit(mMousePos.x, mMousePos.y)
            if self.mScreenSize[0] - 50 > mMousePos.x > 50 and \
                    self.mScreenSize[1] - 50 > mMousePos.y > 50:
                mRegion = [mMousePos.x - 50, mMousePos.y - 50, 100, 100]
                mTempImage = self.ScreenshotWindowRegion(mRegion)
                mTempImage = cv2.circle(mTempImage, (50, 50), 1, (0, 0, 255), 2, cv2.LINE_AA)
                self.mImageShow = mTempImage.copy()
                self.mSignalUpdateImageShow.emit()
            if self.CheckLeftMouseClick() is True:
                self.mCheckMouseRunning = False
            time.sleep(0.01)
        self.mFishingRegion[0] = mMousePos.x - self.mConfig.mRadiusFishingRegion
        self.mFishingRegion[1] = mMousePos.y - self.mConfig.mRadiusFishingRegion
        self.mFishingRegion[2] = self.mConfig.mRadiusFishingRegion * 2
        self.mFishingRegion[3] = self.mConfig.mRadiusFishingRegion * 2
        if self.mFishingRegion[0] < 0 or self.mFishingRegion[1] < 0 \
                or self.mFishingRegion[0] + self.mFishingRegion[2] > self.mScreenSize.width \
                or self.mFishingRegion[1] + self.mFishingRegion[3] > self.mScreenSize.height:
            self.MsgEmit("Vị trí phao câu quá gần viền màn hình")
        self.StatusEmit(f'Vị trí phao câu đã cài đặt:\n{mMousePos}')

    def CheckRegionEmulator(self):
        self.mScreenSize = pyautogui.size()
        self.mEmulatorBox = None
        self.mEmulatorWindow = None
        self.StatusEmit(f'Kích thước màn hình =\n{self.mScreenSize}')
        try:
            mEmulatorWindows = pyautogui.getWindowsWithTitle(self.mConfig.mWindowName)
        except (ValueError, Exception):
            self.MsgEmit(f'Không tìm thấy cửa sổ {self.mConfig.mWindowName}')
            return False
        if len(mEmulatorWindows) > 0:
            self.mEmulatorWindow = mEmulatorWindows[0]
        else:
            self.MsgEmit(f'Không tìm thấy cửa sổ {self.mConfig.mWindowName}')
            return False
        self.mEmulatorBox = self.mEmulatorWindow.box
        self.mEmulatorWindow.activate()
        self.StatusEmit(f'Đã tìm thấy cửa sổ giả lập\n{self.mEmulatorBox}')
        mEmulatorSize = self.mConfig.mEmulatorSize
        if abs(mEmulatorSize[0] - self.mEmulatorBox.width) > 100 or abs(
                mEmulatorSize[1] - self.mEmulatorBox.height) > 100:
            self.MsgEmit(f'Cửa sổ giả lập {self.mEmulatorBox.width}x{self.mEmulatorBox.height} không phù hợp')
            return False
        if self.mEmulatorBox.top < 0:
            self.mEmulatorWindow.activate()
            self.mEmulatorWindow.move(0, abs(self.mEmulatorBox.top))
            # self.StatusEmit("Cửa sổ giả lập bị khuất về bên trên\nTự động di chuyển")
        if self.mEmulatorBox.left < 0:
            self.mEmulatorWindow.activate()
            self.mEmulatorWindow.move(abs(self.mEmulatorBox.left), 0)
            # self.StatusEmit("Cửa sổ giả lập bị khuất về bên trái\nTự động di chuyển")
        if self.mEmulatorBox.top + self.mEmulatorBox.height > self.mScreenSize.height:
            self.mEmulatorWindow.activate()
            self.mEmulatorWindow.move(0, self.mScreenSize.height - (self.mEmulatorBox.top + self.mEmulatorBox.height))
            # self.StatusEmit("Cửa sổ giả lập bị khuất về bên dưới\nTự động di chuyển")
        if self.mEmulatorBox.left + self.mEmulatorBox.width > self.mScreenSize.width:
            self.mEmulatorWindow.activate()
            self.mEmulatorWindow.move(self.mScreenSize.width - (self.mEmulatorBox.left + self.mEmulatorBox.width), 0)
            # self.StatusEmit("Cửa sổ giả lập bị khuất về bên phải\nTự động di chuyển")
        self.mEmulatorBox = self.mEmulatorWindow.box

        # Tọa độ tuyệt đối
        self.mAbsPullingRodPos = self.ConvertCoordinates(self.mConfig.mPullingRodPos)
        mAbsBackpackPos = self.ConvertCoordinates(self.mConfig.mOpenBackPack)
        self.mAbsBackPackRegion[0] = mAbsBackpackPos[0] - self.mConfig.mBackpackRec[0] // 2
        self.mAbsBackPackRegion[1] = mAbsBackpackPos[1] - self.mConfig.mBackpackRec[1] // 2
        self.mAbsBackPackRegion[2] = self.mConfig.mBackpackRec[0]
        self.mAbsBackPackRegion[3] = self.mConfig.mBackpackRec[1]

        mAbsPreservationPos = self.ConvertCoordinates(self.mConfig.mPreservationPos)
        self.mAbsPreservationRegion[2] = self.mConfig.mPreservationRec[0]
        self.mAbsPreservationRegion[3] = self.mConfig.mPreservationRec[1]
        self.mAbsPreservationRegion[0] = mAbsPreservationPos[0] - self.mConfig.mPreservationRec[0] // 2
        self.mAbsPreservationRegion[1] = mAbsPreservationPos[1] - self.mConfig.mPreservationRec[1] // 2

        return True

    def StartAdbServer(self):
        self.StatusEmit("Đang khởi tạo adb-server")
        try:
            subprocess.call(f'{self.mConfig.mAdbPath} devices', creationflags=0x08000000)
        except (ValueError, Exception):
            self.StatusEmit('Khởi tạo adb-server thất bại')
            return False
        self.StatusEmit('Khởi tạo adb-server thành công')
        return True

    def AdbServerConnect(self):
        self.mAdbDevices = []
        self.mListAdbDevicesSerial = []
        try:
            self.mAdbClient = AdbClient(self.mConfig.mAdbHost, self.mConfig.mAdbPort)
            self.mAdbDevices = self.mAdbClient.devices()
        except (ValueError, Exception):
            mCheckStartServer = self.StartAdbServer()
            if mCheckStartServer is False:
                self.MsgEmit('Không tìm thấy adb-server')
                return False
            else:
                self.mAdbClient = AdbClient(self.mConfig.mAdbHost, self.mConfig.mAdbPort)
                self.mAdbDevices = self.mAdbClient.devices()
        if len(self.mAdbDevices) == 0:
            self.mAdbClient = None
            self.MsgEmit('Không kết nối được giả lập qua adb-server\nRestart lại giả lập')
            return False
        else:
            for tempDevice in self.mAdbDevices:
                self.mListAdbDevicesSerial.append(tempDevice.serial)
        return True

    def AdbDeviceConnect(self):
        for index in range(len(self.mAdbDevices)):
            if self.mAdbDevices[index].serial == self.mConfig.mAdbAddress:
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
            self.MsgEmit("Chưa kết nối cửa sổ giả lập")
            return

        if self.mAdbDevice is None:
            self.MsgEmit("Chưa kết nối địa chỉ Adb của thiết bị")
            return

        if self.mMark[0] == 0:
            self.MsgEmit('Chưa xác định vị trí dấu chấm than')
            return False

        if self.mConfig.mFishDetectionCheck is True:
            if self.mFishingRegion[0] == 0:
                self.MsgEmit('Chưa xác định vùng câu')
                return False

        time.sleep(self.mConfig.mDelayTime)
        while self.mAutoFishRunning is True:
            print(752, self.mAutoFishRunning)
            time.sleep(self.mConfig.mDelayTime)
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
