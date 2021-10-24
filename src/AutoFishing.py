import math
import time
import pyautogui
import gc
import cv2
import threading

from ppadb.client import Client as AdbClient
import numpy
import win32api
from src.config import Config
from PyQt5.QtCore import pyqtSignal, QObject
import subprocess
import logging as log
from src.common import Flags


class AutoFishing(QObject):
    mSignalSetPixelPos = pyqtSignal(int, int)
    mSignalSetFishingBobberPos = pyqtSignal(int, int)
    mSignalUpdateFishingNum = pyqtSignal()
    mSignalUpdateFishNum = pyqtSignal()
    mSignalUpdateImageShow = pyqtSignal()
    mSignalMessage = pyqtSignal(str)
    mSignalUpdateStatus = pyqtSignal(str)

    def __init__(self):
        QObject.__init__(self, parent=None)

        self.mConfig = Config()
        self.mFishingNum = 0
        self.mAbsPullingRodPos = [0, 0]
        self.mAbsBackPackRegion = [0, 0, 0, 0]
        self.mAbsPreservationRegion = [0, 0, 0, 0]
        self.mListAbsCaptchaRegion = [[0, 0, 0, 0],
                                      [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                      [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                      [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.mAbsOKCaptchaRegion = [0, 0, 0, 0]
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
        self.mFixRodTime = 0
        self.mCaptchaHandleTime = 0
        self.mCaptchaRecognition = None

        # Khai báo số cá các loại
        self.mAllFish = 0
        self.mVioletFish = 0
        self.mBlueFish = 0
        self.mGreenFish = 0
        self.mGrayFish = 0

        self.mStartTime = time.time()
        self.mSaveTime = 0

        threading.Thread(name="InitClassification", target=self.InitClassification).start()

    def __del__(self):
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
        return mDiffTotal

    def InitClassification(self):
        log.info('InitClassification')
        from src.Classification import Classification
        self.mCaptchaRecognition = Classification()

    def FindImage(self, mImagePath: str, mRegion: list, mConfidence: float):
        if self.mConfig.mDebugMode is True:
            self.mImageShow = self.ScreenshotWindowRegion(mRegion)
            self.mSignalUpdateImageShow.emit()

        try:
            mLocate = pyautogui.locateOnScreen(mImagePath, grayscale=True,
                                               region=(mRegion[0],
                                                       mRegion[1],
                                                       mRegion[2],
                                                       mRegion[3]),
                                               confidence=mConfidence)
        except (ValueError, Exception):
            log.info(f'Error {mImagePath} {mRegion}')
            return False
        gc.collect()
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
        log.info(f'Clicked {self.mConfig.mCloseBackPack}')
        return True

    def OpenTools(self):
        self.AdbClick(self.mConfig.mTools[0],
                      self.mConfig.mTools[1])
        log.info(f'Clicked {self.mConfig.mTools}')
        return True

    def OpenBackPack(self):
        self.AdbClick(self.mConfig.mOpenBackPack[0],
                      self.mConfig.mOpenBackPack[1])
        log.info(f'Clicked {self.mConfig.mOpenBackPack}')

    def FixClick(self):
        self.AdbClick(self.mConfig.mListFishingRodPosition[self.mConfig.mFishingRodIndex][0],
                      self.mConfig.mListFishingRodPosition[self.mConfig.mFishingRodIndex][1])
        log.info(f'Clicked {self.mConfig.mListFishingRodPosition[self.mConfig.mFishingRodIndex]}')

    def FixConfirm(self):
        self.AdbClick(self.mConfig.mConfirm[0],
                      self.mConfig.mConfirm[1])
        log.info(f'Clicked {self.mConfig.mConfirm}')

    def ClickOk(self):
        self.AdbClick(self.mConfig.mOKButton[0],
                      self.mConfig.mOKButton[1])
        log.info(f'Clicked {self.mConfig.mOKButton}')

    def CheckRod(self):
        time.sleep(self.mConfig.mDelayTime)
        for i in range(6):
            # break point thread auto fishing
            if self.mAutoFishRunning is False:
                return Flags.STOP_FISHING
            time.sleep(0.5)

        mCheck = 0
        while mCheck < 5:
            if self.mAutoFishRunning is False:
                return Flags.STOP_FISHING
            mBackpackPos = self.FindImage(self.mConfig.mBackpackImgPath,
                                          self.mAbsBackPackRegion,
                                          self.mConfig.mConfidence)
            if mBackpackPos is not None and mBackpackPos is not False:
                if self.CheckCaptcha() == Flags.CAPTCHA_APPEAR:
                    log.info('CheckRod Captcha Appear')
                    return Flags.CAPTCHA_APPEAR
                log.info('CheckRod Broken')
                return Flags.CHECK_ROD_BROK
            time.sleep(0.2)
            mCheck += 1
        log.info('CheckRod OK')
        return Flags.CHECK_ROD_OK

    def FixRod(self):
        log.info(f'Fix Rod Start')
        self.StatusEmit("Cần câu bị hỏng. Auto đang sửa cần\n"
                        "Nếu không đúng hỏng cần, hãy cài đặt độ trễ từ 1 giây trở lên")
        self.OpenBackPack()
        if self.mAutoFishRunning is False:
            return Flags.STOP_FISHING
        time.sleep(self.mConfig.mDelayTime + 0.5)
        if self.mAutoFishRunning is False:
            return Flags.STOP_FISHING
        self.OpenTools()
        if self.mAutoFishRunning is False:
            return Flags.STOP_FISHING
        time.sleep(self.mConfig.mDelayTime)
        if self.mAutoFishRunning is False:
            return Flags.STOP_FISHING
        self.FixClick()
        if self.mAutoFishRunning is False:
            return Flags.STOP_FISHING
        time.sleep(self.mConfig.mDelayTime + 0.5)
        if self.mAutoFishRunning is False:
            return Flags.STOP_FISHING
        self.FixConfirm()
        if self.mAutoFishRunning is False:
            return Flags.STOP_FISHING
        time.sleep(self.mConfig.mDelayTime + 0.5)
        if self.mAutoFishRunning is False:
            return Flags.STOP_FISHING
        self.ClickOk()
        if self.mAutoFishRunning is False:
            return Flags.STOP_FISHING
        time.sleep(self.mConfig.mDelayTime)
        if self.mAutoFishRunning is False:
            return Flags.STOP_FISHING
        self.CloseBackPack()
        if self.mAutoFishRunning is False:
            return Flags.STOP_FISHING
        time.sleep(self.mConfig.mDelayTime)
        mCheck = 0
        while mCheck < 5:
            # check point break auto fishing thread
            if self.mAutoFishRunning is False:
                return
            mPreservationButtonPos = self.FindImage(self.mConfig.mPreservationImgPath,
                                                    self.mAbsPreservationRegion,
                                                    self.mConfig.mConfidence)
            if mPreservationButtonPos is not None and mPreservationButtonPos is not False:
                self.AdbClick(self.mConfig.mPreservationPos[0],
                              self.mConfig.mPreservationPos[1])
                log.info(f'Click preservation button {self.mConfig.mPreservationPos}')
                return
            mCheck += 1
            time.sleep(0.2)
        return

    def CastFishingRod(self):
        mCheck = 0
        while mCheck < 5:
            # break point thread
            if self.mAutoFishRunning is False:
                return Flags.STOP_FISHING
            mBackpackPos = self.FindImage(self.mConfig.mBackpackImgPath,
                                          self.mAbsBackPackRegion,
                                          self.mConfig.mConfidence)
            if mBackpackPos is not None and mBackpackPos is not False:
                self.AdbClick(self.mConfig.mCastingRodPos[0],
                              self.mConfig.mCastingRodPos[1])
                self.StatusEmit("Đã thả cần câu")
                log.info(f'Clicked {self.mConfig.mCastingRodPos}')
                return True
            mCheck += 1
            time.sleep(0.2)
        self.StatusEmit("Không tìm thấy nút ba lô. Đóng ba lô")
        log.info(f'Cannot find backpack')
        self.CloseBackPack()

        mCheck = 0
        while mCheck < 5:
            # break point thread
            if self.mAutoFishRunning is False:
                return Flags.STOP_FISHING
            mPreservationButtonPos = self.FindImage(self.mConfig.mPreservationImgPath,
                                                    self.mAbsPreservationRegion,
                                                    self.mConfig.mConfidence)
            if mPreservationButtonPos is not None and mPreservationButtonPos is not False:
                self.AdbClick(self.mConfig.mPreservationPos[0],
                              self.mConfig.mPreservationPos[1])
                log.info(f'Click preservation button {self.mConfig.mPreservationPos}')
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
        # trời mưa 70-89 ở hồ home
        elif 70 < mBackGroundColor < 100:
            mMinThreshValue = 25
            mMaxThreshValue = 100
        # buổi chiều nền biền 74, sáng ở camp 149, chiều ở cam 166
        elif 100 < mBackGroundColor < 170:
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
                                   self.mConfig.mTextColor, self.mConfig.mThickness, cv2.LINE_AA)
        mCurrFrameRGB = cv2.circle(mCurrFrameRGB, (mImgCenterX, mImgCenterY),
                                   self.mConfig.mRadiusFishingRegion * 1 // 4,
                                   self.mConfig.mTextColor, self.mConfig.mThickness, cv2.LINE_AA)

        cv2.putText(mCurrFrameRGB, str(mBackGroundColor),
                    (int(10 * self.mConfig.mFontScale),
                     int(40 * self.mConfig.mFontScale)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.mConfig.mFontScale,
                    self.mConfig.mTextColor, self.mConfig.mThickness, cv2.LINE_AA)
        for mContour in mContours:
            # break point thread
            if self.mAutoFishRunning is False:
                return Flags.STOP_FISHING

            # toa do cua cac contour tim duoc
            (x, y, w, h) = cv2.boundingRect(mContour)

            # Neu contour co height > 2.0 weight thi co the la cham than hoac troi mua
            if h > 2 * w:
                continue

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
            if cv2.contourArea(mContour) < self.mConfig.mMinContour:
                continue
            # loai box qua to
            if cv2.contourArea(mContour) > self.mConfig.mMaxContour:
                continue
            mFishArea = int(cv2.contourArea(mContour))
            cv2.rectangle(mCurrFrameRGB, (x, y), (x + w, y + h), self.mConfig.mTextColor, self.mConfig.mThickness,
                          cv2.LINE_AA)
            cv2.putText(mCurrFrameRGB, str(mFishArea), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, self.mConfig.mFontScale, self.mConfig.mTextColor,
                        self.mConfig.mThickness, cv2.LINE_AA)
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
        mStaticFrameGray = None
        if self.mConfig.mFishDetectionCheck is True:
            mStaticFrameGray, mStaticFrameRGB = self.ScreenshotFishingRegion()
            if mStaticFrameRGB is False:
                return
            self.mImageShow = mStaticFrameRGB

        for i in range(self.mConfig.mWaitingFishTime * 2):
            time.sleep(0.5)
            # break point thread
            if self.mAutoFishRunning is False:
                return Flags.STOP_FISHING

        self.StatusEmit("Đang đợi cá")
        log.info(f'Waiting Mark')
        mPixelBase = None
        # mPixelBaseTop = None
        # mPixelBaseLeft = None
        # mPixelBaseRight = None
        mPixelBaseDay = None

        for i in range(10):
            try:
                mPixelBase = pyautogui.pixel(self.mMark[0], self.mMark[1])
                # mPixelBaseTop = pyautogui.pixel(self.mMark[0],
                #                                 self.mMark[1] - self.mConfig.mMarkPixelDist)
                # mPixelBaseLeft = pyautogui.pixel(self.mMark[0] - self.mConfig.mMarkPixelDist,
                #                                  self.mMark[1])
                # mPixelBaseRight = pyautogui.pixel(self.mMark[0] + self.mConfig.mMarkPixelDist,
                #                                   self.mMark[1])
                mPixelBaseDay = pyautogui.pixel(self.mEmulatorBox.left + 20,
                                                self.mEmulatorBox.top + self.mEmulatorBox.height // 5)
            except (ValueError, Exception):
                time.sleep(0.001)
                continue
            break

        time1 = time.time()
        time2 = time.time()
        mStopDetect = False
        mSkipFrame = 0
        while (time2 - time1) < self.mConfig.mWaitingMarkTime:
            # t = time.time()
            # break point thread
            if self.mAutoFishRunning is False:
                return Flags.STOP_FISHING

            if self.mConfig.mFishDetectionCheck is True and mStopDetect is False:
                mCurrentFrameGray, mCurrentFrameRGB = self.ScreenshotFishingRegion()
                if mCurrentFrameGray is False:
                    return
                mSizeFish = self.FishDetection(mStaticFrameGray, mCurrentFrameGray, mCurrentFrameRGB)
                if mSizeFish != 0:
                    mSkipFrame += 1
                if mSkipFrame == 5:
                    mStopDetect = True
                    log.info(f'Size Fish = {mSizeFish}')
                    if mSizeFish < self.mConfig.mFishSize:
                        return
            try:
                mPixelCurr = pyautogui.pixel(self.mMark[0], self.mMark[1])
                # mPixelCurrTop = pyautogui.pixel(self.mMark[0],
                #                                 self.mMark[1] - self.mConfig.mMarkPixelDist)
                # mPixelCurrLeft = pyautogui.pixel(self.mMark[0] - self.mConfig.mMarkPixelDist,
                #                                  self.mMark[1])
                # mPixelCurrRight = pyautogui.pixel(self.mMark[0] + self.mConfig.mMarkPixelDist,
                #                                   self.mMark[1])
                mPixelCurrDay = pyautogui.pixel(self.mEmulatorBox.left + 20,
                                                self.mEmulatorBox.top + self.mEmulatorBox.height // 5)
                mDiffRgb = self.ComparePixel(mPixelCurr, mPixelBase)
                # mDiffRgbTop = self.ComparePixel(mPixelCurrTop, mPixelBaseTop)
                # mDiffRgbLeft = self.ComparePixel(mPixelCurrLeft, mPixelBaseLeft)
                # mDiffRgbRight = self.ComparePixel(mPixelCurrRight, mPixelBaseRight)
                mDiffRgbDay = self.ComparePixel(mPixelCurrDay, mPixelBaseDay)
            except (ValueError, Exception):
                time.sleep(0.001)
                time2 = time.time()
                continue

            # print('*********************************')
            # print(f'__________mDiffRgb = {mDiffRgb}')
            # print(f'++++mDiffRgbTop = {mDiffRgbTop}')
            # print(f'++++mDiffRgbLeft = {mDiffRgbLeft}')
            # print(f'++++mDiffRgbRight = {mDiffRgbRight}')
            # temp = self.ScreenshotWindowRegion([self.mEmulatorBox.left, self.mEmulatorBox.top,
            #                                     40, self.mEmulatorBox.height * 2 // 5])
            # cv2.imshow("temp", mPixelCurrDay)
            # cv2.waitKey(1)
            # print(mPixelCurrDay)

            if mDiffRgb > 30 and mDiffRgbDay > 30:
                mPixelBase = mPixelCurr
                # mPixelBaseTop = mPixelCurrTop
                # mPixelBaseLeft = mPixelCurrLeft
                # mPixelBaseRight = mPixelCurrRight
                mPixelBaseDay = mPixelCurrDay
                time.sleep(0.001)
                time2 = time.time()
                continue

            # if mDiffRgbLeft > 30:
            #     time.sleep(0.001)
            #     time2 = time.time()
            #     continue

            if mDiffRgb > 50:
                log.info(f'mDiffRgb = {mDiffRgb}')
                return

            time2 = time.time()
            time.sleep(0.001)
            # print(time.time() - t)
        self.StatusEmit("Không phát hiện dấu chấm than")
        log.info(f'Cannot find mark')
        return

    def PullFishingRod(self):
        if self.mConfig.mFreeMouseCheck is False:
            try:
                pyautogui.click(x=self.mAbsPullingRodPos[0],
                                y=self.mAbsPullingRodPos[1],
                                clicks=2,
                                interval=0.1,
                                button='left')
            except (ValueError, Exception):
                return
            self.StatusEmit("Đang kéo cần câu")
            log.info(f'Clicked {self.mAbsPullingRodPos}')
            return
        else:
            time1 = time.time()
            self.AdbClick(self.mConfig.mPullingRodPos[0],
                          self.mConfig.mPullingRodPos[1])
            timeDelay = time.time() - time1
            self.StatusEmit(f'Đang kéo cần câu. Độ trễ giật cần {round(timeDelay, 2)} giây')
            log.info(f'Clicked {self.mAbsPullingRodPos}. Delay = {round(timeDelay, 2)} sec')
            if timeDelay > 0.5:
                self.mCheckAdbDelay += 1
                if self.mCheckAdbDelay <= 3:
                    return
                self.mAutoFishRunning = False
                self.MsgEmit(
                    'Độ trễ Adb Server cao trên 0.5 giây\nTắt chế độ "Không chiếm chuột" để không bị kéo hụt cá')
                self.mCheckAdbDelay = 0
            return

    def FishPreservation(self):
        time.sleep(0.1)
        if self.mAutoFishRunning is False:
            return Flags.STOP_FISHING
        time1 = time.time()
        time2 = time.time()
        while (time2 - time1) < 15:
            # check point break auto fishing thread
            if self.mAutoFishRunning is False:
                return Flags.STOP_FISHING
            mBackpackPos = self.FindImage(self.mConfig.mBackpackImgPath,
                                          self.mAbsBackPackRegion,
                                          self.mConfig.mConfidence)
            if mBackpackPos is False:
                time.sleep(0.1)
                continue

            if mBackpackPos is not None:
                self.StatusEmit("Câu thất bại")
                log.info(f'Fishing fail')
                return

            time.sleep(0.1)
            mPreservationPos = self.FindImage(self.mConfig.mPreservationImgPath,
                                              self.mAbsPreservationRegion,
                                              self.mConfig.mConfidence)
            if mPreservationPos is not None and mPreservationPos is not False:
                self.StatusEmit("Câu thành công")
                self.FishCount()
                self.AdbClick(self.mConfig.mPreservationPos[0],
                              self.mConfig.mPreservationPos[1])
                log.info(f'Fishing Success. Click preservation {self.mConfig.mPreservationPos}')
                return
            time.sleep(0.1)
            time2 = time.time()
        self.StatusEmit("Kiểm tra kết quả bị lỗi")
        log.info(f'Check fishing result error')
        return

    def FishCount(self):
        time.sleep(0.5)
        mFishImagePos = self.ConvertCoordinates([self.mConfig.mFishImgRegion[0], self.mConfig.mFishImgRegion[1]])
        mFishImageRegion = [mFishImagePos[0], mFishImagePos[1], self.mConfig.mFishImgRegion[2],
                            self.mConfig.mFishImgRegion[3]]
        mFishImage = self.ScreenshotWindowRegion(mFishImageRegion)
        if mFishImage is False:
            log.info(f'Screen shot fish error, region = {mFishImageRegion}')
            return False
        self.mAllFish += 1
        mPixelCheckTypeFishPosition = [self.mConfig.mCheckTypeFishPos[0] - self.mConfig.mFishImgRegion[0],
                                       self.mConfig.mCheckTypeFishPos[1] - self.mConfig.mFishImgRegion[1]]
        mPixelCheckTypeFish = mFishImage[mPixelCheckTypeFishPosition[1],
                                         mPixelCheckTypeFishPosition[0]]
        if self.ComparePixel(mPixelCheckTypeFish, self.mConfig.mVioletColorBGR) < 10:
            self.mVioletFish += 1
            log.info(f'VioletFish = {self.mVioletFish}')
        elif self.ComparePixel(mPixelCheckTypeFish, self.mConfig.mBlueColorBGR) < 10:
            self.mBlueFish += 1
            log.info(f'BlueFish = {self.mBlueFish}')
        elif self.ComparePixel(mPixelCheckTypeFish, self.mConfig.mGreenColorBGR) < 10:
            self.mGreenFish += 1
            log.info(f'GreenFish = {self.mGreenFish}')
        elif self.ComparePixel(mPixelCheckTypeFish, self.mConfig.mGrayColorBGR) < 10:
            self.mGrayFish += 1
            log.info(f'GrayFish = {self.mGrayFish}')
        else:
            log.info(f'No fish')
            pass
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
        self.StatusEmit(
            "Đưa chuột đến ĐỈNH của CHẤM THAN và bấm chuột\nChấm đỏ NGOÀI VÒNG TRÒN phải nằm NGOÀI CHẤM THAN")
        self.mCheckMouseRunning = True
        while self.mCheckMouseRunning is True:
            mMousePos = pyautogui.position()
            self.mSignalSetPixelPos.emit(mMousePos.x, mMousePos.y)
            if self.mScreenSize[0] - self.mConfig.mMarkPixelRadius > mMousePos.x > self.mConfig.mMarkPixelRadius and \
                    self.mScreenSize[1] - self.mConfig.mMarkPixelRadius > mMousePos.y > self.mConfig.mMarkPixelRadius:
                mRegion = [mMousePos.x - self.mConfig.mMarkPixelRadius, mMousePos.y - self.mConfig.mMarkPixelRadius,
                           2 * self.mConfig.mMarkPixelRadius, 2 * self.mConfig.mMarkPixelRadius]
                mTempImage = self.ScreenshotWindowRegion(mRegion)
                mTempImage = cv2.circle(mTempImage, (self.mConfig.mMarkPixelRadius, self.mConfig.mMarkPixelRadius),
                                        1, (0, 0, 255), 1, cv2.LINE_AA)
                # mTempImage = cv2.circle(mTempImage,
                #                         (self.mConfig.mMarkPixelRadius,
                #                          self.mConfig.mMarkPixelRadius - self.mConfig.mMarkPixelDist),
                #                         1, (0, 0, 255), 1, cv2.LINE_AA)
                # mTempImage = cv2.circle(mTempImage,
                #                         (self.mConfig.mMarkPixelRadius - self.mConfig.mMarkPixelDist,
                #                          self.mConfig.mMarkPixelRadius + self.mConfig.mMarkPixelDist // 2),
                #                         1, (0, 0, 255), 1, cv2.LINE_AA)
                # mTempImage = cv2.circle(mTempImage,
                #                         (self.mConfig.mMarkPixelRadius + self.mConfig.mMarkPixelDist,
                #                          self.mConfig.mMarkPixelRadius + self.mConfig.mMarkPixelDist // 2),
                #                         1, (0, 0, 255), 1, cv2.LINE_AA)
                mTempImage = cv2.rectangle(mTempImage,
                                           (self.mConfig.mMarkPixelRadius - self.mConfig.mMarkPixelDist // 2,
                                            self.mConfig.mMarkPixelRadius - self.mConfig.mMarkPixelDist // 2),
                                           (self.mConfig.mMarkPixelRadius + self.mConfig.mMarkPixelDist // 2,
                                            self.mConfig.mMarkPixelRadius + self.mConfig.mMarkPixelDist * 2),
                                           (0, 0, 255), 1, cv2.LINE_AA)
                self.mImageShow = mTempImage.copy()
                self.mSignalUpdateImageShow.emit()
            if self.CheckLeftMouseClick() is True:
                self.mCheckMouseRunning = False
            time.sleep(0.01)
        self.mMark[0] = mMousePos.x
        self.mMark[1] = mMousePos.y
        self.StatusEmit(f'Vị trí dấu chấm than đã cài đặt:\n{mMousePos}')
        log.info(f'Mark position = {mMousePos}')

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
        log.info(f'Bobber position = {mMousePos}')

    def CheckRegionEmulator(self):
        self.mScreenSize = pyautogui.size()
        self.mEmulatorBox = None
        self.mEmulatorWindow = None
        self.StatusEmit(f'Kích thước màn hình =\n{self.mScreenSize}')
        log.info(f'Screen size = {self.mScreenSize}')
        try:
            mEmulatorWindows = pyautogui.getWindowsWithTitle(self.mConfig.mWindowName)
        except (ValueError, Exception):
            self.MsgEmit(f'Không tìm thấy cửa sổ {self.mConfig.mWindowName}')
            log.info(f'Cannot find window name {self.mConfig.mWindowName}')
            return False
        if len(mEmulatorWindows) > 0:
            self.mEmulatorWindow = mEmulatorWindows[0]
        else:
            self.MsgEmit(f'Không tìm thấy cửa sổ {self.mConfig.mWindowName}')
            log.info(f'Cannot find window name {self.mConfig.mWindowName}')
            return False
        self.mEmulatorBox = self.mEmulatorWindow.box
        self.mEmulatorWindow.activate()
        self.StatusEmit(f'Đã tìm thấy cửa sổ giả lập\n{self.mEmulatorBox}')
        log.info(f'Found {self.mConfig.mWindowName}, size = {self.mEmulatorBox}')
        mEmulatorSize = self.mConfig.mEmulatorSize
        if abs(mEmulatorSize[0] - self.mEmulatorBox.width) > 100 or abs(
                mEmulatorSize[1] - self.mEmulatorBox.height) > 100:
            self.MsgEmit(f'Cửa sổ giả lập {self.mEmulatorBox.width}x{self.mEmulatorBox.height} không phù hợp')
            log.info(f'Emulator size {self.mEmulatorBox.width}x{self.mEmulatorBox.height} not suitable')
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
        log.info(f'Emulator box {self.mEmulatorBox}')

        # Tọa độ tuyệt đối
        self.mAbsPullingRodPos = self.ConvertCoordinates(self.mConfig.mPullingRodPos)
        mAbsBackpackPos = self.ConvertCoordinates(self.mConfig.mOpenBackPack)
        self.mAbsBackPackRegion[0] = mAbsBackpackPos[0] - self.mConfig.mBackpackRec[0] // 2
        self.mAbsBackPackRegion[1] = mAbsBackpackPos[1] - self.mConfig.mBackpackRec[1] // 2
        self.mAbsBackPackRegion[2] = self.mConfig.mBackpackRec[0]
        self.mAbsBackPackRegion[3] = self.mConfig.mBackpackRec[1]
        log.info(f'Abs Pulling Rod Position = {self.mAbsBackPackRegion}')

        mAbsPreservationPos = self.ConvertCoordinates(self.mConfig.mPreservationPos)
        self.mAbsPreservationRegion[2] = self.mConfig.mPreservationRec[0]
        self.mAbsPreservationRegion[3] = self.mConfig.mPreservationRec[1]
        self.mAbsPreservationRegion[0] = mAbsPreservationPos[0] - self.mConfig.mPreservationRec[0] // 2
        self.mAbsPreservationRegion[1] = mAbsPreservationPos[1] - self.mConfig.mPreservationRec[1] // 2
        log.info(f'Abs Preservation Region = {self.mAbsPreservationRegion}')

        for i in range(len(self.mConfig.mListCaptchaRegion)):
            mCaptchaCornerPos = [self.mConfig.mListCaptchaRegion[i][0],
                                 self.mConfig.mListCaptchaRegion[i][1]]
            mAbsCaptchaCornerPos = self.ConvertCoordinates(mCaptchaCornerPos)
            self.mListAbsCaptchaRegion[i][0] = mAbsCaptchaCornerPos[0]
            self.mListAbsCaptchaRegion[i][1] = mAbsCaptchaCornerPos[1]
            self.mListAbsCaptchaRegion[i][2] = self.mConfig.mListCaptchaRegion[i][2]
            self.mListAbsCaptchaRegion[i][3] = self.mConfig.mListCaptchaRegion[i][3]

        log.info(f'List Abs Captcha Region = {self.mListAbsCaptchaRegion}')

        mAbsOKCaptchaPos = self.ConvertCoordinates(self.mConfig.mOKCaptchaPos)
        self.mAbsOKCaptchaRegion[2] = self.mConfig.mOKCaptchaRec[0]
        self.mAbsOKCaptchaRegion[3] = self.mConfig.mOKCaptchaRec[1]
        self.mAbsOKCaptchaRegion[0] = mAbsOKCaptchaPos[0] - self.mConfig.mOKCaptchaRec[0] // 2
        self.mAbsOKCaptchaRegion[1] = mAbsOKCaptchaPos[1] - self.mConfig.mOKCaptchaRec[1] // 2
        log.info(f'Abs OK Captcha Region = {self.mAbsOKCaptchaRegion}')

        return True

    def StartAdbServer(self):
        self.StatusEmit("Đang khởi tạo adb-server")
        log.info(f'')
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
            log.info(f'Complete')
            for tempDevice in self.mAdbDevices:
                self.mListAdbDevicesSerial.append(tempDevice.serial)
        return True

    def AdbDeviceConnect(self):
        for index in range(len(self.mAdbDevices)):
            if self.mAdbDevices[index].serial == self.mConfig.mAdbAddress:
                self.mAdbDevice = self.mAdbDevices[index]
                break
        if self.mAdbDevice is None:
            log.info(f'Device not found')
            return False
        log.info(f'Connected {self.mAdbDevice.serial}')
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
        if self.mCaptchaRecognition is None:
            self.InitClassification()

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
        time.sleep(0.5)
        while self.mAutoFishRunning is True:
            gc.collect()
            time.sleep(self.mConfig.mDelayTime)
            log.info('********************************************************')
            log.info(f'Fishing time {self.mFishingNum + 1}')

            # break point thread auto fishing
            if self.mAutoFishRunning is False:
                return Flags.STOP_FISHING

            # Have captcha?
            mCheckCaptcha = self.CheckCaptcha()
            if mCheckCaptcha == Flags.CAPTCHA_APPEAR:
                self.CaptchaHandle()
                # break point thread auto fishing
                if self.mAutoFishRunning is False:
                    return Flags.STOP_FISHING
                self.mCaptchaHandleTime += 1
                if self.mCaptchaHandleTime == 8:
                    self.MsgEmit('Giải captcha sai 8 lần')
                    self.mAutoFishRunning = False
                    break
                else:
                    continue
            elif mCheckCaptcha == Flags.CAPTCHA_NONE:
                self.mCaptchaHandleTime = 0
            else:
                return Flags.STOP_FISHING

            mCheckCastRod = self.CastFishingRod()
            # break point thread auto fishing
            if self.mAutoFishRunning is False:
                return Flags.STOP_FISHING
            if mCheckCastRod is False:
                continue

            self.mFishingNum += 1
            self.mSignalUpdateFishingNum.emit()

            # break point thread auto fishing
            if self.mAutoFishRunning is False:
                return Flags.STOP_FISHING

            mCheckRod = self.CheckRod()
            # break point thread auto fishing
            if self.mAutoFishRunning is False:
                return Flags.STOP_FISHING
            if mCheckRod == Flags.CHECK_ROD_OK:
                self.mFixRodTime = 0
                self.CheckMark()
                # break point thread auto fishing
                if self.mAutoFishRunning is False:
                    return Flags.STOP_FISHING
                self.PullFishingRod()
                # break point thread auto fishing
                if self.mAutoFishRunning is False:
                    return Flags.STOP_FISHING
                self.FishPreservation()
                # break point thread auto fishing
                if self.mAutoFishRunning is False:
                    return Flags.STOP_FISHING
                self.mSignalUpdateFishNum.emit()
            elif mCheckRod == Flags.CHECK_ROD_BROK:
                self.FixRod()
                # break point thread auto fishing
                if self.mAutoFishRunning is False:
                    return Flags.STOP_FISHING
                self.mFixRodTime += 1
                if self.mFixRodTime == 5:
                    self.MsgEmit("Hết lượt câu")
                    while True:
                        if self.mAutoFishRunning is False:
                            return Flags.STOP_FISHING
                        time.sleep(0.5)
            elif mCheckRod == Flags.CAPTCHA_APPEAR:
                self.CaptchaHandle()
                # break point thread auto fishing
                if self.mAutoFishRunning is False:
                    return Flags.STOP_FISHING

                self.mCaptchaHandleTime += 1
                if self.mCaptchaHandleTime == 8:
                    self.MsgEmit('Giải captcha sai 8 lần')
                    self.mAutoFishRunning = False
                    break
            else:
                pass
        return False

    def CheckCaptcha(self):
        log.info('Check Captcha')
        mCheck = 0
        while mCheck < 3:
            # break point thread auto fishing
            if self.mAutoFishRunning is False:
                return Flags.STOP_FISHING
            mOKCaptchaPos = self.FindImage(self.mConfig.mOKCaptchaImgPath,
                                           self.mAbsOKCaptchaRegion,
                                           self.mConfig.mConfidence)
            if mOKCaptchaPos is not None and mOKCaptchaPos is not False:
                return Flags.CAPTCHA_APPEAR
            time.sleep(0.1)
            mCheck += 1
        return Flags.CAPTCHA_NONE

    def CaptchaHandle(self):
        if self.mAutoFishRunning is False:
            return Flags.STOP_FISHING

        log.info('Captcha Handle Start')
        self.StatusEmit("Phát hiện Captcha. Đang xử lý ...")
        mBigCaptchaImage = self.ScreenshotWindowRegion(self.mListAbsCaptchaRegion[0])
        mBigCaptchaLabel, mBigCaptchaConfident = self.mCaptchaRecognition.Run(mBigCaptchaImage)
        log.info(f'Big captcha info = {mBigCaptchaLabel}, {mBigCaptchaConfident} %')

        mShowCaptcha = mBigCaptchaImage.copy()
        mShowCaptcha = cv2.resize(mShowCaptcha, (200, 200), interpolation=cv2.INTER_AREA)
        cv2.putText(mShowCaptcha, mBigCaptchaLabel, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 1, cv2.LINE_AA)
        self.mImageShow = mShowCaptcha
        self.mSignalUpdateImageShow.emit()

        if mBigCaptchaConfident < 90:
            idTime = int(time.time())
            fileName = f'{mBigCaptchaLabel}_{mBigCaptchaConfident}_{idTime}.jpg'
            cv2.imwrite(f'log/new_captcha/{fileName}', mBigCaptchaImage)

        if self.mConfig.mDebugMode is True:
            idTime = time.time()
            fileName = f'{mBigCaptchaLabel}_{mBigCaptchaConfident}_{idTime}.jpg'
            cv2.imwrite(f'log/log_captcha/{fileName}', mBigCaptchaImage)
        time.sleep(0.1)

        numMatchCaptcha = 0
        for i in range(1, 10):
            if self.mAutoFishRunning is False:
                return Flags.STOP_FISHING

            idTime = time.time()
            mSmallCaptchaImage = self.ScreenshotWindowRegion(self.mListAbsCaptchaRegion[i])
            mSmallCaptchaLabel, mSmallCaptchaConfident = self.mCaptchaRecognition.Run(mSmallCaptchaImage)
            log.info(f'Small captcha info {i} = {mSmallCaptchaLabel}, {mSmallCaptchaConfident} %')

            mShowCaptcha = mSmallCaptchaImage.copy()
            mShowCaptcha = cv2.resize(mShowCaptcha, (200, 200), interpolation=cv2.INTER_AREA)
            cv2.putText(mShowCaptcha, mSmallCaptchaLabel, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 1, cv2.LINE_AA)

            self.mImageShow = mShowCaptcha
            self.mSignalUpdateImageShow.emit()

            if mSmallCaptchaConfident > 90 and mSmallCaptchaLabel == mBigCaptchaLabel:
                numMatchCaptcha += 1
                self.AdbClick((self.mConfig.mListCaptchaRegion[i][0] + self.mConfig.mListCaptchaRegion[i][2] // 2),
                              (self.mConfig.mListCaptchaRegion[i][1] + self.mConfig.mListCaptchaRegion[i][3] // 2))

            if mSmallCaptchaConfident < 90:
                fileName = f'{mSmallCaptchaLabel}_{mSmallCaptchaConfident}_{idTime}.jpg'
                cv2.imwrite(f'log/new_captcha/{fileName}', mSmallCaptchaImage)

            if self.mConfig.mDebugMode is True:
                fileName = f'{mSmallCaptchaLabel}_{mSmallCaptchaConfident}_{idTime}.jpg'
                cv2.imwrite(f'log/log_captcha/{fileName}', mSmallCaptchaImage)
            time.sleep(0.1)

        self.AdbClick(self.mConfig.mOKCaptchaPos[0], self.mConfig.mOKCaptchaPos[1])
        time.sleep(1)

        if self.mAutoFishRunning is False:
            return Flags.STOP_FISHING

        if numMatchCaptcha == 0:
            log.info("Click refresh")
            self.AdbClick(self.mConfig.mRefreshCaptcha[0], self.mConfig.mRefreshCaptcha[1])
            time.sleep(3)
            return

        if self.mAutoFishRunning is False:
            return Flags.STOP_FISHING

        self.AdbClick(self.mConfig.mOKCaptchaComplete[0], self.mConfig.mOKCaptchaComplete[1])
        time.sleep(1)
        log.info('Captcha Handle Complete')
        return
