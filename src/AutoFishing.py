import math
import time
import gc
import cv2
import threading
import pyautogui

from ppadb.client import Client as AdbClient
import win32api
from src.config import Config
from PyQt5.QtCore import pyqtSignal, QObject
import subprocess
import logging as log
import win32gui

from src.common import *
from src.ScreenHandle import ScreenHandle
from src.ReadMemory import ReadMemory


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
        self.mScreenHandle = ScreenHandle()
        self.mReadMemory = ReadMemory()

        self.mFishingNum = 0
        self.mMark = [0, 0]
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
        self.mCheckBobberPos = False
        self.mCheckMarkPos = False
        self.mEmulatorType = ''

        # Khai báo số cá các loại
        self.mAllFish = 0
        self.mVioletFish = 0
        self.mBlueFish = 0
        self.mGreenFish = 0
        self.mGrayFish = 0

        self.mStartTime = time.time()
        self.mSaveTime = 0

        threading.Thread(name="InitClassification", target=self.InitClassification).start()

        self.mScreenHandle.mSignalFindImage.connect(self.EmitUpdateImageShow)

    def __del__(self):
        self.mCheckMouseRunning = False
        self.mAutoFishRunning = False

    def EmitUpdateImageShow(self):
        self.mImageShow = self.mScreenHandle.mImageShow.copy()
        self.mSignalUpdateImageShow.emit()

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
    def ComparePixel(mPixel1: list, mPixel2: list):
        mDiffTotal = 0
        for i in range(3):
            mDiffTotal += abs(int(mPixel1[i]) - int(mPixel2[i]))
        return mDiffTotal

    def InitClassification(self):
        log.info('InitClassification')
        from src.Classification import Classification
        self.mCaptchaRecognition = Classification()

    # Convert tọa độ tuyet doi tren man hinh sang toa do tuong doi
    def ConvertCoordinates(self, absPos: list):
        relativePos = [0, 0]
        relativePos[0] = absPos[0] - self.mEmulatorBox[0]
        relativePos[1] = absPos[1] + self.mConfig.mEmulatorSize[1] - self.mEmulatorBox[1] - self.mEmulatorBox[3]
        return relativePos

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

            mCheckBackpack = self.mScreenHandle.FindImage(self.mConfig.mBackpackImg,
                                                          self.mConfig.mBackpackRegion,
                                                          self.mConfig.mConfidence)
            if mCheckBackpack == Flags.TRUE:
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
        self.StatusEmit("Hỏng cần câu. Đang sửa ...\n"
                        "Nếu gặp lỗi, hãy cài đặt độ trễ cao hơn")
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
        return

    def CastFishingRod(self):
        mCheck = 0
        while mCheck < 5:
            # break point thread
            if self.mAutoFishRunning is False:
                return Flags.STOP_FISHING
            mCheckBackpack = self.mScreenHandle.FindImage(self.mConfig.mBackpackImg,
                                                          self.mConfig.mBackpackRegion,
                                                          self.mConfig.mConfidence)
            if mCheckBackpack == Flags.TRUE:
                self.AdbClick(self.mConfig.mCastingRodPos[0],
                              self.mConfig.mCastingRodPos[1])
                self.StatusEmit("Đã thả cần câu")
                log.info(f'Clicked {self.mConfig.mCastingRodPos}')
                return True
            mCheck += 1
            time.sleep(0.1)
        self.StatusEmit("Không tìm thấy nút ba lô")
        log.info(f'Cannot find backpack')
        self.CloseBackPack()

        mCheck = 0
        while mCheck < 5:
            # break point thread
            if self.mAutoFishRunning is False:
                return Flags.STOP_FISHING
            mCheckPreservation = self.mScreenHandle.FindImage(self.mConfig.mPreservationImg,
                                                              self.mConfig.mFishingResultRegion,
                                                              self.mConfig.mConfidence)
            if mCheckPreservation == Flags.TRUE:
                self.AdbClick(self.mConfig.mPreservationPos[0],
                              self.mConfig.mPreservationPos[1])
                log.info(f'Click preservation button {self.mConfig.mPreservationPos}')
                return False
            mCheck += 1
            time.sleep(0.1)

        self.FixConfirm()
        time.sleep(self.mConfig.mDelayTime)
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
            mMinThreshValue = 20
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
                                   (self.mConfig.mRadiusFishingRegion * 3) // 4,
                                   self.mConfig.mTextColor, self.mConfig.mThickness, cv2.LINE_AA)
        mCurrFrameRGB = cv2.circle(mCurrFrameRGB, (mImgCenterX, mImgCenterY),
                                   self.mConfig.mRadiusFishingRegion // 4,
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

            # neu height > 2 * weight, bo qua de tranh hat mua
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

        self.mSignalUpdateImageShow.emit()
        return mFishArea

    def CheckMark(self):
        mStaticFrameGray = None
        if self.mConfig.mFishDetectionCheck is True:
            mStaticFrameRGB = self.mScreenHandle.RegionScreenshot(self.mFishingRegion)
            if mStaticFrameRGB is None:
                return
            mStaticFrameGray = cv2.cvtColor(mStaticFrameRGB, cv2.COLOR_BGR2GRAY)
            self.mImageShow = mStaticFrameRGB

        self.StatusEmit("Đang câu cá")
        log.info(f'Fishing')
        mPixelBaseMark = None

        for i in range(3):
            mPixelBaseMark = self.mScreenHandle.PixelScreenShot(self.mMark)
            if mPixelBaseMark is None:
                time.sleep(0.001)
                continue
            break

        mBaseTime = time.time()
        mStopDetect = False
        mSkipFrame = 0

        fpsTime = time.time()
        time.sleep(0.01)
        while (time.time() - mBaseTime) < self.mConfig.mFishingPeriod:
            # t1 = fpsTime
            # fpsTime = time.time()
            # print(f'FPS = {1 / (fpsTime - t1)}')
            if self.mAutoFishRunning is False:
                return Flags.STOP_FISHING

            if (time.time() - mBaseTime) < self.mConfig.mWaitingFishTime:
                time.sleep(0.1)
                continue

            if self.mConfig.mFishDetectionCheck is True and mStopDetect is False and (
                    time.time() - mBaseTime) < self.mConfig.mWaitingMarkTime:
                mCurrentFrameRGB = self.mScreenHandle.RegionScreenshot(self.mFishingRegion)
                if mCurrentFrameRGB is None:
                    return
                mCurrentFrameGray = cv2.cvtColor(mCurrentFrameRGB, cv2.COLOR_BGR2GRAY)
                mSizeFish = self.FishDetection(mStaticFrameGray, mCurrentFrameGray, mCurrentFrameRGB)
                if mSizeFish != 0:
                    mSkipFrame += 1
                if mSkipFrame == 10:
                    mStopDetect = True
                    log.info(f'Size Fish = {mSizeFish}')
                    if mSizeFish < self.mConfig.mFishSize:
                        return
            # mode cheat engine
            if self.mConfig.mCheatEngine is True:
                check = self.mReadMemory.GetData(self.mReadMemory.mRodAddress)
                print(f'check = {check}')
                if check == 4:
                    return
                time.sleep(0.001)

            # Mode ko cheat engine
            else:
                if (time.time() - mBaseTime) < self.mConfig.mWaitingMarkTime:
                    time.sleep(0.02)
                    continue

                mPixelCurrMark = self.mScreenHandle.PixelScreenShot(self.mMark)
                if mPixelCurrMark is None:
                    # print("none pixel")
                    time.sleep(0.001)
                    continue
                mDiffRgbMark = self.ComparePixel(mPixelCurrMark, mPixelBaseMark)
                if mDiffRgbMark > 50:
                    log.info(f'mDiffRgb = {mDiffRgbMark}')
                    return
                time.sleep(0.015)
        self.StatusEmit("Hết chu kỳ câu. Kéo cần")
        log.info(f'End fishing period. Pulling Rod')
        return

    def PullFishingRod(self):
        if self.mConfig.mSendKeyCheck is True:
            self.mScreenHandle.SendKey()
            self.StatusEmit("Đang kéo cần câu")
            log.info('PullFishingRod')
            return
        else:
            time1 = time.time()
            self.AdbClick(self.mConfig.mPullingRodPos[0],
                          self.mConfig.mPullingRodPos[1])
            timeDelay = time.time() - time1
            self.StatusEmit(f'Đang kéo cần câu. Độ trễ giật cần {round(timeDelay, 2)} giây')
            log.info(f'Clicked {self.mConfig.mPullingRodPos}. Delay = {round(timeDelay, 2)} sec')
            if timeDelay > 0.5:
                self.mCheckAdbDelay += 1
                if self.mCheckAdbDelay <= 3:
                    return
                self.mAutoFishRunning = False
                self.MsgEmit(
                    'Kéo cần qua ADB bị trễ quá cao\nHãy chọn chế độ kéo cần bằng phím Space (KHÔNG hỗ trợ trên LD Player)')
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
            mCheckBackpack = self.mScreenHandle.FindImage(self.mConfig.mBackpackImg,
                                                          self.mConfig.mBackpackRegion,
                                                          self.mConfig.mConfidence)
            if mCheckBackpack == Flags.TRUE:
                self.StatusEmit("Câu thất bại")
                log.info(f'Fishing fail')
                return

            mCheckPreservation = self.mScreenHandle.FindImage(self.mConfig.mPreservationImg,
                                                              self.mConfig.mFishingResultRegion,
                                                              self.mConfig.mConfidence)
            if mCheckPreservation == Flags.TRUE:
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
        time.sleep(0.05)
        mFishImage = self.mScreenHandle.RegionScreenshot(self.mConfig.mFishImgRegion)
        if mFishImage is None:
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

        self.mSignalUpdateImageShow.emit()

    def SetMarkPos(self):
        if self.mEmulatorBox is None:
            self.MsgEmit("Chưa kết nối cửa sổ giả lập\t\t")
            return
        if self.mAdbDevice is None:
            self.MsgEmit("Chưa kết nối địa chỉ ADB của thiết bị")
            return
        self.mMark = [0, 0]
        time.sleep(0.1)
        self.StatusEmit(
            "Đưa chuột đến ĐỈNH của CHẤM THAN và bấm chuột\nChấm đỏ phải nằm trong CHẤM THAN")
        self.mCheckMouseRunning = True
        while self.mCheckMouseRunning is True:
            # kiem tra xem cham than co nam ngoai cua so game khong?
            self.mCheckMarkPos = False
            mAbsMousePos = win32gui.GetCursorPos()

            if mAbsMousePos[0] - self.mConfig.mMarkPixelRadius > self.mEmulatorBox[0] and \
                    mAbsMousePos[0] + self.mConfig.mMarkPixelRadius < self.mConfig.mEmulatorSize[0] + self.mEmulatorBox[
                0] and \
                    mAbsMousePos[1] - self.mConfig.mMarkPixelRadius - self.mScreenHandle.mTopBar > self.mEmulatorBox[
                1] and \
                    mAbsMousePos[1] + self.mConfig.mMarkPixelRadius < self.mEmulatorBox[1] + self.mEmulatorBox[3]:
                self.mMark = self.ConvertCoordinates(mAbsMousePos)
                self.mSignalSetPixelPos.emit(self.mMark[0], self.mMark[1])

                mRegion = [self.mMark[0] - self.mConfig.mMarkPixelRadius, self.mMark[1] - self.mConfig.mMarkPixelRadius,
                           2 * self.mConfig.mMarkPixelRadius, 2 * self.mConfig.mMarkPixelRadius]
                mTempImage = self.mScreenHandle.RegionScreenshot(mRegion)
                mTempImage = cv2.circle(mTempImage, (self.mConfig.mMarkPixelRadius, self.mConfig.mMarkPixelRadius),
                                        1, (0, 0, 255), 1, cv2.LINE_AA)
                # mTempImage = cv2.circle(mTempImage,
                #                         (self.mConfig.mMarkPixelRadius,
                #                          self.mConfig.mMarkPixelRadius - self.mConfig.mMarkPixelDist),
                #                         1, (0, 0, 255), 1, cv2.LINE_AA)
                # mTempImage = cv2.circle(mTempImage,
                #                         (self.mConfig.mMarkPixelRadius - self.mConfig.mMarkPixelDist,
                #                          self.mConfig.mMarkPixelRadius),
                #                         1, (0, 0, 255), 1, cv2.LINE_AA)
                # mTempImage = cv2.circle(mTempImage,
                #                         (self.mConfig.mMarkPixelRadius + self.mConfig.mMarkPixelDist,
                #                          self.mConfig.mMarkPixelRadius),
                #                         1, (0, 0, 255), 1, cv2.LINE_AA)
                mTempImage = cv2.rectangle(mTempImage,
                                           (self.mConfig.mMarkPixelRadius - self.mConfig.mMarkPixelDist // 2,
                                            self.mConfig.mMarkPixelRadius - self.mConfig.mMarkPixelDist // 2),
                                           (self.mConfig.mMarkPixelRadius + self.mConfig.mMarkPixelDist // 2,
                                            self.mConfig.mMarkPixelRadius + self.mConfig.mMarkPixelDist * 2),
                                           (0, 0, 255), 1, cv2.LINE_AA)
                self.mImageShow = mTempImage.copy()
                self.mSignalUpdateImageShow.emit()
                self.mCheckMarkPos = True

            for i in range(30):
                if self.CheckLeftMouseClick() is True:
                    self.mCheckMouseRunning = False
                time.sleep(0.001)

        if self.mCheckMarkPos is False:
            self.MsgEmit('Lỗi vị trí chấm than đã lấy nằm ngoài cửa sổ giả lập')
            self.mMark = [0, 0]

        self.StatusEmit(f'Vị trí chấm than trên cửa sổ game:\n{self.mMark}')
        log.info(f'Mark position in game = {self.mMark}')

    def SetFishingBobberPos(self):
        if self.mEmulatorBox is None:
            self.MsgEmit("Chưa kết nối cửa sổ giả lập")
            return

        if self.mAdbDevice is None:
            self.MsgEmit("Chưa kết nối địa chỉ ADB của thiết bị")
            return
        self.mFishingRegion = [0, 0, 0, 0]

        self.StatusEmit("Di chuyển chuột đến phao câu và Click")
        self.mCheckMouseRunning = True

        while self.mCheckMouseRunning is True:
            # kiem tra xem vi tri phao cau co qua gan vien gia lap?
            self.mCheckBobberPos = False
            mAbsMousePos = win32gui.GetCursorPos()

            if mAbsMousePos[0] - self.mConfig.mRadiusFishingRegion > self.mEmulatorBox[0] and \
                    mAbsMousePos[0] + self.mConfig.mRadiusFishingRegion < self.mConfig.mEmulatorSize[0] + \
                    self.mEmulatorBox[
                        0] and \
                    mAbsMousePos[1] - self.mConfig.mRadiusFishingRegion - self.mScreenHandle.mTopBar > \
                    self.mEmulatorBox[1] and \
                    mAbsMousePos[1] + self.mConfig.mRadiusFishingRegion < self.mEmulatorBox[1] + self.mEmulatorBox[3]:
                mRelativeMousePos = self.ConvertCoordinates(mAbsMousePos)
                self.mSignalSetFishingBobberPos.emit(mRelativeMousePos[0], mRelativeMousePos[1])

                self.mFishingRegion[0] = mRelativeMousePos[0] - self.mConfig.mRadiusFishingRegion
                self.mFishingRegion[1] = mRelativeMousePos[1] - self.mConfig.mRadiusFishingRegion
                self.mFishingRegion[2] = self.mConfig.mRadiusFishingRegion * 2
                self.mFishingRegion[3] = self.mConfig.mRadiusFishingRegion * 2

                mRegion = [mRelativeMousePos[0] - self.mConfig.mRadiusFishingRegion,
                           mRelativeMousePos[1] - self.mConfig.mRadiusFishingRegion,
                           self.mConfig.mRadiusFishingRegion * 2,
                           self.mConfig.mRadiusFishingRegion * 2]
                mTempImage = self.mScreenHandle.RegionScreenshot(mRegion)
                mTempImage = cv2.circle(mTempImage,
                                        (self.mConfig.mRadiusFishingRegion, self.mConfig.mRadiusFishingRegion),
                                        1, (0, 0, 255), 5, cv2.LINE_AA)
                mTempImage = cv2.circle(mTempImage,
                                        (self.mConfig.mRadiusFishingRegion, self.mConfig.mRadiusFishingRegion),
                                        (self.mConfig.mRadiusFishingRegion * 3) // 4,
                                        (0, 0, 255),
                                        self.mConfig.mThickness, cv2.LINE_AA)
                mTempImage = cv2.circle(mTempImage,
                                        (self.mConfig.mRadiusFishingRegion, self.mConfig.mRadiusFishingRegion),
                                        self.mConfig.mRadiusFishingRegion // 4,
                                        (0, 0, 255),
                                        self.mConfig.mThickness, cv2.LINE_AA)
                self.mImageShow = mTempImage.copy()
                self.mSignalUpdateImageShow.emit()
                self.mCheckBobberPos = True

            for i in range(30):
                if self.CheckLeftMouseClick() is True:
                    self.mCheckMouseRunning = False
                time.sleep(0.001)

        if self.mCheckBobberPos is False:
            self.MsgEmit(
                'Lỗi vị trí phao câu quá gần viền cửa sổ giả lập.\nĐiều chỉnh góc nhìn của nhân vật để phao câu xa viền màn hình hơn')
            self.mFishingRegion = [0, 0, 0, 0]

        self.StatusEmit(f'Vùng câu trong game:\n{self.mFishingRegion}')
        log.info(f'Fishing region in game = {self.mFishingRegion}')

    def CheckRegionEmulator(self):
        try:
            mWindowTitle = pyautogui.getWindowsWithTitle(self.mConfig.mWindowName)[-1].title
        except (ValueError, Exception):
            self.MsgEmit(f'Không tìm thấy cửa sổ giả lập {self.mConfig.mWindowName}. Hãy kiểm tra tên cửa sổ')
            log.info(f'pyautogui cannot find window name {self.mConfig.mWindowName}')
            return False

        check = self.mScreenHandle.CheckWindowApplication(mWindowTitle)
        if check is False:
            self.MsgEmit(f'Không tìm thấy cửa sổ giả lập {self.mConfig.mWindowName}. Hãy kiểm tra tên cửa sổ')
            log.info(f'win32gui cannot find window name {self.mConfig.mWindowName}')
            return False

        self.mScreenHandle.ActivateWindow()
        self.mScreenHandle.SetWindowApplication(self.mConfig.mEmulatorSize[0],
                                                self.mConfig.mEmulatorSize[1])

        self.mEmulatorBox = self.mScreenHandle.GetWindowBox()
        if self.mEmulatorBox[2] < 200:
            self.MsgEmit(f'Giả lập không được để tên mặc định.\n'
                         f'Hãy đổi tên giả lập. Yêu cầu như sau:\n'
                         f'- Viết liền không dấu.\n'
                         f'- Không được trùng với tên cửa sổ khác.\n'
                         f'- Ví dụ auto01, auto02, auto03, ...')
            log.info(f'Error default name')
            return False

        self.StatusEmit(f'Đã tìm thấy cửa sổ giả lập\n{self.mEmulatorBox}')

        log.info(f'Found {self.mConfig.mWindowName} = {mWindowTitle}, box = {self.mEmulatorBox}')
        mEmulatorSize = self.mConfig.mEmulatorSize

        if abs(mEmulatorSize[0] - self.mEmulatorBox[2]) > 100 or abs(
                mEmulatorSize[1] - self.mEmulatorBox[3]) > 100:
            self.MsgEmit(
                f'Cửa sổ giả lập không phải {self.mConfig.mStrListEmulatorSize[self.mConfig.mEmulatorSizeId]}\n'
                f'Hãy cài đặt độ phân giải giả lập về một trong 3 dạng sau:\n'
                f'1280x720 (dpi 240)\n'
                f'960x540 (dpi 160)\n'
                f'640x360 (dpi 120)')
            log.info(f'Emulator size {self.mEmulatorBox} not suitable')
            return False

        self.mEmulatorType = self.mScreenHandle.FindLogo()
        if self.mEmulatorType == OTHER:
            self.MsgEmit(
                f'Không tìm thấy giả lập. Hãy đổi tên giả lập\n'
                f'Lưu ý phần mềm chỉ hỗ trợ 3 loại giả lập LDPlayer, NOX, MEmu')
            log.info(f'Emulator type not suitable')
            return False

        if self.mEmulatorType == NOX:
            self.mImageShow = cv2.imread('data/noxlogo.png')
        elif self.mEmulatorType == LD:
            self.mImageShow = cv2.imread('data/ldlogo.png')
        elif self.mEmulatorType == MEMU:
            self.mImageShow = cv2.imread('data/memulogo.png')
        else:
            pass
        self.mSignalUpdateImageShow.emit()
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
                self.MsgEmit('Không tìm thấy ADB server\t\t')
                return False
            else:
                self.mAdbClient = AdbClient(self.mConfig.mAdbHost, self.mConfig.mAdbPort)
                self.mAdbDevices = self.mAdbClient.devices()
        if len(self.mAdbDevices) == 0:
            self.mAdbClient = None
            self.MsgEmit('Không kết nối được giả lập qua adb-server\nHãy khởi động lại giả lập')
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
        try:
            self.mAdbDevice.shell(f'input tap {str(mCoordinateX)} {str(mCoordinateY)}')
        except (ValueError, Exception):
            self.MsgEmit("Mất kết nối địa chỉ ADB đến giả lập.\n"
                         "Để tránh xung đột ADB, KHÔNG được bật cùng lúc 2 loại giả lập khác nhau")
            self.mAutoFishRunning = False

    def AdbDoubleClick(self, mCoordinateX, mCoordinateY):
        try:
            self.mAdbDevice.shell(
                f'input tap {str(mCoordinateX)} {str(mCoordinateY)} & sleep 0.1; input tap {str(mCoordinateX)} {str(mCoordinateY)}')
        except (ValueError, Exception):
            self.MsgEmit("Mất kết nối ADB đến giả lập")
            self.mAutoFishRunning = False

    def AdbHoldClick(self, mCoordinateX, mCoordinateY, mTime):
        try:
            self.mAdbDevice.shell(
                f'input swipe {str(mCoordinateX)} {str(mCoordinateY)} {str(mCoordinateX)} {str(mCoordinateY)} {str(mTime)}')
        except (ValueError, Exception):
            self.MsgEmit("Mất kết nối ADB đến giả lập")
            self.mAutoFishRunning = False

    def StartAuto(self):
        if self.mCaptchaRecognition is None:
            self.InitClassification()

        self.mAutoFishRunning = True
        if self.mEmulatorBox is None:
            self.MsgEmit("Chưa kết nối cửa sổ giả lập\t\t")
            return

        if self.mAdbDevice is None:
            self.MsgEmit("Chưa kết nối địa chỉ ADB của giả lập\t\t")
            return

        if self.mMark[0] == 0 and self.mMark[1] == 0:
            self.MsgEmit("Chưa lấy tọa độ chấm than \t\t")
            return

        if self.mFishingRegion[2] == 0:
            self.MsgEmit("Chưa lấy tọa độ phao câu\t\t")
            return

        if self.mEmulatorType == LD:
            self.mConfig.mSendKeyCheck = False
        elif self.mEmulatorType == NOX:
            self.mConfig.mSendKeyCheck = True
        else:
            pass

        if self.mConfig.mCheatEngine is True:
            print(f'self.mReadMemory.mBaseAddress {self.mReadMemory.mBaseAddress}')
            if self.mReadMemory.mBaseAddress == 0:
                print("Fail Base Address")
                return

        time.sleep(0.1)
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
                    self.MsgEmit('Giải captcha sai 8 lần liên tiếp. Thoát game để tránh bị ban 24h câu')
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
                if self.mFixRodTime == 20:
                    self.MsgEmit("Thả câu lỗi 20 lần liên tiếp. Kiểm tra xem có hết lượt câu không?")
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
                    self.MsgEmit('Giải captcha sai 8 lần\t\t')
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
            mCheckCaptcha = self.mScreenHandle.FindImage(self.mConfig.mOKCaptchaImg,
                                                         self.mConfig.mOKCaptchaRegion,
                                                         self.mConfig.mConfidence)
            if mCheckCaptcha == Flags.TRUE:
                return Flags.CAPTCHA_APPEAR
            time.sleep(0.05)
            mCheck += 1
        return Flags.CAPTCHA_NONE

    def CaptchaHandle(self):
        if self.mAutoFishRunning is False:
            return Flags.STOP_FISHING

        log.info('Captcha Handle Start')
        self.StatusEmit("Phát hiện Captcha. Đang xử lý ...")
        mBigCaptchaImage = self.mScreenHandle.RegionScreenshot(self.mConfig.mListCaptchaRegion[0])
        if mBigCaptchaImage is None:
            return

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
            mSmallCaptchaImage = self.mScreenHandle.RegionScreenshot(self.mConfig.mListCaptchaRegion[i])
            if mSmallCaptchaImage is None:
                return

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
        time.sleep(2)
        log.info('Captcha Handle Complete')
        return
