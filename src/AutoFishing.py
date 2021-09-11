import math
import time
import pyautogui
import os
import gc
import cv2
from ppadb.client import Client as AdbClient
import numpy
import win32api
from src.config import Config
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QMessageBox


class AutoFishing(QObject):
    mSignalSetPixelPos = pyqtSignal(int, int)
    mSignalSetFishingBobberPos = pyqtSignal(int, int)
    mSignalUpdateFishingNum = pyqtSignal(int)
    mSignalUpdateFishNum = pyqtSignal(int)
    mSignalUpdateFishDetectionImage = pyqtSignal()
    mSignalMessage = pyqtSignal(str)

    def __init__(self):
        QObject.__init__(self, parent=None)  # Kế thừa QObject
        self.mConfig = Config()
        self.mFishingNum = 0
        self.mRodFixedCheck = 0
        self.mFishNum = 0
        self.mPullingRodPos = [0, 0]
        self.mMark = [0, 0]
        self.mFishingRegion = [0, 0, 0, 0]
        self.mAdbDevice = None
        self.mFishDetectionRunning = True
        self.mCheckMouseRunning = True
        self.mAutoFishRunning = True
        self.mCheckFish = False
        self.mEmulatorWindow = None
        self.mEmulatorBox = None
        self.mFishImage = None
        self.mCurrentTime = time.time()

    def __del__(self):
        self.mFishDetectionRunning = False
        self.mCheckMouseRunning = False
        self.mAutoFishRunning = False

    @staticmethod
    def MsgBox(mText: str):
        msgBox = QMessageBox()
        msgBox.setText(mText)
        msgBox.setWindowTitle("Thông báo")
        msgBox.exec()

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

    def TakeFishingRod(self):
        self.AdbClick(self.mConfig.GetFishingRodPosition()[0],
                      self.mConfig.GetFishingRodPosition()[1])
        print("Đã lấy cần câu trong ba lô")

    def FixClick(self):
        mCheck = 0
        mRegionFixClickX = self.mEmulatorBox.left + self.mConfig.GetFishingRodPosition()[0] - 100
        mRegionFixClickY = self.mEmulatorBox.top + self.mEmulatorBox.height + \
                           self.mConfig.GetFishingRodPosition()[1] - 820
        while mCheck < 3:
            mFixedRodButtonPos = pyautogui.locateOnScreen(f'{self.mConfig.GetDataPath()}fix_rod.png',
                                                          grayscale=True,
                                                          region=(mRegionFixClickX, mRegionFixClickY, 200, 200),
                                                          confidence=self.mConfig.GetConfidence())

            if mFixedRodButtonPos is not None:
                self.AdbClick(self.mConfig.GetFishingRodPosition()[0],
                              self.mConfig.GetFishingRodPosition()[1])
                print("Đã bấm nút sửa cần")
                return True

            mCheck += 1
            time.sleep(1)

        print("Không tìm được nút sửa cần")
        return False

    def FixConfirm(self):
        mCheck = 0
        while mCheck < 3:
            mFixedRodConfirmButtonPos = pyautogui.locateOnScreen(
                f'{self.mConfig.GetDataPath()}confirm.png',
                grayscale=True,
                region=(self.mEmulatorBox.left,
                        self.mEmulatorBox.top,
                        self.mEmulatorBox.width,
                        self.mEmulatorBox.height),
                confidence=self.mConfig.GetConfidence())

            if mFixedRodConfirmButtonPos is not None:
                self.AdbClick(self.mConfig.GetConfirm()[0],
                              self.mConfig.GetConfirm()[1])
                print("Đã xác nhận sửa cần")
                return True

            mCheck += 1
            time.sleep(1)

        print("Không tìm thấy xác nhận sửa cần")
        return False

    def ClickOk(self):
        mCheck = 0
        while mCheck < 3:
            mFixedRodDoneButtonPos = pyautogui.locateOnScreen(
                f'{self.mConfig.GetDataPath()}OK.png',
                grayscale=True,
                region=(self.mEmulatorBox.left,
                        self.mEmulatorBox.top,
                        self.mEmulatorBox.width,
                        self.mEmulatorBox.height),
                confidence=self.mConfig.GetConfidence())

            if mFixedRodDoneButtonPos is not None:
                self.AdbClick(self.mConfig.GetOKButton()[0],
                              self.mConfig.GetOKButton()[1])
                print("Đã bấm nút OK sau khi sửa cần")
                return True

            mCheck += 1
            time.sleep(1)

        print("Không tìm thấy nút OK sau khi sửa cần")
        return False

    def CheckRod(self):
        time.sleep(3)
        mCheck = 0
        while mCheck < 3:
            mPullRodButtonPos = pyautogui.locateOnScreen(f'{self.mConfig.GetDataPath()}pull_rod.png',
                                                         grayscale=True,
                                                         region=(self.mEmulatorBox.left,
                                                                 self.mEmulatorBox.top,
                                                                 self.mEmulatorBox.width,
                                                                 self.mEmulatorBox.height),
                                                         confidence=self.mConfig.GetConfidence())
            if mPullRodButtonPos is not None:
                self.mPullingRodPos[0] = mPullRodButtonPos[0] + mPullRodButtonPos[2] // 2
                self.mPullingRodPos[1] = mPullRodButtonPos[1] + mPullRodButtonPos[3] // 2
                return True

            mCheck += 1
            time.sleep(1)

        print("Không thể thả cần câu. Tiến hành kiểm tra túi đồ")
        self.OpenBackPack()
        time.sleep(3)
        self.OpenTools()
        time.sleep(2)
        self.FixClick()
        time.sleep(2)
        self.FixConfirm()
        time.sleep(2)
        self.ClickOk()
        time.sleep(2)
        self.CloseBackPack()
        time.sleep(2)
        mCheck = 0
        while mCheck < 3:
            mCastFishingRodButtonPos = pyautogui.locateOnScreen(f'{self.mConfig.GetDataPath()}cast_fishing_rod.png',
                                                                grayscale=True,
                                                                region=(self.mEmulatorBox.left,
                                                                        self.mEmulatorBox.top,
                                                                        self.mEmulatorBox.width,
                                                                        self.mEmulatorBox.height),
                                                                confidence=self.mConfig.GetConfidence())
            if mCastFishingRodButtonPos is not None:
                self.mRodFixedCheck += 1
                return False
            mCheck += 1
            time.sleep(0.5)
        self.AdbClick(self.mConfig.GetPreservation()[0],
                      self.mConfig.GetPreservation()[1])
        self.mRodFixedCheck += 1
        return False

    def CastFishingRod(self):
        mCheck = 0
        while mCheck < 5:
            mCastFishingRodButtonPos = pyautogui.locateOnScreen(f'{self.mConfig.GetDataPath()}cast_fishing_rod.png',
                                                                grayscale=True,
                                                                region=(self.mEmulatorBox.left,
                                                                        self.mEmulatorBox.top,
                                                                        self.mEmulatorBox.width,
                                                                        self.mEmulatorBox.height),
                                                                confidence=self.mConfig.GetConfidence())
            if mCastFishingRodButtonPos is not None:
                self.AdbClick(self.mConfig.GetCastingRod()[0],
                              self.mConfig.GetCastingRod()[1])
                print("Thả cần câu")
                return True
            mCheck += 1
            time.sleep(0.01)
        print("Không tìm thấy nút thả cần câu")
        return False

    def FishDetection(self, mPrevFrame, mCurrFrame):
        # tối ở camp 49
        if mPrevFrame[0, 0] <= 50:
            mMinThreshValue = 10
            mMaxThreshValue = 100
            mColor = (255, 255, 255)
        # tối ở biển 57
        elif 50 < mPrevFrame[0, 0] <= 70:
            mMinThreshValue = 15
            mMaxThreshValue = 100
            mColor = (255, 255, 255)
        # buổi chiều nền biền 74, sáng ở camp 149, chiều ở cam 166
        elif 70 < mPrevFrame[0, 0] < 170:
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
                                self.mConfig.GetRadiusFishingRegion() - 50, mColor, 1)
        mCurrFrame = cv2.circle(mCurrFrame, (mImgCenterX, mImgCenterY), 50, mColor, 1)
        cv2.putText(mCurrFrame, str(mPrevFrame[0, 0]), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, mColor, 2)
        for mContour in mContours:
            # check coordinates of all found contours
            (x, y, w, h) = cv2.boundingRect(mContour)
            mContourCenterX = x + w // 2
            mContourCenterY = y + h // 2
            mRadius = math.sqrt(pow((mImgCenterX - mContourCenterX), 2) + pow((mImgCenterY - mContourCenterY), 2))
            # loại bỏ phao câu
            if mRadius < 50:
                continue
            # loại box nhỏ tránh nhiễu
            if cv2.contourArea(mContour) < 100:
                continue
            # loại bỏ box xuất hiện ở viền
            if mRadius > self.mConfig.GetRadiusFishingRegion() - 50:
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
        time.sleep(1)
        mStaticFrame = None
        if self.mConfig.GetFishDetection() is True:
            mStaticFrame = self.ScreenshotFishingRegion()
            self.mFishImage = cv2.resize(mStaticFrame, (200, 200), interpolation=cv2.INTER_AREA)
        print("Đang đợi dấu chấm than")
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
            print("Lỗi hệ thống. Không xác định được màu nền tại vị trí dấu chấm than")
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
                        print("Kích thước cá =", int(mSizeFish), ".Bỏ qua")
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
                print("Phát hiện dấu chấm than. Độ lệch màu = ", mDiffRgb)
                return True

            time2 = time.time()
            time.sleep(0.01)

        print("Không phát hiện dấu chấm than")
        return False

    def PullFishingRod(self):
        if self.mConfig.GetFreeMouse() is False:
            pyautogui.click(x=self.mPullingRodPos[0],
                            y=self.mPullingRodPos[1],
                            clicks=2,
                            interval=0.1,
                            button='left')
            print("Đang kéo cần câu")
            return True
        else:
            print("Đang kéo cần câu")
            time1 = time.time()
            self.AdbClick(self.mConfig.GetPullingRod()[0],
                          self.mConfig.GetPullingRod()[1])
            time2 = time.time()
            print("Độ trễ giật cần ", round(time2 - time1, 2), " giây")
            return True

    def FishPreservation(self):
        time.sleep(2)
        time1 = time.time()
        time2 = time.time()
        mCheckPullRodButton = 0
        while (time2 - time1) < int(self.mConfig.GetPullingFishTime()):
            mCastFishingButtonPos = pyautogui.locateOnScreen(f'{self.mConfig.GetDataPath()}cast_fishing_rod.png',
                                                             region=(self.mEmulatorBox.left,
                                                                     self.mEmulatorBox.top,
                                                                     self.mEmulatorBox.width,
                                                                     self.mEmulatorBox.height),
                                                             grayscale=True,
                                                             confidence=self.mConfig.GetConfidence())
            mPullRodButtonPos = pyautogui.locateOnScreen(f'{self.mConfig.GetDataPath()}pull_rod.png',
                                                         region=(self.mEmulatorBox.left,
                                                                 self.mEmulatorBox.top,
                                                                 self.mEmulatorBox.width,
                                                                 self.mEmulatorBox.height),
                                                         grayscale=True,
                                                         confidence=self.mConfig.GetConfidence())
            if mCastFishingButtonPos is not None:
                print("Câu thất bại")
                self.mRodFixedCheck = 0
                return True
            if mPullRodButtonPos is None:
                mCheckPullRodButton += 1
                if mCheckPullRodButton > 2:
                    print("Câu thành công")
                    self.mFishNum += 1
                    self.AdbClick(self.mConfig.GetPreservation()[0],
                                  self.mConfig.GetPreservation()[1])
                    self.mRodFixedCheck = 0
                    return True

            time.sleep(0.2)
            time2 = time.time()

        self.mRodFixedCheck = 0
        print("Kiểm tra kết quả bị lỗi")
        return False

    def SetPixelPos(self):
        self.mMark = [0, 0]
        time.sleep(0.1)
        mMousePos = pyautogui.position()
        print("\nDi chuyển chuột đến đầu của dấu chấm than và Click")
        self.mCheckMouseRunning = True
        while self.mCheckMouseRunning is True:
            mMousePos = pyautogui.position()
            self.mSignalSetPixelPos.emit(int(mMousePos.x), int(mMousePos.y))
            if self.CheckLeftMouseClick() is True:
                self.mCheckMouseRunning = False
            time.sleep(0.01)
        self.mMark[0] = int(mMousePos.x)
        self.mMark[1] = int(mMousePos.y)
        print("Vị trí dấu chấm than đã cài đặt: ", mMousePos)

    def SetFishingBobberPos(self):
        self.mFishingRegion = [0, 0, 0, 0]
        time.sleep(0.1)
        mScreenSize = pyautogui.size()
        mMousePos = pyautogui.position()
        print("\nDi chuyển chuột đến phao câu và Click")
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
        if self.mFishingRegion[0] <= 0:
            self.mFishingRegion[0] = 1
            self.mFishingRegion[2] = 2 * (mMousePos.x - self.mFishingRegion[0])
        if self.mFishingRegion[1] <= 0:
            self.mFishingRegion[1] = 1
            self.mFishingRegion[3] = 2 * (mMousePos.y - self.mFishingRegion[1])

        if self.mFishingRegion[0] + self.mFishingRegion[2] >= mScreenSize.width:
            self.mFishingRegion[0] = 2 * mMousePos.x - mScreenSize.width
            self.mFishingRegion[2] = 2 * (mMousePos.x - self.mFishingRegion[0])

        if self.mFishingRegion[1] + self.mFishingRegion[3] >= mScreenSize.height:
            self.mFishingRegion[1] = 2 * mMousePos.y - mScreenSize.height
            self.mFishingRegion[3] = 2 * (mMousePos.y - self.mFishingRegion[1])
        print("Vị trí phao câu đã cài đặt: ", mMousePos)
        time.sleep(2)

    def CheckRegionEmulator(self):
        mScreenSize = pyautogui.size()
        self.mEmulatorBox = None
        self.mEmulatorWindow = None
        mEmulatorWindows = []
        print("\nKích thước màn hình = ", mScreenSize)
        try:
            mEmulatorWindows = pyautogui.getWindowsWithTitle(self.mConfig.GetWindowName())
        except:
            self.MsgBox(f'Không tìm thấy cửa sổ {self.mConfig.GetWindowName()}')
            return False
        if len(mEmulatorWindows) > 0:
            self.mEmulatorWindow = mEmulatorWindows[0]
        else:
            self.MsgBox(f'Không tìm thấy cửa sổ {self.mConfig.GetWindowName()}')
            return False
        self.mEmulatorBox = self.mEmulatorWindow.box
        print("\nĐã tìm thấy cửa sổ giả lập ", self.mEmulatorBox)
        time.sleep(1)
        if self.mEmulatorBox.width < 1280 or self.mEmulatorBox.height < 720:
            self.MsgBox("Cửa sổ giả lập bị ẩn hoặc độ phân giải không phù hợp")
            return False
        if self.mEmulatorBox.top < 0:
            self.mEmulatorWindow.activate()
            self.mEmulatorWindow.move(0, abs(self.mEmulatorBox.top))
            print("Cửa sổ giả lập bị khuất về bên trên, tự động di chuyển")
        if self.mEmulatorBox.left < 0:
            self.mEmulatorWindow.activate()
            self.mEmulatorWindow.move(abs(self.mEmulatorBox.left), 0)
            print("Cửa sổ giả lập bị khuất về bên trái, tự động di chuyển")
        if self.mEmulatorBox.top + self.mEmulatorBox.height > mScreenSize.height:
            self.mEmulatorWindow.activate()
            self.mEmulatorWindow.move(0, 0 - self.mEmulatorBox.top)
            print("Cửa sổ giả lập bị khuất về bên dưới, tự động di chuyển")
        if self.mEmulatorBox.left + self.mEmulatorBox.width > mScreenSize.width:
            self.mEmulatorWindow.activate()
            self.mEmulatorWindow.move(0 - self.mEmulatorBox.left, 0)
            print("Cửa sổ giả lập bị khuất về bên phải, tự động di chuyển")
        time.sleep(0.1)
        self.mEmulatorBox = self.mEmulatorWindow.box
        self.MsgBox("Kết nối giả lập thành công")
        time.sleep(0.1)
        return True

    def StartAdbServer(self):
        print("\nĐang khởi tạo adb-server\n")
        try:
            os.system(f'{self.mConfig.GetCurrentPath()}\\adb\\adb.exe devices')
        except:
            print('\nKhởi tạo adb-server thất bại')
            return False
        print('\nKhởi tạo adb-server thành công')
        time.sleep(3)
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
                self.MsgBox('Không tìm thấy adb-server')
                return False
            else:
                mAdbClient = AdbClient(self.mConfig.GetAdbHost(),
                                       self.mConfig.GetAdbPort())
                mDevices = mAdbClient.devices()
        if mDevices is None:
            self.MsgBox('Không tìm thấy phần mềm giả lập')
            return False

        if len(mDevices) == 0:
            self.MsgBox('Không tìm thấy phần mềm giả lập')
            return False
        elif len(mDevices) == 1:
            print("Đã tìm thấy phần mềm giả lập")
            self.mAdbDevice = mDevices[0]
        else:
            print("Đã tìm thấy nhiều thiết bị android:")
            for i in range(len(mDevices)):
                print("Mã số ", i, "   :   ", mDevices[i].client)
            m_device_index = int(input("Hãy nhập thiết bị bạn sử dụng theo mã số tương ứng và bấm Enter:"))
            self.mAdbDevice = mDevices[m_device_index]

        if self.mAdbDevice is None:
            self.MsgBox('Không tìm thấy phần mềm giả lập')
            return False

        print("Kết nối giả lập thành công")
        return True

    def AdbDisconnect(self):
        print("\nĐang đóng adb-server")
        os.system(f'{self.mConfig.GetCurrentPath()}\\adb\\adb.exe kill-server')
        time.sleep(3)

    def AdbClick(self, mCoordinateX, mCoordinateY):
        self.mAdbDevice.shell(f'input tap {str(mCoordinateX)} {str(mCoordinateY)}')

    def AdbDoubleClick(self, mCoordinateX, mCoordinateY):
        self.mAdbDevice.shell(
            f'input tap {str(mCoordinateX)} {str(mCoordinateY)} & sleep 0.1; input tap {str(mCoordinateX)} {str(mCoordinateY)}')

    def AdbHoldClick(self, mCoordinateX, mCoordinateY, mTime):
        self.mAdbDevice.shell(
            f'input swipe {str(mCoordinateX)} {str(mCoordinateY)} {str(mCoordinateX)} {str(mCoordinateY)} {str(mTime)}')

    def StartAuto(self):
        print(self.mEmulatorBox)
        if self.mEmulatorBox is None:
            print(1)
            self.MsgBox("")
            time.sleep(0.1)

            return

        if self.mMark[0] == 0:
            self.MsgBox('Chưa xác định vị trí dấu chấm than')
            return False
        print(2)

        if self.mConfig.GetFishDetection() is True:
            if self.mFishingRegion[0] == 0:
                self.MsgBox('Chưa xác định vùng câu')
                return False
        print(3)

        time.sleep(0.1)
        self.mCurrentTime = time.time()
        self.mAutoFishRunning = True
        while self.mAutoFishRunning is True:
            t1 = time.time()
            time.sleep(2)
            self.mFishingNum += 1
            print("____________________________")
            print('Lượt câu thứ ', self.mFishingNum)
            print("Số vật phẩm câu được = ", self.mFishNum)

            self.mSignalUpdateFishingNum.emit(self.mFishingNum)

            self.CastFishingRod()
            mOutPutCheckRod = self.CheckRod()
            if mOutPutCheckRod is True:
                mCheckMarkRgb = self.CheckMark()
                if mCheckMarkRgb is True:
                    mPullingRod = self.PullFishingRod()
                    if mPullingRod is True:
                        self.FishPreservation()
                        t2 = time.time()
                        print("Thời gian câu = ", int(t2 - t1), " giây")

            else:
                print("\nKhông thể thả cần câu lần ", self.mRodFixedCheck)
                print("Sau 3 lần không thể thả cần câu sẽ tiến hành lấy lại cần câu trong ba lô")
                if self.mRodFixedCheck == 3:
                    time.sleep(1)
                    self.OpenBackPack()
                    time.sleep(3)
                    self.OpenTools()
                    time.sleep(1)
                    self.TakeFishingRod()
                if self.mRodFixedCheck > 3:
                    self.MsgBox('Lỗi không tìm được nút thả cần. Kiểm tra lại giả lập game')
                    self.mRodFixedCheck = 0
            self.mSignalUpdateFishNum.emit(self.mFishNum)
            gc.collect()
