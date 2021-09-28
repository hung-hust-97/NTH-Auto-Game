import time
import threading
import subprocess
import base64

import cv2
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.QtCore import Qt, QTimer, QUrl, QSize, QCoreApplication, QByteArray
from PyQt5 import QtGui

from ui.UiMainWindow import Ui_MainWindow
from src.config import *
from src.AutoFishing import AutoFishing


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self, parent=None)
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        self.mConfig = Config()
        self.mAutoFishing = AutoFishing()

        self.mLogo = self.Base64ToQImage(self.mConfig.mAppLogo)
        self.mAutoFishingThread = None
        self.mWaitStatus = "Auto đang đóng chu trình câu\nVui lòng đợi trong giây lát"
        self.mTimer = QTimer()
        self.mTimer.setObjectName("AutoFishingTimer")
        self.OpenApp()

    def __del__(self):
        self.mAutoFishing.mAutoFishRunning = False
        self.mAutoFishing.mCheckMouseRunning = False

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Thông báo', "Chắc chắn thoát?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.mAutoFishing.mAutoFishRunning = False
            self.mAutoFishing.mCheckMouseRunning = False
            event.accept()
        else:
            event.ignore()

    def Show(self):
        self.show()

    def OpenApp(self):
        self.setWindowTitle(QCoreApplication.translate("MainWindow", self.mConfig.mAppTitle))

        # Hien thi cac du lieu da luu trong config.ini
        self.uic.txtEmulatorName.setText(self.mConfig.mWindowName)
        self.uic.txtEmulatorName.setAlignment(Qt.AlignLeft)

        self.uic.txtFishingRodPosition.setText(str(self.mConfig.mFishingRodIndex))
        self.uic.txtFishingRodPosition.setAlignment(Qt.AlignCenter)
        # self.uic.txtFishingRodPosition.setStyleSheet(HIDE_TEXT_BOX_STYLE)

        self.uic.txtPullingFishTime.setText(str(self.mConfig.mPullingFishTime))
        self.uic.txtPullingFishTime.setAlignment(Qt.AlignCenter)
        # self.uic.txtPullingFishTime.setStyleSheet(HIDE_TEXT_BOX_STYLE)

        self.uic.txtWaitingFishTime.setText(str(self.mConfig.mWaitingFishTime))
        self.uic.txtWaitingFishTime.setAlignment(Qt.AlignCenter)
        # self.uic.txtWaitingFishTime.setStyleSheet(HIDE_TEXT_BOX_STYLE)

        self.uic.txtMinFishSize.setText(str(self.mConfig.mFishSize))
        self.uic.txtMinFishSize.setAlignment(Qt.AlignCenter)
        # self.uic.txtMinFishSize.setStyleSheet(HIDE_TEXT_BOX_STYLE)

        self.uic.txtShutdownTime.setText("0")
        self.uic.txtShutdownTime.setAlignment(Qt.AlignCenter)
        # self.uic.txtShutdownTime.setStyleSheet(HIDE_TEXT_BOX_STYLE)

        self.uic.txtDelayTime.setText(str(self.mConfig.mDelayTime))
        self.uic.txtDelayTime.setAlignment(Qt.AlignCenter)
        # self.uic.txtDelayTime.setStyleSheet(HIDE_TEXT_BOX_STYLE)

        self.SlotShowNumFish()
        self.ShowListEmulatorSize()
        self.uic.listAdbAddress.addItem(self.mConfig.mAdbAddress)

        self.uic.cbFreeMouse.setChecked(self.mConfig.mFreeMouseCheck)
        self.uic.cbFishDetection.setChecked(self.mConfig.mFishDetectionCheck)
        self.uic.cbShowFish.setChecked(self.mConfig.mShowFishCheck)
        self.uic.cbShutdownPC.setChecked(False)

        # Set button color
        # self.uic.btnConnectAdb.setStyleSheet(BUTTON_COLOR)
        # self.uic.btnPauseFishing.setStyleSheet(BUTTON_COLOR)
        # self.uic.btnConnectWindowTitle.setStyleSheet(BUTTON_COLOR)
        # self.uic.btnStartFishing.setStyleSheet(BUTTON_COLOR)
        # self.uic.btnGetBobberPosition.setStyleSheet(BUTTON_COLOR)
        # self.uic.btnGetMarkPosition.setStyleSheet(BUTTON_COLOR)

        # Show logo
        self.uic.lblShowFish.setPixmap(QtGui.QPixmap.fromImage(self.mLogo).scaled(200, 200))
        self.uic.btnYoutube.setIcon(QtGui.QIcon(self.mConfig.mYoutubeImgPath))
        self.uic.btnYoutube.setIconSize(QSize(40, 40))
        self.uic.btnYoutube.setFlat(True)
        self.uic.btnFacebook.setIcon(QtGui.QIcon(self.mConfig.mFacebookImgPath))
        self.uic.btnFacebook.setIconSize(QSize(40, 40))
        self.uic.btnFacebook.setFlat(True)

        # Show time
        self.uic.lcdTime.setNumDigits(8)
        self.uic.lcdTime.setSegmentStyle(2)
        self.uic.lcdTime.display('00:00:00')
        self.uic.lcdNumFish.setSegmentStyle(2)
        self.uic.lcdNumFishing.setSegmentStyle(2)
        self.uic.lcdRodX.setSegmentStyle(2)
        self.uic.lcdRodY.setSegmentStyle(2)
        self.uic.lcdMarkX.setSegmentStyle(2)
        self.uic.lcdMarkY.setSegmentStyle(2)

        # Connect btn
        self.uic.btnConnectWindowTitle.clicked.connect(self.OnClickConnectWindowTitle)
        self.uic.btnStartFishing.clicked.connect(self.OnClickStart)
        self.uic.btnPauseFishing.clicked.connect(self.OnClickPause)
        self.uic.btnGetMarkPosition.clicked.connect(self.OnClickGetMarkPosition)
        self.uic.btnGetBobberPosition.clicked.connect(self.OnClickGetBobberPosition)
        self.uic.btnConnectAdb.clicked.connect(self.OnClickConnectAdbAddress)
        self.uic.btnFacebook.clicked.connect(self.SlotOpenFacebook)
        self.uic.btnYoutube.clicked.connect(self.SlotOpenYoutube)

        # Connect from auto fishing class to def in this class
        self.mAutoFishing.mSignalSetPixelPos.connect(self.SlotShowMarkPosition)
        self.mAutoFishing.mSignalSetFishingBobberPos.connect(self.SlotShowBobberPosition)
        self.mAutoFishing.mSignalUpdateFishingNum.connect(self.SlotShowFishingNum)
        self.mAutoFishing.mSignalUpdateFishNum.connect(self.SlotShowNumFish)
        self.mAutoFishing.mSignalUpdateImageShow.connect(self.SlotShowImage)
        self.mAutoFishing.mSignalMessage.connect(self.SlotShowMsgBox)
        self.mAutoFishing.mSignalUpdateStatus.connect(self.SlotShowStatus)

        # Connect timer to slot
        self.mTimer.timeout.connect(self.SlotShowTime)
        self.mTimer.timeout.connect(self.SlotCheckThread)
        self.mTimer.timeout.connect(self.ShowShutdownPCTime)

        # Disable btnPauseFishing
        self.uic.btnPauseFishing.setDisabled(True)

        # Show Author
        self.SlotShowStatus(self.mConfig.mLicenseText)

        # Show status bar
        self.uic.statusbar.showMessage("Phần mềm miễn phí, không dùng cho mục đích thương mại")

    def OnClickConnectWindowTitle(self):
        self.mConfig.SetWindowName(self.uic.txtEmulatorName.toPlainText())
        self.mConfig.SetEmulatorSize(self.uic.listEmulatorSize.currentIndex())

        if self.mAutoFishing.CheckRegionEmulator() is True:
            self.SlotShowStatus("Kết nối cửa sổ giả lập thành công\n"
                                f"{self.mAutoFishing.mEmulatorBox}")
        else:
            self.uic.listAdbAddress.clear()
            self.uic.listAdbAddress.addItem("None")
            return

        self.mAutoFishing.AdbServerConnect()
        self.UpdateListAdbAddress()
        self.SaveConfig()
        self.SlotShowMsgBox("Kết nối cửa sổ giả lập thành công\nChọn địa chỉ ADB của giả lập và kết nối")
        return

    def OnClickConnectAdbAddress(self):
        if self.uic.listAdbAddress.currentText() == "None":
            self.SlotShowMsgBox("Xác nhận lại cửa sổ giả lập để tìm địa chỉ Adb")
            return False
        self.mConfig.SetAdbAddress(self.uic.listAdbAddress.currentText())
        if self.mAutoFishing.AdbDeviceConnect() is True:
            self.SlotShowStatus("Kết nối địa chỉ Adb giả lập thành công")
            self.SlotShowMsgBox("Kết nối địa chỉ Adb giả lập thành công")
            return True
        else:
            self.SlotShowMsgBox("Kết nối địa chỉ Adb giả lập thất bại\n Restart lại giả lập")
            return False

    def OnClickStart(self):
        # Apply and save all config
        if self.SaveConfig() is False:
            return

        # Hide button
        self.uic.btnConnectWindowTitle.setDisabled(True)
        self.uic.btnConnectAdb.setDisabled(True)
        self.uic.btnStartFishing.setDisabled(True)
        self.uic.btnGetMarkPosition.setDisabled(True)
        self.uic.btnGetBobberPosition.setDisabled(True)

        # Hide text box
        self.uic.txtPullingFishTime.setDisabled(True)
        self.uic.txtWaitingFishTime.setDisabled(True)
        self.uic.txtFishingRodPosition.setDisabled(True)
        self.uic.txtMinFishSize.setDisabled(True)
        self.uic.txtShutdownTime.setDisabled(True)
        self.uic.txtEmulatorName.setDisabled(True)
        self.uic.txtDelayTime.setDisabled(True)

        # Hide list box
        self.uic.listAdbAddress.setDisabled(True)
        self.uic.listEmulatorSize.setDisabled(True)

        # Hide check box
        self.uic.cbShowFish.setDisabled(True)
        self.uic.cbFreeMouse.setDisabled(True)
        self.uic.cbShutdownPC.setDisabled(True)
        self.uic.cbFishDetection.setDisabled(True)

        # All thread flag = False
        self.mAutoFishing.mCheckMouseRunning = False
        self.mAutoFishing.mAutoFishRunning = False

        # Set image on graphic label
        if self.mConfig.mShowFishCheck is False:
            self.uic.lblShowFish.setPixmap(QtGui.QPixmap.fromImage(self.mLogo).scaled(200, 200))

        # Define thread start fishing
        self.mAutoFishingThread = threading.Thread(name="AutoFishing", target=self.mAutoFishing.StartAuto)

        # Set time fishing
        self.mAutoFishing.mStartTime = time.time()

        self.mTimer.start(200)
        # Start thread auto fishing
        self.mAutoFishingThread.start()

        # Check thread is not live, return
        if self.mAutoFishingThread.is_alive() is False:
            return

        # Disable Pause button
        self.uic.btnPauseFishing.setDisabled(False)

    def OnClickPause(self):
        # Disable Pause button
        self.uic.btnPauseFishing.setDisabled(True)

        # Pause all thread flag
        self.mAutoFishing.mCheckMouseRunning = False
        self.mAutoFishing.mAutoFishRunning = False

        # Show status notice doing Pause
        self.SlotShowStatus("")

    def OnClickGetMarkPosition(self):
        self.mAutoFishing.mCheckMouseRunning = False
        threading.Thread(name="SetMarkPosition", target=self.mAutoFishing.SetPixelPos).start()

    def ShowListEmulatorSize(self):
        for mSize in self.mConfig.mListEmulatorSize:
            strSize = f'{mSize[0]}x{mSize[1]}'
            self.uic.listEmulatorSize.addItem(strSize)
        self.uic.listEmulatorSize.setCurrentIndex(self.mConfig.mEmulatorSizeId)

    def ShowShutdownPCTime(self):
        if self.mConfig.mShutdownCheckBox is False:
            return
        if self.mAutoFishing.mAutoFishRunning is False:
            return
        mCountDownTime = (self.mConfig.mShutdownTime * 60 - (time.time() - self.mAutoFishing.mStartTime)) / 60
        self.uic.txtShutdownTime.setText(str(int(mCountDownTime) + 1))
        self.uic.txtShutdownTime.setAlignment(Qt.AlignCenter)
        if mCountDownTime < 0:
            subprocess.call(["shutdown", "/s"], creationflags=0x08000000)

    def SlotShowMarkPosition(self, x: int, y: int):
        self.uic.lcdMarkX.display(str(x))
        self.uic.lcdMarkY.display(y)

    def SlotShowFishingNum(self):
        self.uic.lcdNumFishing.display(str(self.mAutoFishing.mFishingNum))

    def SlotShowNumFish(self):
        self.uic.lcdNumFish.display(str(self.mAutoFishing.mAllFish))

        # font = QtGui.QFont()
        # font.setPointSize(10)
        self.uic.txtVioletFish.setText(str(self.mAutoFishing.mVioletFish))
        self.uic.txtVioletFish.setAlignment(Qt.AlignCenter)
        self.uic.txtVioletFish.setDisabled(True)
        # self.uic.txtVioletFish.setFont(font)
        self.uic.txtVioletFish.setStyleSheet(
            f'border: 0px; background-color: rgba({self.mConfig.mVioletColorRGB[0]},'
            f' {self.mConfig.mVioletColorRGB[1]}, {self.mConfig.mVioletColorRGB[2]}, 255);')

        self.uic.txtBlueFish.setText(str(self.mAutoFishing.mBlueFish))
        self.uic.txtBlueFish.setAlignment(Qt.AlignCenter)
        self.uic.txtBlueFish.setDisabled(True)
        # self.uic.txtBlueFish.setFont(font)
        self.uic.txtBlueFish.setStyleSheet(
            f'border: 0px; background-color: rgba({self.mConfig.mBlueColorRGB[0]},'
            f' {self.mConfig.mBlueColorRGB[1]}, {self.mConfig.mBlueColorRGB[2]}, 255);')

        self.uic.txtGreenFish.setText(str(self.mAutoFishing.mGreenFish))
        self.uic.txtGreenFish.setAlignment(Qt.AlignCenter)
        self.uic.txtGreenFish.setDisabled(True)
        # self.uic.txtGreenFish.setFont(font)
        self.uic.txtGreenFish.setStyleSheet(
            f'border: 0px; background-color: rgba({self.mConfig.mGreenColorRGB[0]},'
            f' {self.mConfig.mGreenColorRGB[1]}, {self.mConfig.mGreenColorRGB[2]}, 255);')

        self.uic.txtGrayFish.setText(str(self.mAutoFishing.mGrayFish))
        self.uic.txtGrayFish.setAlignment(Qt.AlignCenter)
        self.uic.txtGrayFish.setDisabled(True)
        # self.uic.txtGrayFish.setFont(font)
        self.uic.txtGrayFish.setStyleSheet(
            f'border: 0px; background-color: rgba({self.mConfig.mGrayColorRGB[0]},'
            f' {self.mConfig.mGrayColorRGB[1]}, {self.mConfig.mGrayColorRGB[2]}, 255);')

    def OnClickGetBobberPosition(self):
        self.mAutoFishing.mCheckMouseRunning = False
        time.sleep(0.1)
        threading.Thread(name="SetBobberPosition", target=self.mAutoFishing.SetFishingBobberPos).start()

    def SlotShowBobberPosition(self, x: int, y: int):
        self.uic.lcdRodX.display(str(x))
        self.uic.lcdRodY.display(y)

    def SlotShowImage(self):
        mMatImage = self.mAutoFishing.mImageShow.copy()
        mMatImage = cv2.resize(mMatImage, (200, 200), interpolation=cv2.INTER_AREA)
        mQImage = QtGui.QImage(mMatImage.data,
                               mMatImage.shape[1],
                               mMatImage.shape[0],
                               QtGui.QImage.Format_RGB888).rgbSwapped()
        mQPixmap = QtGui.QPixmap.fromImage(mQImage).scaled(200, 200)
        self.uic.lblShowFish.setPixmap(mQPixmap)

    def SlotShowTime(self):
        mShowTime = self.mAutoFishing.mSaveTime + int(time.time() - self.mAutoFishing.mStartTime)
        h = mShowTime // 3600
        m = (mShowTime - h * 3600) // 60
        s = ((mShowTime - h * 3600) - m * 60)
        str_h = str(h)
        str_m = str(m)
        str_s = str(s)
        if h < 10:
            str_h = f'0{h}'
        if m < 10:
            str_m = f'0{m}'
        if s < 10:
            str_s = f'0{s}'
        self.uic.lcdTime.display(f'{str_h}:{str_m}:{str_s}')

    def SlotShowStatus(self, mText: str):
        if mText == self.mConfig.mLicenseText:
            self.uic.lblStatus.setText(self.mConfig.mLicenseText)
        elif self.uic.btnStartFishing.isEnabled() is False and self.uic.btnPauseFishing.isEnabled() is False:
            self.uic.lblStatus.setText(self.mWaitStatus)
        else:
            self.uic.lblStatus.setText(mText)
        self.uic.lblStatus.setAlignment(Qt.AlignLeft)
        self.uic.lblStatus.setAlignment(Qt.AlignVCenter)
        self.uic.lblStatus.setWordWrap(True)

    # Watchdog thread auto fishing
    def SlotCheckThread(self):
        if self.mAutoFishingThread.is_alive() is False:
            # Disable thread flag
            self.mAutoFishing.mAutoFishRunning = False

            # Show all button
            self.uic.btnConnectWindowTitle.setDisabled(False)
            self.uic.btnConnectAdb.setDisabled(False)
            self.uic.btnGetMarkPosition.setDisabled(False)
            self.uic.btnGetBobberPosition.setDisabled(False)

            # Show all check box
            self.uic.cbShowFish.setDisabled(False)
            self.uic.cbFreeMouse.setDisabled(False)
            self.uic.cbShutdownPC.setDisabled(False)
            self.uic.cbFishDetection.setDisabled(False)

            # Show all text box
            self.uic.txtPullingFishTime.setDisabled(False)
            self.uic.txtWaitingFishTime.setDisabled(False)
            self.uic.txtFishingRodPosition.setDisabled(False)
            self.uic.txtMinFishSize.setDisabled(False)
            self.uic.txtShutdownTime.setDisabled(False)
            self.uic.txtEmulatorName.setDisabled(False)
            self.uic.txtDelayTime.setDisabled(False)

            self.uic.listAdbAddress.setDisabled(False)
            self.uic.listEmulatorSize.setDisabled(False)

            self.uic.txtShutdownTime.setText(str(self.mConfig.mShutdownTime))
            self.uic.txtShutdownTime.setAlignment(Qt.AlignCenter)
            self.uic.lblShowFish.setPixmap(QtGui.QPixmap.fromImage(self.mLogo).scaled(200, 200))

            self.uic.btnPauseFishing.setDisabled(True)
            self.uic.btnStartFishing.setDisabled(False)

            self.uic.lblStatus.setText(self.mConfig.mLicenseText)

            self.mTimer.stop()
            self.mAutoFishing.mSaveTime += int(time.time() - self.mAutoFishing.mStartTime)

    def SlotOpenFacebook(self):
        QtGui.QDesktopServices.openUrl(QUrl(self.mConfig.mFacebookLink))

    def SlotOpenYoutube(self):
        QtGui.QDesktopServices.openUrl(QUrl(self.mConfig.mYoutubeLink))

    def SaveConfig(self):
        if (self.uic.txtPullingFishTime.toPlainText()).isnumeric() is False:
            self.SlotShowMsgBox("Thời gian kéo cá sai định dạng")
            return False

        if (self.uic.txtWaitingFishTime.toPlainText()).isnumeric() is False:
            self.SlotShowMsgBox("Thời gian chờ cá sai định dạng")
            return False

        if (self.uic.txtFishingRodPosition.toPlainText()).isnumeric() is False:
            self.SlotShowMsgBox("Thời gian chờ cá sai định dạng")
            return False

        if int(self.uic.txtFishingRodPosition.toPlainText()) not in range(1, 7, 1):
            self.SlotShowMsgBox("Vị trí cần câu phải từ 1 đến 6")
            return False

        if (self.uic.txtMinFishSize.toPlainText()).isnumeric() is False:
            self.SlotShowMsgBox("Lọc cả nhỏ sai định dạng")
            return False

        if self.uic.txtShutdownTime.toPlainText().isnumeric() is False:
            self.SlotShowMsgBox("Hẹn giờ tắt PC sai định dạng")
            return False

        try:
            mDelayTime = float(self.uic.txtDelayTime.toPlainText())
        except ValueError:
            self.SlotShowMsgBox("Độ trễ sửa cần sai định dạng")
            return False

        self.mConfig.SetWindowName(self.uic.txtEmulatorName.toPlainText())
        self.mConfig.SetShutdownTime(int(self.uic.txtShutdownTime.toPlainText()))
        self.mConfig.SetFishingRod(int(self.uic.txtFishingRodPosition.toPlainText()))
        self.mConfig.SetPullingFishTime(int(self.uic.txtPullingFishTime.toPlainText()))
        self.mConfig.SetWaitingFishTime(int(self.uic.txtWaitingFishTime.toPlainText()))
        self.mConfig.SetFishSize(int(self.uic.txtMinFishSize.toPlainText()))

        self.mConfig.SetShutdownCheckBox(self.uic.cbShutdownPC.isChecked())
        self.mConfig.SetFreeMouse(self.uic.cbFreeMouse.isChecked())
        self.mConfig.SetFishDetection(self.uic.cbFishDetection.isChecked())
        self.mConfig.SetShowFishShadow(self.uic.cbShowFish.isChecked())

        self.mConfig.SetDelayTime(mDelayTime)

        self.mConfig.SaveConfig()
        return True

    def UpdateListAdbAddress(self):
        self.uic.listAdbAddress.clear()
        if len(self.mAutoFishing.mListAdbDevicesSerial) == 0:
            self.uic.listAdbAddress.addItem("None")
        for AdbDevicesSerial in self.mAutoFishing.mListAdbDevicesSerial:
            self.uic.listAdbAddress.addItem(AdbDevicesSerial)

    @staticmethod
    def SlotShowMsgBox(mText: str):
        mMsgBox = QMessageBox()
        mMsgBox.setText(mText)
        mMsgBox.setWindowTitle("Thông báo")
        mMsgBox.setWindowFlags(Qt.WindowStaysOnTopHint)
        mMsgBox.exec()

    @staticmethod
    def Base64ToQImage(base64img: str):
        image_64_decode = base64.b64decode(base64img)
        image = QtGui.QImage()
        image.loadFromData(image_64_decode, 'PNG')
        return image

    @staticmethod
    def Base64ToQIcon(base64img: str):
        image_64_decode = base64.b64decode(base64img)
        image = QtGui.QImage()
        image.loadFromData(image_64_decode, 'PNG')
        icon = QtGui.QIcon(QtGui.QPixmap(image))
        return icon
