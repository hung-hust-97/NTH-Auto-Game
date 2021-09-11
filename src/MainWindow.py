import time
import threading

from PyQt5.QtWidgets import QMainWindow, QAction, QMessageBox, QWidget
from PyQt5.QtCore import QObject, Qt, QTimer
from PyQt5 import QtGui

from ui.UiMainWindow import Ui_MainWindow
from src.config import Config
from src.AutoFishing import AutoFishing


class MainWindow(QObject):
    def __init__(self):
        QObject.__init__(self, parent=None)
        self.main_win = QMainWindow()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self.main_win)

        self.mConfig = Config()
        self.mAutoFishing = AutoFishing()

        self.mTimer = QTimer()

        self.OpenApp()

    def __del__(self):
        self.mTimer.destroyed()

    def Show(self):
        self.main_win.show()

    def OpenApp(self):
        # Hien thi cac du lieu da luu trong config.ini
        self.uic.listEmulator.addItems(self.mConfig.GetWindowNames())

        self.uic.txtFishingRodPosition.setText(str(self.mConfig.GetFishingRod()))
        self.uic.txtFishingRodPosition.setAlignment(Qt.AlignCenter)

        self.uic.txtPullingFishTime.setText(str(self.mConfig.GetPullingFishTime()))
        self.uic.txtPullingFishTime.setAlignment(Qt.AlignCenter)

        self.uic.txtWaitingFishTime.setText(str(self.mConfig.GetWaitingFishTime()))
        self.uic.txtWaitingFishTime.setAlignment(Qt.AlignCenter)

        self.uic.txtMinFishSize.setText(str(self.mConfig.GetFishSize()))
        self.uic.txtMinFishSize.setAlignment(Qt.AlignCenter)

        self.uic.cbFreeMouse.setChecked(self.mConfig.GetFreeMouse())
        self.uic.cbFishDetection.setChecked(self.mConfig.GetFishDetection())
        self.uic.cbShowFish.setChecked(self.mConfig.GetShowFishShadow())

        self.uic.lblShowFish.setPixmap(
            QtGui.QPixmap(f'{self.mConfig.GetDataPath()}iconapp.ico').scaled(200,200))

        # Connect btn
        self.uic.btnConnectEmulator.clicked.connect(self.OnClickConnectEmulator)
        self.uic.btnStartFishing.clicked.connect(self.OnClickStart)
        self.uic.btnStopFishing.clicked.connect(self.OnClickStop)
        self.uic.btnGetMarkPosition.clicked.connect(self.OnClickGetMarkPosition)
        self.uic.btnGetBobberPosition.clicked.connect(self.OnClickGetBobberPosition)

        # Connect from auto fishing class to def in this class
        self.mAutoFishing.mSignalSetPixelPos.connect(self.SlotShowMarkPosition)
        self.mAutoFishing.mSignalSetFishingBobberPos.connect(self.SlotShowBobberPosition)
        self.mAutoFishing.mSignalUpdateFishingNum.connect(self.SlotShowFishingNum)
        self.mAutoFishing.mSignalUpdateFishNum.connect(self.SlotShowNumFish)
        self.mAutoFishing.mSignalUpdateFishDetectionImage.connect(self.SlotShowFishImage)
        self.mTimer.timeout.connect(self.SlotShowTime)

        # Hide btnStopFishing
        self.uic.btnStopFishing.hide()

    def OnClickConnectEmulator(self):
        self.mConfig.SetWindowName(self.uic.listEmulator.currentText())
        self.mAutoFishing.AdbConnect()
        self.mAutoFishing.CheckRegionEmulator()

    def OnClickStart(self):
        self.uic.btnConnectEmulator.hide()
        self.uic.btnStartFishing.hide()
        self.uic.btnStopFishing.show()
        self.mAutoFishing.mCheckMouseRunning = False
        self.mAutoFishing.mAutoFishRunning = False
        self.SaveConfig()
        time.sleep(0.1)

        self.mAutoFishing.mFishingNum = 0
        self.mAutoFishing.mFishNum = 0

        threading.Thread(target=self.mAutoFishing.StartAuto).start()

        self.mTimer.start(300)

    def OnClickStop(self):
        self.uic.btnConnectEmulator.show()
        self.uic.btnStopFishing.hide()
        self.uic.btnStartFishing.show()
        self.mAutoFishing.mCheckMouseRunning = False
        self.mAutoFishing.mAutoFishRunning = False

        self.mAutoFishing.mFishingNum = 0
        self.mAutoFishing.mFishNum = 0

        self.mAutoFishing.mSignalUpdateFishingNum.emit(0)
        self.mAutoFishing.mSignalUpdateFishNum.emit(0)

        self.mTimer.stop()

        self.SlotShowTime(True)

    def OnClickGetMarkPosition(self):
        self.mAutoFishing.mCheckMouseRunning = False
        time.sleep(0.1)
        threading.Thread(target=self.mAutoFishing.SetPixelPos).start()

    def SlotShowMarkPosition(self, x: int, y: int):
        self.uic.lcdMarkX.display(str(x))
        self.uic.lcdMarkX.setSegmentStyle(2)
        self.uic.lcdMarkY.display(y)
        self.uic.lcdMarkY.setSegmentStyle(2)

    def SlotShowFishingNum(self, x: int):
        self.uic.lcdNumFishing.display(str(x))
        self.uic.lcdNumFishing.setSegmentStyle(2)

    def SlotShowNumFish(self, x: int):
        self.uic.lcdNumFish.display(str(x))
        self.uic.lcdNumFish.setSegmentStyle(2)

    def OnClickGetBobberPosition(self):
        self.mAutoFishing.mCheckMouseRunning = False
        time.sleep(0.1)
        threading.Thread(target=self.mAutoFishing.SetFishingBobberPos).start()

    def SlotShowBobberPosition(self, x: int, y: int):
        self.uic.lcdRodX.display(str(x))
        self.uic.lcdRodX.setSegmentStyle(2)
        self.uic.lcdRodY.display(y)
        self.uic.lcdRodY.setSegmentStyle(2)

    def SlotShowFishImage(self):
        mMatImage = self.mAutoFishing.mFishImage
        mQImage = QtGui.QImage(mMatImage.data,
                               mMatImage.shape[1],
                               mMatImage.shape[0],
                               QtGui.QImage.Format_Grayscale8)
        mQPixmap = QtGui.QPixmap.fromImage(mQImage).scaled(200, 200)
        self.uic.lblShowFish.setPixmap(mQPixmap)

    def SlotShowTime(self, mReset=False):
        if mReset is True:
            self.uic.lcdTime.display('0')
        else:
            mTime = int(time.time() - self.mAutoFishing.mCurrentTime)
            self.uic.lcdTime.display(str(mTime))
        self.uic.lcdTime.setSegmentStyle(2)

    def SaveConfig(self):
        self.mConfig.SetFishingRod(int(self.uic.txtFishingRodPosition.toPlainText()))
        self.mConfig.SetPullingFishTime(int(self.uic.txtPullingFishTime.toPlainText()))
        self.mConfig.SetWaitingFishTime(int(self.uic.txtWaitingFishTime.toPlainText()))
        self.mConfig.SetFishSize(int(self.uic.txtMinFishSize.toPlainText()))
        self.mConfig.SetFreeMouse(self.uic.cbFreeMouse.isChecked())
        self.mConfig.SetFishDetection(self.uic.cbFishDetection.isChecked())
        self.mConfig.SetShowFishShadow(self.uic.cbShowFish.isChecked())
        self.mConfig.SaveConfig()
