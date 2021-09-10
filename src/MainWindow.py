import time

from PyQt5.QtWidgets import QMainWindow, QAction, QMessageBox
from PyQt5.QtCore import *

from ui.UiMainWindow import Ui_MainWindow
from src.config import Config
from src.AutoFishing import AutoFishing


class MainWindow:
    def __init__(self):
        self.main_win = QMainWindow()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self.main_win)

        self.mConfig = Config()
        self.mAutoFishing = AutoFishing()
        self.OpenApp()

    def __del__(self):
        pass

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

        # Connect btn
        self.uic.btnConnectEmulator.clicked.connect(self.OnClickConnectEmulator)
        self.uic.btnStartFishing.clicked.connect(self.OnClickStart)
        self.uic.btnStopFishing.clicked.connect(self.OnClickStop)
        self.uic.btnGetMarkPosition.clicked.connect(self.OnClickGetMarkPosition)
        self.uic.btnGetBobberPosition.clicked.connect(self.OnClickGetBobberPosition)

    def OnClickConnectEmulator(self):
        self.mConfig.SetWindowName(self.uic.listEmulator.currentText())
        self.mAutoFishing.AdbConnect()
        self.mAutoFishing.CheckRegionEmulator()

    def OnClickStart(self):
        self.SaveConfig()

    def OnClickStop(self):
        pass

    def OnClickGetMarkPosition(self):
        pass

    def OnClickGetBobberPosition(self):
        pass

    def SaveConfig(self):
        self.mConfig.SetFishingRod(int(self.uic.txtFishingRodPosition.toPlainText()))
        self.mConfig.SetPullingFishTime(int(self.uic.txtPullingFishTime.toPlainText()))
        self.mConfig.SetWaitingFishTime(int(self.uic.txtWaitingFishTime.toPlainText()))
        self.mConfig.SetFishSize(int(self.uic.txtMinFishSize.toPlainText()))
        self.mConfig.SetFreeMouse(self.uic.cbFreeMouse.isChecked())
        self.mConfig.SetFishDetection(self.uic.cbFishDetection.isChecked())
        self.mConfig.SetShowFishShadow(self.uic.cbShowFish.isChecked())

        self.mConfig.SaveConfig()

