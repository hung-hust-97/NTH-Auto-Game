import sys
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtGui
from src.MainWindow import MainWindow
from src.config import Config

if __name__ == "__main__":
    mConfig = Config()
    mIconPath = f'{mConfig.GetDataPath()}iconapp.ico'

    app = QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(mIconPath))
    main_win = MainWindow()
    main_win.Show()
    sys.exit(app.exec())
