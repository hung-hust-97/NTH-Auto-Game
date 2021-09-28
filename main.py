import sys
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtGui
from src.MainWindow import MainWindow
from src.config import Config


def main():
    mConfig = Config()
    app = QApplication(sys.argv)
    main_win = MainWindow()
    app.setWindowIcon(main_win.Base64ToQIcon(mConfig.mIcon))
    main_win.Show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
