import sys
from PyQt5.QtWidgets import QApplication
import logging as log
from src.MainWindow import MainWindow
from src.config import Config
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():
    mConfig = Config()
    if mConfig.mDebugMode is True:
        log.basicConfig(filename=mConfig.mLogPath,
                        filemode='w',
                        format="[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s",
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=log.INFO)
    else:
        log.basicConfig(level=log.INFO)
        log.disable(log.INFO)

    app = QApplication(sys.argv)
    main_win = MainWindow()
    app.setWindowIcon(main_win.Base64ToQIcon(mConfig.mIcon))
    main_win.Show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
