# Requirement lib:
    Pillow
    PyAutoGUI
    opencv-python
    pure-python-adb
    pyinstaller
    PyQt5
    pywin32

    SDK Platform-Tools Android for adb-server
    https://developer.android.com/studio/releases/platform-tools

# Build app to file .exe for windows:
pyinstaller main.py --onefile --noconsole --icon="data/iconapp.ico"