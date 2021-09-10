# Requirement lib:
    Pillow
    PyAutoGUI
    keyboard
    opencv-python
    pure-python-adb
    pyinstaller

    SDK Platform-Tools Android for adb-server
    https://developer.android.com/studio/releases/platform-tools

# Build app to file .exe for windows:
pyinstaller main.py --onefile --noconsole --icon="data/iconapp.ico"