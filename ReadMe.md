# Requirement lib:
    Pillow
    PyAutoGUI
    opencv-python
    pure-python-adb
    pyinstaller
    PyQt5
    pywin32
    urllib3

    SDK Platform-Tools Android for adb-server
    https://developer.android.com/studio/releases/platform-tools

# Build app to file .exe for windows:
pyinstaller main.py --onefile --noconsole --icon="data/AutoFishing.ico" --name=AutoFishing

# Branch 2.x.x license NTH Auto Game 
remove captcha