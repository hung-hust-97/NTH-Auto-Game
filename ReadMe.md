# Requirement lib:
	PyQt5==5.15.6
	opencv-python==4.5.4.60
	PyAutoGUI==0.9.53
	pure-python-adb==0.3.0.dev0
	pywin32==302
	Pillow==8.4.0
	psutil==5.8
	urllib3==1.26.7
	tensorflow==2.7.0
    pyinstaller==4.7.0

    SDK Platform-Tools Android for adb-server
    https://developer.android.com/studio/releases/platform-tools

# Build app to file .exe for windows:
pyinstaller main.py --onefile --noconsole --icon="data/AutoFishing.ico" --name=AutoFishing

# Branch 2.x.x license NTH Auto Game 
remove captcha