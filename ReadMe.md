# Requirement lib:
    python==3.9.7
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

# Build app to file .exe for windows:
pyinstaller main.py --noconsole --noconfirm --icon="data/nth_auto_game.ico" --name=AutoFishing
