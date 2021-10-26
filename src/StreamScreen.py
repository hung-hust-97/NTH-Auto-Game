import cv2
import numpy as np
import win32gui
import win32ui
import win32con


class ScreenHandle:
    def __init__(self):
        self.hwnd = None
        self.w = 0
        self.h = 0
        self.x = 0
        self.y = 0

    def SetWindowName(self, window_name):
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception('Window not found: {}'.format(window_name))

        # get the window size
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]
        self.x = window_rect[0]
        self.y = window_rect[1]

    def WindowScreenShot(self):
        # get the window image data
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (0, 0), win32con.SRCCOPY)

        # convert the raw data into a format opencv can read
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())
        img = img[..., :3]
        img = np.ascontiguousarray(img)
        return img

    def Stream(self):
        while True:
            cv2.imshow("nth", self.WindowScreenShot())
            cv2.waitKey(1)

    @staticmethod
    def FindImage(template_gray, img_gray):
        check = False
        # Store width and height of template in w and h
        w, h = template_gray.shape[::-1]
        # Perform match operations.
        res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        # Specify a threshold
        threshold = 0.7
        # Store the coordinates of matched area in a numpy array
        loc = np.where(res >= threshold)
        # Draw a rectangle around the matched region.
        for pt in zip(*loc[::-1]):
            # cv2.rectangle(img_gray, pt, (pt[0] + w, pt[1] + h), (255, 255, 255), 1)
            check = True
            break
        return check
