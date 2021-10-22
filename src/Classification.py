import cv2
import numpy as np
from src.mobilenet_v2 import MobileNetv2
import operator


class Classification:
    def __init__(self):
        self.mLabelList = ['Ao', 'Balo', 'CanCau', 'Cho', 'Ga', 'Giay', 'Heo', 'Khac', 'Kinh', 'Meo', 'Non', 'Quan',
                           'Tho', 'Toc', 'Trung', 'Xe']
        self.model = MobileNetv2((224, 224, 3), len(self.mLabelList))
        self.model.load_weights("model/weights.h5")

    def Run(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        x_img = np.expand_dims(img, axis=0)
        x_img = x_img / 255.0
        list_score = self.model.predict(x_img)[0].copy()
        dict_label_score = {}

        for i in range(len(self.mLabelList)):
            dict_label_score[self.mLabelList[i]] = list_score[i]

        dict_label_score_sorted = sorted(dict_label_score.items(), key=operator.itemgetter(1))

        best_result = dict_label_score_sorted[-1]
        result_label = best_result[0]
        result_confident = int(best_result[1] * 100)
        return result_label, result_confident
