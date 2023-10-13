import pickle
import json
import numpy as np

def predictcar(m,HOG):
    result = m.predict(np.array(HOG).reshape(1,-1)) # ทำการ predict เพื่อทำนายยี่ห้อรถยนต์
    return result[0]
