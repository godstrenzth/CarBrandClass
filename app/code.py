import pickle
import json

def predictcar(m,HOG):
    result = m.predict(HOG) # ทำการ predict เพื่อทำนายยี่ห้อรถยนต์
    return result[0]