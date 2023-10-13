import pickle
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
from app.code import predictcar

app = FastAPI() 

m = pickle.load(open(f'model/img_model.pkl', 'rb'))
#m = pickle.load(open(r'..\model\img_model.pkl', 'rb'))# ..ย้อนไปก่อนทำการโหลด model 
#gethog = 'http://localhost:5000/api/gethog/'
# docker to docker
gethog = 'http://172.17.0.3:80/api/gethog/'

@app.get("/")
def read_root():
    return {"Class Head Car"}

@app.post("/api/carbrand")
async def genhog(request: Request):
    data = await request.json()
    hog1 =requests.get(gethog,json=data)
    reshead = predictcar(m,hog1.json()['HOG Descriptor'])
    return {"Headcar":reshead}
    
    
