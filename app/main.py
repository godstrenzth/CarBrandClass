
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pickle
from pydantic import BaseModel
import numpy as np
import requests
import os
import aiohttp
from app.code import predictcar

app = FastAPI()

@app.get("/")
def read_root():
    return {"Class Head Car"}


m = pickle.load(open(r'..\model\img_model.pkl', 'rb'))# ..ย้อนไปก่อนทำการโหลด model 
# gethog = 'http://localhost:8080/api/gethog/'
# docker to docker
gethog = 'http://172.17.0.2:80/api/gethog/'

@app.post("/api/carbrand")
async def genhog(request: Request):
    data = await request.json()
    async with aiohttp.ClientSession() as session:
        async with session.get(gethog, json=data) as response:
            hog_re = await response.json()

    reshead = predictcar(m,[hog_re['HOG Descriptor']])
    return {"Headcar":reshead}
    

