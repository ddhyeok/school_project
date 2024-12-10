from flask import Flask, request, render_template, redirect, url_for

import numpy as np
import pandas as pd

from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import cv2
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Grad_CAM, CAE_v2, Classifier_v2



device = torch.device("cpu")

                
fin_model = torch.load('model_save/CNN_based_on_CAE_v2.pth')
fin_model = fin_model.to(device)
fin_model.eval()







app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():    
    return render_template('index.html')

@app.route("/process", methods=['POST'])
def process():

    if 'image' not in request.files:
        return "No image file uploaded", 400

    
    try:
        file = request.files['image']
        img = Image.open(file.stream)
        img = img.convert("RGB")
        img.thumbnail((384,192), Image.LANCZOS)
        padded_img = ImageOps.pad(img, (384,192), color=(0, 0, 0)) 
        
        padded_img.save('static/input.png')


        padded_img = np.array(padded_img, dtype=np.float32)  
        padded_img = torch.tensor(padded_img).type(torch.FloatTensor)    
        padded_img=padded_img.permute(2, 0, 1).unsqueeze(0)#.transpose(2,3)
        padded_img=padded_img/255
        
        mean=[0.7101, 0.4827, 0.3970]#val mean
        std=[0.2351, 0.2195, 0.1862]#val std
        tmp=transforms.Normalize(mean=mean, std=std)
        padded_img=tmp(padded_img)
        

        padded_img=padded_img.to(device)
    except:
        return "preprocessing error", 100



    try:
        grad_cam = Grad_CAM.GradCam(fin_model)
        cam, pred = grad_cam(padded_img)
    except:
        return "Inference error", 200




    try:
        cam=np.uint8(255*cv2.resize(cam.cpu().detach().numpy(),None, fx=32,fy=32, interpolation=cv2.INTER_LINEAR))

        cam=255-cam
        
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        
        f=plt.figure(figsize=(10,5))
        
        plt.axis('off') 
        plt.imshow((padded_img.permute(0,2,3,1).squeeze(0).detach().numpy()*std)+mean)
        plt.imshow(heatmap,  alpha=0.4) 

        plt.savefig('static/result.png', bbox_inches='tight', pad_inches=0)
        plt.close(f)
    
        value = pred[0][1].item()*100
        if value>=100:
            value=99.99
        elif value<=0 :
            value=0.01
    except:
        return "visualization Error", 300
        

    
    return render_template('results.html', RESULT_PRED=float(value))


if __name__ == '__main__':
     app.run(debug=True)
