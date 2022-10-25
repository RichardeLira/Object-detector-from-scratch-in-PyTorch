# import the necessary packages
from cProfile import label
from ObjectDetector import ObjectDetector
from Custom_tensor_dataset import CustomTensorDataset
from Config import Config 
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.optim import Adam
from torchvision.models import resnet50
from sklearn.model_selection import train_test_split
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import time
import cv2
import os

# ---------------------------------------------------------------- # 

# Initialize the listt of data (images), class labels, target bounding 
# box coordinates, and image paths

print("[INFO] loading dataset ....")

data = []
labels = []
bboxes = []
imagePaths = []

# loop over all CSV files in the annotation directory

for csvPath in paths.list_files(Config.ANNOTS_PATH, validExts=(".csv")):  
    # load the contents of the current CSV 
    rows = open(csvPath).read().strip().split("\n")

    for row in rows:
        row = row.split(",")
        (filename,startX, startY, endX, endY, label_) = row

        # derive the path to the input image, load the image (in OpenCV format), and grab its dimensions 
        imagePath = os.path.join(Config.IMAGE_PATH,label_, filename)
        image = cv2.imread(imagePath)
        (imageHeight, imageWidth) = image.shape[0:2]

        # Scale the bouding box coordinates relative to the spatial 
        # dimension of the input image 

        startX = float(startX) / imageWidth
        startY = float(startY) / imageHeight
        endX = float(endX) / imageWidth
        endY = float(endY) / imageHeight


        # load the image and preprocess it 
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224,224))

        # Update our list of data, class labels, bouding box, and 
        # image paths

        data.append(image)
        labels.append(label_)
        bboxes.append((startX, startY, endX, endY))
        imagePaths.append(imagePath)



