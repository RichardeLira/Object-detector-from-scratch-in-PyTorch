from torch.nn import Dropout
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from torch.nn import Sigmoid
import torch


class objectDetector(torch.nn.Module): 
    def __init__(self, baseModel, numClasses):
        super(objectDetector,self).__init__()

        self.baseModel = baseModel
        self.numClasses = numClasses 

        self.regressor = Sequential ( 
            Linear(baseModel.fc.in_features, 128),
            ReLU(),
            Linear(128,64),
            ReLU(),
            Linear(64,32), 
            ReLU(), 
            Linear(32,4), 
            Sigmoid()

        )

        # build the classifier head to predict the class labels
        self.classifier = Sequential (
            Linear(baseModel.fc.in_features, 512),
			ReLU(),
			Dropout(),
			Linear(512, 512),
			ReLU(),
			Dropout(),
			Linear(512, self.numClasses)
        )

        self.baseModel = Identity()

    def forward(self, X): 
        # pass the inputs through the base model and then obtain
	    # predictions from two different branches of the network 

        features = self.baseModel(X)
        bboxes = self.regressor(features)
        classLogits = self.classifier(features)

        return (bboxes,classLogits)
