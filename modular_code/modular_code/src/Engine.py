from ML_Pipeline.training import Classifier
from ML_Pipeline.inference import Inference
from ML_Pipeline.admin import train_dir


### training ###
train_object = Classifier(train_dir) #training model
train_object.train()


### inference ###
filename = "../input/Data/Testing_Data/driving_license/1.jpg" # read the test image
infer_object = Inference()
response = infer_object.infer(filename)
print("The result is : ", response)      # check the output
