from ultralytics import YOLO

import os
from os import listdir
from os.path import isfile, join

# Load a model
#model = YOLO("basketball_origin.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
#model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # load a pretrained model (recommended for training)
#model = YOLO("basketball_origin.pt")  # load a pretrained model (recommended for training)

# Train the model
#results = model.train(data="datasets/basketball/data.yaml", device="cpu", time=3, classes=[0,3], save_period=3, name="basketball", resume=True)

cwd = os.getcwd()
files = [os.path.join(cwd + "\\tests", f) for f in os.listdir(cwd + "\\tests") if 
os.path.isfile(os.path.join(cwd + "\\tests", f))]


for jpg_file in files:
    results = model(jpg_file)  # predict on an image

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="result-detection.jpg")  # save to disk


