<h1>YOLOv8 Cards Detection from video</h1>
<p>Simple website (not responsive) for cards detection and classification from uploaded video. Project made with Roboflow, ultralytics and YOLOv8. </p>
<h2>Load dataset and model</h2>

Install ultralytics and libraries
``` python
!pip install ultralytics
from ultralytics import YOLO
import os
from IPython.display import display, Image
from IPython import display
display.clear_output()
!yolo checks
```

Install Roboflow and load dataset
``` python
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="diizX4Cv14C2yoNXKnxj")
project = rf.workspace("rafalbaron").project("cardsdetection-ekd3m")
dataset = project.version(4).download("yolov8")
```

<h2>Train</h2>

Train your chosen YOLOv8 model with loaded dataset and desired parameters. Make sure that paths in <b>data.yaml</b> leads to your dataset train, valid and test images!
``` python
!yolo task=detect mode=train model=yolov8n.pt data={dataset.location}/data.yaml epochs=30 imgsz=640
```
<h2>Validate</h2>
<h2>Predict</h2>
