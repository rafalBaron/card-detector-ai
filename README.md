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

Install Roboflow and load dataset (https://universe.roboflow.com/augmented-startups/playing-cards-ow27d/dataset/4)
``` python
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="GENERATED_API_KEY_FROM_ROBOFLOW_ACCOUNT")
project = rf.workspace("rafalbaron").project("cardsdetection-ekd3m")
dataset = project.version(4).download("yolov8")
```

<h2>Train</h2>

Train your chosen YOLOv8 model with loaded dataset and desired parameters. Make sure that paths in <b>data.yaml</b> leads to your dataset train, valid and test images!
``` python
!yolo task=detect mode=train model=yolov8n.pt data={dataset.location}/data.yaml epochs=30 imgsz=640
```

<h2>Validate</h2>

Validate using best weights
``` python
!yolo task=detect mode=val model=/content/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml
```

<h2>Predict</h2>

Test how model predicts your images
``` python
!yolo task=detect mode=predict model=/content/runs/detect/train/weights/best.pt conf=0.6 source=/path/to/your/image
```

<h2>Generated output video</h2>
![Output video](https://github.com/rafalBaron/card-detector-ai/output.gif)
