from tkinter import W
from token import LEFTSHIFT
from turtle import width
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from flask.ctx import F
from werkzeug.utils import secure_filename
from flask import Flask, render_template, Response
from ultralytics import YOLO
import torch
import supervision as sv
from roboflow import Roboflow, load_model
import json
import numpy as np
import os
import cv2
import tempfile
from collections import Counter
import omegaconf

model = YOLO("wytrenowany2.pt")

box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'mp4', 'avi'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

listaWykrytych = []

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    output_filename = None
    generated = False
    filtered = []
    if request.method == 'POST':
        file = request.files['file']
        conf = float(request.form['mySlider1'])
        iou = float(request.form['mySlider2'])
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            output_filename,generated,filtered = process_file(os.path.join(UPLOAD_FOLDER, filename),conf,iou)
    return render_template('index.html', output_filename=output_filename,filtered=filtered)

@app.route('/download')
def download_file():
    response = send_from_directory(UPLOAD_FOLDER, 'output.mp4', as_attachment=True)
    response.headers.add('Refresh', '1; URL=/' )
    return response

def process_file(filepath,conf,iou):
    generated = False
    pojedyncza = []
    cap = cv2.VideoCapture(filepath)
    
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(UPLOAD_FOLDER, 'output.mp4'), fourcc, 20.0, (width, height),True)
    i = 1
    while(cap.isOpened()):
        ret, frame = cap.read()
        print(f" {i} | {conf} | {iou}")
        print("______________________")
        if ret:
            results = model.predict(frame,imgsz=640,conf=conf,iou=iou)
        
            detections = sv.Detections.from_ultralytics(results[0])
            
            labels = [model.names[class_id] for _,_,_,class_id,_ in detections]
            
            [listaWykrytych.append(x) for x in labels]
            
            annotated_image = box_annotator.annotate(scene=frame, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

            out.write(annotated_image)
        else:
            break
        i+=1
    
    amount = Counter(listaWykrytych)
    print(amount)
    total = sum(amount.values())
    procenty = {element: ilosc/total*100 for element, ilosc in amount.items()}
    print(procenty)
    filtered = [element for element, procent in procenty.items() if procent>=1.5]
    
    generated = True
    print(filtered)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return 'output.mp4',generated,filtered

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True,threaded=True)
