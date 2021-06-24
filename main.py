import os
import cv2
import urllib
import numpy as np

from werkzeug.utils import secure_filename
from urllib.request import Request, urlopen

from flask import Flask, render_template, Response, request, redirect, flash, url_for

from camera import VideoCamera
from Graphical_Visualisation import Emotion_Analysis

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
import cv2
from model import FacialExpressionModel
import numpy as np
from cv2 import cv2
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        ret,frame = self.video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            fc = gray[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            cv2.putText(frame, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.imshow('Video',frame)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
def gen(VideoCamera):
    while True:
        frame = VideoCamera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def allowed_file(filename):
    return ('.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

@app.route('/')
def Start():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/Live', methods=['POST'])
def Live():
    return render_template('Live.html')

@app.route('/takeimage', methods=['POST'])
def takeimage():
    v = VideoCamera()
    _, frame = v.video.read()
    save_to = "static/"
    cv2.imwrite(save_to + "capture" + ".jpg", frame)
    result = Emotion_Analysis("capture.jpg")
    if(len(result)==1):
        return render_template('Error.html', orig=result[0])
    return render_template('Result.html', orig=result[0], pred=result[1])

@app.route('/uploadmanually', methods=['POST'])
def uploadmaually():
    return render_template('uploadmanually.html')

@app.route('/uploadimage', methods=['POST'])
def uploadimage():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            result = Emotion_Analysis(filename)
            if(len(result)==1):
                return render_template('Error.html', orig=result[0])
            return render_template('Result.html', orig=result[0], pred=result[1])

@app.route('/imageurl', methods=['POST'])
def imageurl():
    url = request.form['url']
    req = Request(url,headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    arr = np.asarray(bytearray(webpage), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    save_to = "static/"
    cv2.imwrite(save_to + "url.jpg", img)
    result = Emotion_Analysis("url.jpg")
    if(len(result)==1):
        return render_template('Error.html', orig=result[0])
    return render_template('Result.html', orig=result[0], pred=result[1])

if __name__ == '__main__':
    app.run(debug=True)
