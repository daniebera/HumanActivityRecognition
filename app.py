from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import base64
import cv2
import os

from deepmodel.model import DeepModel
import config.config as cfg

app = Flask(__name__, static_folder='static')


@app.route('/', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        if request.files:
            data = request.files.get('video')
            data.save(os.path.join(cfg.UPLOADS_PATH, data.filename))
            predictor = DeepModel()
            response = predictor.get_prediction(os.path.join(cfg.UPLOADS_PATH, data.filename))
            return render_template("edmin/code/index.html", response=response, filename=data.filename)
    return render_template("edmin/code/index.html")


if __name__ == '__main__':
    app.run()
