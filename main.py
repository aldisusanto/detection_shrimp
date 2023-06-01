from flask import Flask, request, jsonify
from keras.applications.efficientnet_v2 import EfficientNetV2M
import numpy as np
from PIL import Image
import cv2

app = Flask(__name__)

model = EfficientNetV2M()
model.summary()


@app.route("/", methods=['POST'])
def predict():
    file = request.files["image"]
    img = Image.open(file)
    img = img.resize((480, 480))
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = np.expand_dims(img, axis=0)

    yh = model.predict(img)

    if (np.argmax(yh) >= 122 and np.argmax(yh) <= 127):
        result = {
            "code" : "200",
            "message": "Udang"}
    else:
        result = {
            "code": "400",
            "message": "Bukan udang"}

    print(np.argmax(yh))
    return jsonify(result)

if __name__ == '__main__':
    app.run()
