from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS, cross_origin
from utils import decodeImage
from model_fitting import trained_model

app = Flask(__name__)


# class ClientApp:
#     def __init__(self):
#         self.filename = "person-wearing-a-mask.jpg"
#         self.classifier = model(image_file=self.filename).trained_model()


@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predictRoute():
    image = request.json["image"]
    filename = "new image.jpg"
    decodeImage(image, filename)
    result = trained_model(filename)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
