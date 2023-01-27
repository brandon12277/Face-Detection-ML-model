from flask import Flask,render_template,request,Response
from flask_sqlalchemy import SQLAlchemy
import base64
import urllib
import numpy as np
from pyngrok import ngrok
from sklearn.neighbors import KNeighborsClassifier
import os
import cv2
import joblib
import json
from json import JSONEncoder
import numpy
app = Flask(__name__, static_folder="static")
app.config['SQLALCHEMY_DATABASE_URI']="sqlite:///images.db"
ngrok.set_auth_token("2Kp1iJytvsNhjKisg68c71vFMGx_F7pDwLcMT1PVJnRnjKzp")
public_url=ngrok.connect(5000).public_url
print(public_url)
db = SQLAlchemy(app)



class Img(db.Model):
    id = db.Column(db.Integer(),primary_key=True)
    name = db.Column(db.String(length=30), unique=True)
    img_blob = db.Column(db.String())


class User(db.Model):
    id = db.Column(db.Integer(),primary_key=True)
    name = db.Column(db.String(length=30),unique=True)
    roll= db.Column(db.Integer(),unique=True)


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")





def gen_image_from_base64(url):
    with urllib.request.urlopen(url) as resp:
        # read image as an numpy array
        image = np.asarray(bytearray(resp.read()), dtype="uint8")

        # use imdecode function
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image

def gen_frames(frame):
        faces = faceCascade.detectMultiScale(frame, 1.1, 4)
        for (x, y, w, h) in faces:
            return frame[y:y + h, x:x + w]
        return ""


def train_model(img_blob_data):
    faces = []
    labels = []
    for user, data in img_blob_data.items():
        for img in data:
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)




@app.route("/",methods=["GET","POST"])
def home_page():

    return render_template("home.html",notice="")



@app.route("/videofeed",methods=["GET","POST"])
def video():
    if request.method == "POST":
        newusername = request.form.get('username')
        roll = request.form.get('roll')
        return render_template("video.html",username=newusername,roll=roll)


@app.route("/results",methods=["GET","POST"])
def result():
    if request.method=="POST":
      img_data=request.form.get("photo_data")
      img_data=img_data.split("*")
      img_frame_data = []

      newusername = request.form.get('username')
      roll = request.form.get('roll')
      user=User(name=newusername,roll=roll)
      db.session.add(user)
      db.session.commit()
      userid = User.query.filter_by(name=newusername).first().id

      for frame in img_data:
         img_frame_data.append(frame)

      img_blob=json.dumps(img_frame_data)
      img_rec = Img(id=userid,name=newusername,img_blob=img_blob)
      db.session.add(img_rec)
      db.session.commit()

      img_blobs=Img.query.all()
      img_blob_data={}
      for imgs in img_blobs:
          img_data=json.loads(imgs.img_blob)
          img_blob_data[imgs.name]=[]
          for frame in img_data:
            frame_2 = gen_image_from_base64(frame)
            frame_3 = gen_frames(frame_2)
            if (frame_3 != ""):
                img_blob_data[imgs.name].append(frame_3)



      display=""

      train_model(img_blob_data)
      return render_template("home.html", notice="registered")



@app.route("/modeltest",methods=["GET","POST"])
def modelTesting():
    if request.method=="POST":
      img_data=request.form.get("photo_data")
      img_data=img_data.split("*")
      prediction=""
      for frame in img_data:
            frame_2 = gen_image_from_base64(frame)
            frame_3 = gen_frames(frame_2)
            if (frame_3 != ""):
                face = cv2.resize(frame_3, (50, 50))
                prediction=identify_face(face.reshape(1,-1))
                # data=User.query.filter_by(name=prediction).first()
                # display="USER : "+data.name+str(data.roll)
                return render_template("home.html",notice=prediction)

    if request.method == "GET":
        return render_template("guess_face.html")















    return render_template("home.html", notice="registered")


if __name__=="__main__":
    app.run(port=5000)





