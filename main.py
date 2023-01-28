from flask import Flask,render_template,request,Response
from flask_sqlalchemy import SQLAlchemy
import base64
import urllib
import numpy as np
# from pyngrok import ngrok
from sklearn.neighbors import KNeighborsClassifier
import os
import cv2
import joblib
import json
import face_recognition
from json import JSONEncoder
import numpy
app = Flask(__name__, static_folder="static")
app.config['SQLALCHEMY_DATABASE_URI']="sqlite:///images.db"
# ngrok.set_auth_token("2Kp1iJytvsNhjKisg68c71vFMGx_F7pDwLcMT1PVJnRnjKzp")
# public_url=ngrok.connect(5000).public_url
# print(public_url)
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
known_face_encodings = []
known_face_names = []




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
            # return frame[y:y+h,x:x+w]
            return frame
        return ""


def train_model(link):
    frame = face_recognition.load_image_file(link)
    face_encoding = face_recognition.face_encodings(frame)[0]
    return face_encoding




def identify_face(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    name = "Unknown"
    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)


        # # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

    return name


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
      global known_face_encodings, known_face_names
      img_data=request.form.get("photo_data")
      img_data=img_data.split("*")

      newusername = request.form.get('username')
      roll = request.form.get('roll')
      user=User(name=newusername,roll=roll)
      db.session.add(user)
      db.session.commit()
      userid = User.query.filter_by(name=newusername).first().id


      img_blob=json.dumps(img_data)
      img_rec = Img(id=userid,name=newusername,img_blob=img_blob)
      db.session.add(img_rec)
      db.session.commit()

      img_blobs=Img.query.all()
      for imgs in img_blobs:
          img_data=json.loads(imgs.img_blob)
          for frame in img_data:
            frame_2 = gen_image_from_base64(frame)
            frame_3=gen_frames(frame_2)
            if frame_3!="":
                userimagefolder = 'static/faces/' + imgs.name + '_' + str(imgs.id)
                if not os.path.isdir(userimagefolder):
                    os.makedirs(userimagefolder)
                    name=imgs.name+'_'+str(imgs.id)+'.jpg'
                    cv2.imwrite(userimagefolder + '/' + name, frame_3)
                break
      userlist = os.listdir('static/faces')
      for user in userlist:
          for imgname in os.listdir(f'static/faces/{user}'):
              known_face_encodings.append(train_model(f'static/faces/{user}/{imgname}'))
              known_face_names.append(user)














      display=""


      return render_template("home.html", notice="registered")
    return f"<p></p>"



@app.route("/modeltest",methods=["GET","POST"])
def modelTesting():
    if request.method=="POST":
      print(known_face_names)
      img_data=request.form.get("photo_data")
      img_data=img_data.split("*")
      prediction=""
      for frame in img_data:
            frame_2 = gen_image_from_base64(frame)
            frame_3 = gen_frames(frame_2)
            if (frame_3 != ""):
                prediction=identify_face(frame_3)
                return render_template("home.html",notice=prediction)

    if request.method == "GET":
        return render_template("guess_face.html")















    return render_template("home.html", notice="registered")


if __name__=="__main__":
    app.run(port=5000)





