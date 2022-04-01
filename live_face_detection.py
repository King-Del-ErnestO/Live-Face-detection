import cv2
import os

face_cascade = cv2.CascadeClassifier('D:\A.I\Python_files\haarcascade_frontalface_default.xml')
img = cv2.VideoCapture(0)
# img.set(3, 640)
# img.set(4, 480)
face_id = input("\n  enter user id end press <return> ==> ")
print("\n [INFO] Initialzing face capture. Look the camera and wait...")
count = 0
while True:
    
    ret, frame = img.read()
    frame = cv2.flip(frame, -1)
    # cv2.imshow("Capturing frame", frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Capturing", gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(20, 20))
    
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        count += 1
        cv2.imwrite("D:\A.I\Python_files\live_face_detection\dataset/User." + str(face_id) + '.' + str(count) + '.jpg', gray[y:y + h, x:x + w])

        # roi_gray = gray[y:y + h, x:x + w]
        # roi_color = frame[y:y + h, x:x + w]

    cv2.imshow('Video', frame)

    key = cv2.waitKey(30)

    if key == ord('q'):
        break
    elif count >= 30:
        break

print("\n [INFO] Exiting Program and cleanup stuff")
img.release()
cv2.destroyAllWindows()

# from flask import Flask, Response
# import cv2
# app = Flask(__name__)
# video = cv2.VideoCapture(0)@app.route('/')
# def index():
#     return "Default Message"
# def gen(video):
#     while True:
#         success, image = video.read()
#         ret, jpeg = cv2.imencode('.jpg', image)
#         frame = jpeg.tobytes()
#         yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#         @app.route('/video_feed')
# def video_feed():
#     global video
#     return Response(gen(video),mimetype='multipart/x-mixed-replace; boundary=frame')
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=2204, threaded=True)