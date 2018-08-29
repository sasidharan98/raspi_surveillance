import io
import picamera
import cv2
import numpy
import cv2
import os
from PIL import Image
import os
import sys
import string
import urllib
import urllib2
import re
import requests
from time import sleep
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)


Motor1A = 16
Motor1B = 18
Motor1E = 22
motor_delay=10
reed=2
motor=10
pir=23
motor_status=0
GPIO.setup(reed,GPIO.IN,pull_up_down=GPIO.PUD_UP)
GPIO.setup(motor,GPIO.IN,pull_up_down=GPIO.PUD_UP)
GPIO.setup(pir,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)

GPIO.setup(Motor1A,GPIO.OUT)
GPIO.setup(Motor1B,GPIO.OUT)
GPIO.setup(Motor1E,GPIO.OUT)
 


while(1):

    r= requests.post("https://api.telegram.org/bot251431441:AAF5guX6OGgYPiWbvCYhf5PwrLYPzMhUzTQ/sendmessage?chat_id=222152951&text=working")
    if (r.status_code==200):
        print r.status_code
        break

#1.door status closed na check pir sensor
2.pir status interrupt achuna take pic send to messu & do opening/closing
#3.note motor status so that ifs interrupt button pressed decides one action.

def reed_op(channel):
    while(GPIO.input(reed)):
        sleep(30)
    motor_status=0
    motor_move()
def motor_move():


    if (motor_status==0):# closing
        GPIO.output(Motor1A,GPIO.HIGH)
        GPIO.output(Motor1B,GPIO.LOW)
        GPIO.output(Motor1E,GPIO.HIGH)
        sleep(motor_delay)
        GPIO.output(Motor1A,GPIO.LOW)
        GPIO.output(Motor1B,GPIO.LOW)
        GPIO.output(Motor1E,GPIO.LOW)
    else:           #opening
        GPIO.output(Motor1A,GPIO.LOW)
        GPIO.output(Motor1B,GPIO.HIGH)
        GPIO.output(Motor1E,GPIO.HIGH)
        sleep(motor_delay)
        GPIO.output(Motor1A,GPIO.LOW)
        GPIO.output(Motor1B,GPIO.LOW)
        GPIO.output(Motor1E,GPIO.LOW)


def motor_op(channel):
    if (motor_status==0):
        break
    else:
        motor_move()
    sleep(5)


def checku(str ):
 for f in range (d-1,0,-1):
    if( line[f]=="t"):
        e=f+1
        f=f-1
        if(line[f]=="x"):
            f=f-1
            if(line[f]=="e"):
                f=f-1
                if(line[f]=="t"):
                    break
 return line[e:d]

GPIO.add_event_detect(reed,GPIO.FALLING,callback=reed_op,bouncetime=300)
GPIO.add_event_detect(motor,GPIO.FALLING,callback=motor_op,bouncetime=300)

while(1):
    while(1):
        if (GPIO.input(pir)):
            sleep(2)
            if (GPIO.input(pir)):
                sleep(2)
                if (GPIO.input(pir)):
                    break
            else:
                continue
    stream = io.BytesIO()
with picamera.PiCamera() as camera:
    camera.resolution = (2592,1944)
    camera.capture(stream, format='jpeg')
buff = numpy.fromstring(stream.getvalue(), dtype=numpy.uint8)
image = cv2.imdecode(buff, 1)
face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

#print "Found "+str(len(faces))+" face(s)"

for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)

cv2.imwrite('result.jpg',image)



case=0
path1=[]
init_path = "\home\pi\training\\"
for root, dirs, files in os.walk(init_path):
    for subdir in dirs:
        path1.append(subdir)

while(case<=2):
    path = init_path+path1[case]
    recognizer = cv2.createLBPHFaceRecognizer()
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    labels =[]
    for i in image_paths:
        image_pil = Image.open(i).convert('L')
        img = np.array(image_pil, 'uint8')
        label = int(os.path.split(i)[1].split(".")[0])
        labels.append(label)
        faces = face_cascade.detectMultiScale(img,1.3,5)
       # for(x,y,w,h) in faces:
        images.append(img)



    train = recognizer.train(images,np.array(labels))


    imge = cv2.imread('result.jpeg')
    gray = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)
    #face1 = cv2.resize(gray,(640,480),interpolation = cv2.INTER_AREA)
    face = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in face:
          predicted,base = recognizer.predict(gray)
          base1.append(base)
          
    if(len(faces)==None):
        print "none"
    case=case+1
#cv2.waitKey(0)
cv2.destroyAllWindows()

ans=sorted(range(len(base1)),key=lambda k:base1[k])
#print ans
base1.sort()
if (base1[0]<30):                                      ################################################################base is here
    #print "ur recognised"+path1[ans[0]]
    msg= "open"
    msg1= path1[ans[0]]
else:
    msg="no"

r=requests.get("https://api.telegram.org/bot251431441:AAF5guX6OGgYPiWbvCYhf5PwrLYPzMhUzTQ/getupdates")
line= re.sub('[{",}:]', '', r.text)
d=len(line)-1
l=d


if (msg=="open"):
    motor_status=1
    motor_move()
    r = requests.post("https://api.telegram.org/bot251431441:AAF5guX6OGgYPiWbvCYhf5PwrLYPzMhUzTQ/sendmessage?chat_id=222152951&text"+"="+msg1)


elif(msg=="no"):
    url = "https://api.telegram.org/bot251431441:AAF5guX6OGgYPiWbvCYhf5PwrLYPzMhUzTQ/sendPhoto";
    files = {'photo': open('result.jpg', 'rb')}
    data = {'chat_id': "222152951"}
    r = requests.post(url, files=files, data=data)
    print(r.status_code, r.reason, r.content)
    l=d
    while(d==l):
        r = requests.get("https://api.telegram.org/bot251431441:AAF5guX6OGgYPiWbvCYhf5PwrLYPzMhUzTQ/getupdates")
        line = re.sub('[{",}:]', '', r.text)
        d = len(line) - 1
    control=checku( 2 )
    if (control=="open")
         motor_status=1
         motor_move()
         result = requests.post("https://api.telegram.org/bot251431441:AAF5guX6OGgYPiWbvCYhf5PwrLYPzMhUzTQ/sendmessage?chat_id=222152951&text=opened")
        




        
        
        
    
