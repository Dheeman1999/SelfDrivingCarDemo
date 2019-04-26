# SelfDrivingCarDemo
A repository created for testing the efficacy of neural networks designed for self driving cars.
## Getting Started 
The repository will contain 7 files: </br>
- FurtherOptimisation.ipynb
- OriginalNetwork.ipynb
- The Simulator application </br> 
- The Driver.py script </br>
- 3 keras models : 
  - model.h5  
  - originalmodel.h5 
  - NoAug.h5 . </br>
 All the three files are trained keras neural net models for the simulator.The ```model.h5``` is a newly created and trained optimal model, the ```originalmodel.h5``` is a trained model of the neural net proposed by NVIDIA. The third one is the same original model trained without data augmentation.It has been added just to show how important image augmentation is(if you test the NoAug.h5 model you would see that the model fails miserably in the simulation!).The Driver.py is the interface that connects the model with the simulator.
## Important note
The two jupyter notebooks are for reference only and are advised to be opened only in Google Collaboratory. You can train the network there if you wish to and see the results.(Make sure the GPU mode is on for hardware acceleration).
## Dependencies
The repository is designed for linux only. In addition the following would be required to be installed : </br>
- Python 3 </br>
- Keras </br>
- socketio </br>
- eventlet </br>
- flask </br>
- base64 </br>
- opencv </br>
- io </br>
- PIL </br>
- numpy </br>

## Execution
Go to the local clone of the repository, open the terminal and execute the command :</br>
```
python driver.py
```
Now run the simulator, which should look like this :</br>
![Screenshot from 2019-04-26 22-35-29](https://user-images.githubusercontent.com/39672404/56824401-0f146280-6874-11e9-93a8-257b070b6dc0.png)
Now, select one of the tracks by clicking on them, and then click on 'Autonomous mode'.You should see the model driving the car on its own.

## Simulating a new keras model.
Suppose you have your own keras model for testing in the simulator, say ```'yourmodel.h5'``` (it must be in the same directory as the simulator).All you need to do is to change the ```'Driver.py'``` code. Replace ```'model.h5'``` to ```'yourmodel.h5'``` in the third last line of the code (you can also change the speed limit by changing the ```speed_limit``` variable):
```import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

sio = socketio.Server()

app = Flask(__name__) #'__main__'
speed_limit = 30
def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)



@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


```
## Author
Created By Dheeman Kuaner ( https://github.com/Dheeman1999) .</br>
linkedin: https://www.linkedin.com/in/dheeman-kuaner

## Acknowledgements
Udacity for making the simulator open source </br>

## Further references
visit  https://github.com/Dheeman1999/track  for sampe dataset for training your neural net.
