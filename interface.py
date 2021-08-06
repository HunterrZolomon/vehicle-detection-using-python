from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import vehicles


def detect_vehicle(file):

    file_name = str(file)
    cap=cv2.VideoCapture(file_name)
    cnt_up=0
    cnt_down=0

    #Get width and height of video
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameArea=h*w
    areaTH=frameArea/400

    #Lines
    line_up=int(2*(h/5))
    line_down=int(3*(h/5))

    up_limit=int(1*(h/5))
    down_limit=int(4*(h/5))


    #Background Subtractor - to generate a foreground mask 
    fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    #Kernals for Morphological Transform
    kernalOp = np.ones((3,3),np.uint8)
    kernalCl = np.ones((11,11),np.uint)


    font = cv2.FONT_HERSHEY_SIMPLEX
    cars = []
    max_p_age = 5
    pid = 1

    while(cap.isOpened()):
        ret,frame=cap.read()
        for i in cars:
            i.age_one()
        fgmask=fgbg.apply(frame)

        if ret==True:

            #Binarization of Image
            ret,imBin=cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
            
            #Morphological Opening (Basically Erosion followed by Dilation) - to remove small objects from foreground
            mask=cv2.morphologyEx(imBin,cv2.MORPH_OPEN,kernalOp)

            #Morphological Closing (Reverse of Opening) - to remove small holes from foreground
            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.float32(kernalCl))


            #Find Contours in the image
            countours0,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for cnt in countours0:
                area=cv2.contourArea(cnt)
                #Filters contours which are greater than Threshold Area(Fixed Value)
                if area>areaTH:
                    
                    #Checking Center of mass of the contour
                    m=cv2.moments(cnt)
                    cx=int(m['m10']/m['m00'])
                    cy=int(m['m01']/m['m00'])
                    
                    #Bounding Rectangle to show in Output
                    x,y,w,h=cv2.boundingRect(cnt)

                    #Checking for Direction and Updating Up/Down Counter
                    new=True
                    if cy in range(up_limit,down_limit):
                        for i in cars:
                            if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                                new = False
                                i.updateCoords(cx, cy)

                                if i.going_UP(line_down,line_up)==True:
                                    cnt_up+=1
                                elif i.going_DOWN(line_down,line_up)==True:
                                    cnt_down+=1
                                break
                            
                            if i.getState()=='1':
                                if i.getDir()=='down'and i.getY()>down_limit:
                                    i.setDone()
                                elif i.getDir()=='up'and i.getY()<up_limit:
                                    i.setDone()
                        
                            if i.timedOut():
                                index=cars.index(i)
                                cars.pop(index)
                                del i

                        if new==True:
                            p=vehicles.Car(pid,cx,cy,max_p_age)
                            cars.append(p)
                            pid+=1

                    #Adding a Green Rectangle to show that Vehicle is detected
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            #Adding Text to output to show Number of Vehicles going in Up/Down Direction
            str_up='UP: '+str(cnt_up)
            str_down='DOWN: '+str(cnt_down)
            cv2.putText(frame, str_up, (10, 40), font,1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, str_down, (10, 90), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        else:
            break

    cap.release()


app = Flask(__name__)


@app.route('/video_feed')
def video_feed():
    return Response(detect_vehicle(file_name), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploader', methods = ['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        global file_name
        file_name = f.filename
        return render_template('output.html')


if __name__ == '__main__':
    app.run(debug=True)
