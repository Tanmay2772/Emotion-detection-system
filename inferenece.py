import mediapipe as mp
import numpy as np
import cv2
from keras.models import load_model

model=load_model("model.h5")
label=np.load("labels.npy")

cap=cv2.VideoCapture(0)



holistic=mp.solutions.holistic
hands=mp.solutions.hands
hoils=holistic.Holistic()
drawing=mp.solutions.drawing_utils

X=[]
data_size=0


while True:
    lst=[]

    _,frm=cap.read()

    frm=cv2.flip(frm,1)

    res=hoils.process(cv2.cvtColor(frm,cv2.COLOR_BGR2RGB))
    
    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x-res.face_landmarks.landmark[1].x)
            lst.append(i.y-res.face_landmarks.landmark[1].y)
        
        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x-res.face_landmarks.landmark[8].x)
                lst.append(i.y-res.face_landmarks.landmark[8].y)

        else:
            for i in range(42):
                lst.append(0.0)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x-res.face_landmarks.landmark[8].x)
                lst.append(i.y-res.face_landmarks.landmark[8].y)

        else:
            for i in range(42):
                lst.append(0.0)
        


        lst = np.array(lst).reshape(1, -1)


        pred=label[np.argmax(model.predict(lst))]

        print(pred)
        cv2.putText(frm,pred, (50,50),cv2.FONT_ITALIC,1,(255,0,0),2)

       

    drawing.draw_landmarks(frm,res.face_landmarks,holistic.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm,res.left_hand_landmarks,hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm,res.right_hand_landmarks,hands.HAND_CONNECTIONS)
    


    cv2.imshow("Window",frm)

    if cv2.waitKey(1)==27 :
        cv2.destroyAllWindows()
        cap.release()
        break 


