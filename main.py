import cv2
import mediapipe as mp
import time
import PoseModule as pm


cap = cv2.VideoCapture('PoseVideos/squat_front_view.mp4')
pTime = 0
detector = pm.poseDetector()
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img,False)
    if len(lmList)!= 0:
        detector.findAngle(img,23,25,27)

    # print(lmList[14])
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
