import cv2
import mediapipe as mp
import time
import math

class poseDetector():
    #Constructor
    def __init__(self, mode=False, complexity = 1, smooth=True,enableSegmentation = False,smoothSegmentation=True,
                  detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.enableSegmentation = enableSegmentation
        self.smoothSegmentation = smoothSegmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity,self.smooth,self.enableSegmentation,self.smoothSegmentation,
                                     self.detectionCon, self.trackCon)

        #Function to find the pose
    def findPose(self,img,draw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if(draw):
            if (self.results.pose_landmarks):
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self,img,draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                #print(id,lm)
                cx,cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id,cx,cy])
                if(draw):
                    cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
        return self.lmList
    def findAngle(self, img, p1, p2 , p3, draw = True):
        x1,y1 = self.lmList[p1][1:]
        x2,y2 = self.lmList[p2][1:]
        x3,y3 = self.lmList[p3][1:]

        angle = math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))
        print(angle)
        if draw:
            cv2.circle(img,(x1,y1),5,(255,0,0),cv2.FILLED)
            cv2.circle(img,(x2,y2),5,(255,0,0),cv2.FILLED)
            cv2.circle(img,(x3,y3),5,(255,0,0),cv2.FILLED)

    # for id, lm in enumerate(results.pose_landmarks.landmark):
    #     h,w,c = img.shape
    #     print(id,lm)
    #     #obtaining pixel values of the image
    #     cx,cy = int(lm.x * w), int(lm.y * h)
    #     cv2.circle(img,(cx,cy),10,(255,0,0),cv2.FILLED)


def main():

    cap = cv2.VideoCapture('PoseVideos/squat_front_view.mp4')
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        print(lmList[14])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__=="__main__":
    main()
