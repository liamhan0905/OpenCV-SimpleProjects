import cv2
import numpy as np

widthImg = 480
heightImg = 640
cap = cv2.VideoCapture(0)
cap.set(3,widthImg)
cap.set(4,heightImg)
cap.set(10,150) #brightness

def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    # cv2.imshow("blur", imgBlur)
    imgCanny = cv2.Canny(imgBlur,30,90)
    # cv2.imshow("canny", imgCanny)
    kernel = np.ones((5,5))
    imgDilation = cv2.dilate(imgCanny,kernel, iterations = 2)
    # cv2.imshow("dilation", imgDilation)
    imgErode = cv2.erode(imgDilation,kernel,iterations = 1 )
    # cv2.imshow("erode", imgErode)
    imgThres = imgErode
    return imgThres

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 100000:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), thickness=2)  # -1 to draw on all contour
            peri = cv2.arcLength(cnt,True)
            # print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True) # find corners
            # print(len(approx))
            if area > maxArea and len(approx == 4):
                biggest = approx
                maxArea = area
            # print(biggest)
    cv2.drawContours(imgContour, biggest, -1, (0, 0, 255), thickness=10)  # -1 to draw on all contour
    return biggest

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    newPoints = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    # print("add",add)
    # print("min",myPoints[np.argmin(add)])
    newPoints[0] = myPoints[np.argmin(add)]
    newPoints[3] = myPoints[np.argmax(add)]

    diff = np.diff(myPoints, axis=1)
    newPoints[1] = myPoints[np.argmin(diff)]
    newPoints[2] = myPoints[np.argmax(diff)]

    return newPoints

def getWarp(img,biggest):
    # print(biggest)
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    imgCropped = imgOutput[10:imgOutput.shape[0]-10,10:imgOutput.shape[1]-10]
    imgCropped = cv2.resize(imgCropped,(widthImg,heightImg))

    return imgCropped


while True:
    success, img = cap.read()
    img = cv2.resize(img,(widthImg,heightImg))
    imgContour = img.copy()
    imgThresh = preProcessing(img)
    biggest = getContours(imgThresh)
    imgWarped = getWarp(img,biggest)

    cv2.imshow("Original Img", img)
    cv2.imshow("Original Img with Contour", imgContour)
    cv2.imshow("Video",imgWarped)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
