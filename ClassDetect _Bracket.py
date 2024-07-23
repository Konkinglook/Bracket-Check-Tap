import cv2
import serial
import time
import os

# arduinoData = serial.Serial('com4' , baudrate=9600)

# def arduinoDataOut():
#     if className[2] and className[3] :
#         print("switch ON")
#         arduinoData.write(b'H')
#     else:
#         print("switch OFF")
#         arduinoData.write(b'L')

path = 'Img_Query'
orb = cv2.ORB_create(nfeatures=2000)
###### Import Images
images = []
className = []
myList = os.listdir(path) #แสดงรายชื่อไฟล์ที่อยู่ใน path ที่กำหนดโดยแสดงผลลัพธ์ออกมาเป็นชนิดข้อมูลแบบ list

for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}',0)
    images.append(imgCur)
    className.append(os.path.splitext(cl)[0]) #ตัดตัวอักษร .jpg ออกให้เหลือแต่่ชื่อไฟล์ที่ตั้งไว้
    # print(className)

def findDes(images):
    desList=[]
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        desList.append(des)
    return desList

def findID(img,desList,thres=13):
    kp2,des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des,des2,k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m]) 
            matchList.append(len(good))    
    except:
        pass
    time.sleep(0.05)
    # print(matchList)

    if len(matchList)!= 0:
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))
            # print("max = ",finalVal)
    return finalVal

desList = findDes(images)
# print(len(desList))

cap1 = cv2.VideoCapture(0) 
cap2 = cv2.VideoCapture(1)
while (True):
    check, fram1 = cap1.read()
    check, fram2 = cap2.read()

    imgOrg1 = fram1.copy()
    imgOrg2 = fram2.copy()
    
    fram1 = cv2.cvtColor(fram1,cv2.COLOR_BGR2GRAY)
    fram2 = cv2.cvtColor(fram2,cv2.COLOR_BGR2GRAY)

    id = findID(fram1,desList)
    if id != -1:
        if id == 2:
            cv2.putText(imgOrg1,className[2],(250,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)
            print("H1",className[2])
            # arduinoDataOut()
            
        else:
            cv2.putText(imgOrg1,className[0],(250,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
            print("L1",className[0])
            # arduinoDataOut()

    id = findID(fram2,desList)
    if id != -1:
        if id == 3:
            cv2.putText(imgOrg2,className[3],(250,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)
            print("H2",className[3])
            # arduinoDataOut()

        else:
            cv2.putText(imgOrg2,className[1],(250,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
            print("L2",className[1])
            # arduinoDataOut()

    if check == True:
        cv2.imshow('CAM1',imgOrg1)
        cv2.imshow('CAM2',imgOrg2)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap1.release()
cv2.destroyAllWindows()   

   
        
    
