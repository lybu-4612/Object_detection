import cv2 as cv

# img= cv.imread("lena.jpg")
cap= cv.VideoCapture("Traffic_IP_Camera_video_Gr0HpDM8Ki8_137.mp4")

cap.set(3,648)
cap.set(4,480)

classnames= []
Classfile= "coco.names.txt"
with open(Classfile, "rt") as f:
    classnames = f.read().rstrip('\n').split('\n')
print(type(classnames))

configpath= "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightpath= "frozen_inference_graph.pb"

net= cv.dnn_DetectionModel(weightpath,configpath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    
    set,img= cap.read()
    classids, configs, bbox= net.detect(img, confThreshold=0.5) 
    # print (classids, configs, bbox)
    if len(classids)!= 0:
        for classid,confidence,box in zip(classids.flatten(), configs.flatten(), bbox):
            cv.rectangle(img,box,color=(0,255,0),thickness= 3)
            cv.putText(img,classnames[classid-1],(box[0]+10,box[1]+30),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        




    cv.imshow("image",img) # image is the name of the window
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()


# import cv2 as cv

# img= cv.imread("lena.jpg")

# Classfile= []
# Classfile= "coco.names.txt"
# with open(Classfile, "rt") as f:
#     classnames = f.read().rstrip('\n').split('\n')

# configpath= "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"


# weightpath= "frozen_inference_graph.pb"

# net= cv.dnn_DetectionModel(weightpath,configpath)
# net.setInputSize(320,320)
# net.setInputScale(1.0/127.5)
# net.setInputMean((127.5, 127.5, 127.5))
# net.setInputSwapRB(True)
