import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

'''
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)     
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  
    
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', frame)
    
    if cv2.waitKey(1)== ord('s'):
        cv2.imwrite('cvtest.jpg', frame)
    if cv2.waitKey(1)== ord('q'):
        break

cap.release() 
cv2.destroyAllWindows() 
'''
img = cv2.imread('ai/cvtest.jpg')
face = face_cascade.detectMultiScale(img, 1.1, 4)
for (x, y, w, h) in face:
    face2 = img[x:x+w, y:y+h]
# face = img[130:330, 180:380]
cv2.imshow('face', face2)
cv2.waitKey()
cv2.destroyAllWindows()