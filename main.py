import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)     
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  
    
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', frame)
    
    if cv2.waitKey(90)== ord('s'):
        cv2.imwrite('cvtest.jpg', frame)
    if cv2.waitKey(90)== ord('q'):
        break

cap.release() 
cv2.destroyAllWindows() 

img = cv2.imread('ai/cvtest.jpg')
face = face_cascade.detectMultiScale(img, 1.1, 4)
for (x, y, w, h) in face:
    r = max(w, h) / 2
    centerx = x + w / 2
    centery = y + h / 2
    nx = int(centerx - r)
    ny = int(centery - r)
    nr = int(r * 2)

    faceimg = img[ny:ny+nr, nx:nx+nr]
    
    cv2.imshow('img ori', img)
    cv2.imshow("face" , faceimg)

cv2.waitKey()
cv2.destroyAllWindows()