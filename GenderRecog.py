import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
cap = cv2.VideoCapture(0)
width = cap.get(3)
height = cap.get(4)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']
genderNet = cv2.dnn.readNet(genderModel, genderProto)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return_value, image = cap.read()
    if format(len(faces)) == "1":
        ToBeTrueOrNotToBeTrue = 1
    elif format(len(faces)) == "0":
        ToBeTrueOrNotToBeTrue = 1
    else:
        ToBeTrueOrNotToBeTrue = 0
    xx = -1
    yy = -1
    for (x, y, w, h) in faces:
        if h > 0 and w > 0:
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')
            cv2.putText(img, f'{gender}', ((x, y, w, h)[0], (x, y, w, h)[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(img, (x + w // 2, y + h // 2), (w // 2), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
    blob = cv2.dnn.blobFromImage(img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]
    print(f'Gender: {gender}')
    img = cv2.resize(img, (0, 0), fx=3, fy=2.5)
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
