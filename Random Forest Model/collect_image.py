import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 100
classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + [chr(i) for i in range(ord('0'), ord('9') + 1)]
for classs in classes:
    dir = os.path.join(DATA_DIR, classs)
    if not os.path.exists(dir):
        os.makedirs(dir)

    print("Collecting data for Class " , classs)

    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready for {}? Press "Q" ! :)'.format(classs), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break
    
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, classs, '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
