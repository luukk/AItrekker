import cv2

cap1 = cv2.VideoCapture('data/video/video_cut_2.mp4')


while(True):
    ret, frame = cap1.read()
    if ret:
        cv2.imshow("frame", frame)

cap1.release()
cv2.destroyAllWindows()