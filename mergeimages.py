import cv2
import numpy as np
import matplotlib.pyplot as plt
# script creates perspective transformation view from video

# # def resizeimage(frame):
# #     height, width, layers = frame.shape
# #     new_w = 640
# #     new_h = 363
# #     frame = cv2.resize(frame, (new_w, new_h))
# #     return frame


# # def getPerspectiveTransformation(frame):
# #     rows, cols, ch = frame.shape
# #     pts1 = np.float32([[19, 20], [600, 20], [0, 363], [649, 363]])
# #     pts2 = np.float32([[0, 0], [640, 0], [0, 363], [640, 363]])
# #     M = cv2.getPerspectiveTransform(pts1, pts2)
# #     dst = cv2.warpPerspective(frame, M, (640, 363))
# #     return dst

# def resizeimage(frame):
#     height, width, layers = frame.shape
#     new_h = 270
#     new_w = 430
#     frame = cv2.resize(frame, (new_w, new_h))
#     return frame


# def getPerspectiveTransformation(frame):
#     rows, cols, ch = frame.shape
#     pts1 = np.float32([[60, 20], [380, 20], [0, 100], [430, 100]])
#     pts2 = np.float32([[0, 0], [430, 0], [0, 270], [430, 270]])
#     M = cv2.getPerspectiveTransform(pts1, pts2)
#     dst = cv2.warpPerspective(frame, M, (430, 270))
#     return dst


# def getPerspectiveTransformationMiddle(frame):
#     rows, cols, ch = frame.shape
#     pts1 = np.float32([[150, 20], [550, 20], [0, 363], [649, 363]])
#     pts2 = np.float32([[0, 0], [640, 0], [0, 363], [640, 363]])
#     M = cv2.getPerspectiveTransform(pts1, pts2)
#     dst = cv2.warpPerspective(frame, M, (640, 363))
#     return dst


# # img = cv2.imread('images/ball.png')
# # img1 = cv2.imread('images/ball.png')

# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1290, 270))

cap1 = cv2.VideoCapture('data/video/video_cut_2.mp4')
# cap2 = cv2.VideoCapture('data/video/filmrole3.avi')
# cap3 = cv2.VideoCapture('data/video/filmrole5.avi')
# cap4 = cv2.VideoCapture('data/video/filmrole2.avi')
# cap5= cv2.VideoCapture('data/video/filmrole4.avi')
# cap6 = cv2.VideoCapture('data/video/filmrole6.avi')

# vis = np.concatenate((img, img1), axis = 1)
# cv2.imshow('out', vis)

while(1):
    ret1, frame1 = cap1.read()
    if ret1:
        # ret2, frame2 = cap2.read()
        # ret3, frame3 = cap3.read()

        # ret1, frame4 = cap4.read()
        # ret2, frame5 = cap5.read()
        # ret3, frame6 = cap6.read()

        # frame1 = resizeimage(frame1)
        # frame2 = resizeimage(frame2)
        # frame3 = resizeimage(frame3)
        # original = np.concatenate((frame3, frame2, frame1), axis=1)

        # cv2.imshow('original frames merged', frame1)
        plt.imshow(frame1)
        plt.show()
        # cv2.imwrite('merged_original.jpg', original);
        # cv2.imshow("original", frame1)

        # frame4 = resizeimage(frame4)
        # frame5 = resizeimage(frame5)
        # frame6 = resizeimage(frame6)

        # frame3 = getPerspectiveTransformation(frame3)
        # frame2 = getPerspectiveTransformation(frame2)
        # frame1 = getPerspectiveTransformation(frame1)
        # # cv2.imshow("transformed", frame1)

        # # frame4 = getPerspectiveTransformation(frame4)
        # # frame5 = getPerspectiveTransformation(frame5)
        # # frame6 = getPerspectiveTransformation(frame6)

        # # frame5 = cv2.flip(frame5, 1)
        # # frame1 = cv2.flip(frame1, 1)
        # result = np.concatenate((frame3, frame2, frame1), axis=1)

        # # totalresult = np.concatenate((result, result2), axis=0)
        # cv2.imshow('result', result)
        # cv2.imwrite('merged_transformed.jpg', result);

        # write the flipped frame
        # out.write(result)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break

cap1.release()
# cap2.release()
# cap3.release()
out.release()
cv2.destroyAllWindows()