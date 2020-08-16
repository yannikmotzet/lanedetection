# date: 13.11.2019
# name: Yannik Motzet
# description: a basic program to illustrate lane contours

import cv2
import numpy as np
import random

# video source
cap = cv2.VideoCapture('lane_detection.mp4')

picHeight = 960
picWidth = 1280

while(cap.isOpened()):
    # capture frame-by-frame
    _, frame = cap.read()

    # illustrate vertices for perspective transformation
    # point top left
    cv2.circle(frame, (20, 300), 5, (255, 0, 0), -1)
    # point top right
    cv2.circle(frame, (picWidth-20, 300), 5, (0, 255, 0), -1)
    # point bottom left
    cv2.circle(frame, (0, picHeight), 5, (255, 0, 0), -1)
    # point bottom right
    cv2.circle(frame, (picWidth, picHeight), 5, (0, 0, 255), -1)

    # perspective transformation
    pts1 = np.float32([[0, 300], [picWidth, 300], [
                      0, picHeight], [picWidth, picHeight]])
    pts2 = np.float32([[0, 0], [645, 0], [290, 480], [645-290, 480]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    result = cv2.warpPerspective(frame, matrix, (645, 480))
    contour_image = result.copy()

    # binary image
    grey = cv2.cvtColor(contour_image, cv2.COLOR_RGB2GRAY)
    _, threshold = cv2.threshold(grey, 200, 255, cv2.THRESH_BINARY)
    
    # contour detection
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)    #CHAIN_APPROX_SIMPLE

    # print(contours[0].shape)
    # print(contours[0].shape[0])

    number_of_countours = 0
    for cnt in contours:
        # b, g, r = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        b, g, r = 0, 0, 0
        if cnt.shape[0] > 200:
            print(cnt.shape[0])
            number_of_countours += 1
            if number_of_countours > 3:
                b, g, r = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                print("something went wrong")
            for l in range(cnt.shape[0]):
                cv2.circle(contour_image, (cnt[l, 0, 0] , cnt[l, 0, 1]), 1, (b, g, r), -1)
    print("number contours: " + str(number_of_countours))

    
    # # draw contours
    # for cnt in contours:
    #     cv2.drawContours(contour_image, [cnt], 0, (0, 0, 255), 3)

    # cv2.drawContours(contour_image, contours, 1, (0, 0, 255), 3)

    
    # output windows
    # cv2.imshow('input', frame)
    # cv2.imshow('binary', threshold)
    # cv2.imshow('perspective transformation', result)
    cv2.imshow('contours', contour_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# cv2.waitKey(0)
cv2.destroyAllWindows()