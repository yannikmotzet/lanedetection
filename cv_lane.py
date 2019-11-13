# date: 13.11.2019
# username: mageejo
# name: Yannik Motzet
# description: a basic program to illustrate lane contours

#sample image from wikipedia.org: https://en.wikipedia.org/wiki/Types_of_road#/media/File:Road_in_Norway.jpg


import cv2 as cv

#read image from source
raw = cv.imread("Road_in_Norway.jpg")

#convert raw picture to greyscale
grey = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)

#convert greyscale img to binary img
_, binary = cv.threshold(grey, 150, 255, cv.THRESH_BINARY)

#detect contours in binary img
contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

#approximate coutours with function and draw contours to raw pimg
for cnt in contours:
    approx = cv.approxPolyDP(cnt, 0.01*cv.arcLength(cnt, True), True)
    cv.drawContours(raw, [approx], 0, (0), 5)

#show images in windows
cv.imshow("Raw", raw)
cv.imshow("Binary", binary)
cv.waitKey(0)
cv.destroyAllWindows()
