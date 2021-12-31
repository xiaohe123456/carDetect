import cv2


carPath = "D:/Data/Image/beijing_bohaidongdt23/railpic/"
imagepath1 = carPath + "beijing_bohaidongdt23_00061.jpg"

path = "D:/Data/Image/beijing_bohaidongdt22/railpic/"
imagepath2 = path + "beijing_bohaidongdt22_00021.jpg"

image1 = cv2.imread(imagepath1)
height1, width1, channels1 = image1.shape
grayImage1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
_, binary1 = cv2.threshold(grayImage1,100,255,cv2.THRESH_BINARY)

image2 = cv2.imread(imagepath2)
height2, width2, channels2 = image2.shape
grayImage2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
_, binary2 = cv2.threshold(grayImage2,100,255,cv2.THRESH_BINARY)

# sub = cv2.subtract(grayImage1,grayImage2)
# _, dst = cv2.threshold(sub,100,255,cv2.THRESH_BINARY)

sub = cv2.subtract(binary1,binary2)
# _, dst = cv2.threshold(sub,100,255,cv2.THRESH_BINARY)
cv2.imwrite("sub.jpg", sub)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
dst = cv2.erode(sub,kernel, iterations=3)
cv2.imwrite("dst.jpg",dst)

image, contours, hierarchy = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cropImage1 = cv2.drawContours(image1, contours, -1, (255, 0, 0), 3)
cv2.imwrite("contours.jpg", cropImage1)
area = []
for i in range(0,len(contours)):
    temp = cv2.contourArea(contours[i])
    area.append(temp)


area.sort(reverse=True)
print(area)

print(len(area))
contours = sorted(area, reverse=True)[:3]
print(contours)