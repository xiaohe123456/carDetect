import cv2
import numpy as np
import os

carPath = "D:/Data/Image/beijing_bohaidongdt23/JPEGImages/"
imagepath1 = carPath + "beijing_bohaidongdt18_00801.jpg"

path = "D:/Data/Image/beijing_bohaidongdt28/railpic/"
imagepath2 = path + "beijing_bohaidongdt28_00031.jpg"

# imagepath3 = path + "beijing_bohaidongdt18_00031.jpg"

image2 = cv2.imread(imagepath2)
height2, width2, channels2 = image2.shape
#cropImage2 = image2[0:height2, int(width2 / 2) : width2]
# cropImage2 = image2[0:height2, 0 : int(width2 / 2)]
cropImage2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

dir_list = [
    "D:/Data/Image/beijing_bohaidongdt28/JPEGImages"
]

dist_dir = "D:/Data/test"
if dist_dir[-1] == '/' or dist_dir[-1] == '\\':
    dist_dir = dist_dir[:-1]
if not os.path.isdir(dist_dir):
    os.makedirs(dist_dir)

num = []
size = [0]

for dir in dir_list:
    if dir[-1] == '/' or dir[-1] == '\\':
        dir = dir[:-1]

    file_list = os.listdir(dir)
    size = len(file_list)
    area = [[] for index in range(size+1)]
    area[0].append(0)
    number = 1
    for file_name in file_list:
        file_path = dir + '/' + file_name
        image1 = cv2.imread(file_path)
        height, width, channels = image1.shape
        # cropImage1 = image1[0:height, int(width / 2) : width]
        # cropImage1 = image1[0:height, 0 : int(width / 2)]
        cropImage1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
        sub = cv2.subtract(cropImage1,cropImage2)
        cv2.imwrite(dist_dir + '/' + 'sub' + file_name, sub)
        ret, binary = cv2.threshold(sub,50,255,cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        #dst = cv2.dilate(binary,kernel,iterations=3)
        dst = cv2.erode(binary,kernel)
        image, contours,hierarchy = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cropImage1 = cv2.drawContours(cropImage1,contours,-1,(0,255,0),8)
        cv2.imwrite(dist_dir + '/' + file_name, cropImage1)
        count = len(contours)
        num.append(count)

        for i in range(0,count - 1):
            tmp = cv2.contourArea(contours[i])
            # area[number].append(tmp)
            if (tmp > 500):
                area[number].append(tmp)

        area[number].sort(reverse=True)
        print(number)
        print(file_path)
        print(len(area[number]))
        print(area[number])
        contours = sorted(area[number], reverse=True)[:3]
        print(contours)
        if (len(area[number]) == 0):
            print("无车")
            area[number].append(0)
        elif (len(area[number]) > 0 and (area[number][0] == area[number - 1][0]) and (len(area[number]) == len(area[number - 1]))):
            print("有车静止")
        elif (len(area[number]) > 0):
            print("有车经过")
        number = number + 1

print(num)


