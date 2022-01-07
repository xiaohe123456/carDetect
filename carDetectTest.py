import cv2
import os
import numpy as np

# 设置标准图像，检测图像的下半部分
# 先将待检测图像与标准图像相减，再将标准图像与待检测图像相减，对相减的图像做二值化，提取轮廓，
# 将轮廓合并，根据轮廓面积和数量判定是否有车来。
# 目前程序可以通过修改二值化时的阈值实现对各个场景的轨道进行检测，若调整阈值后仍存在误报，则可以调整图像的灰度缩放比例，即调整图像对比度。在轨道上有人时，可以将面积适当调大，
# 对于晚上的图像二值化时的小阈值可设为50，白天的可以设为100或128。

# # JPEGImages
path = "D:/Data/farOrbitPeople-noCar/"
imagepath2 = path + "test_00005.jpg"
# path = "D:/Data/Image/closeFarRail/jiugang_rail01/railpic/"
# imagepath2 = path + "jiugang_rail01_00001.jpg"
# path = "D:/Data/Image/beijing_bohaidongdt23-day/railpic/"
# imagepath2 = path + "beijing_bohaidongdt23_00031.jpg"
image2 = cv2.imread(imagepath2)
height2, width2, channels2 = image2.shape
image2 = image2[int(height2 / 2) : height2, 0:width2]
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
cv2.imwrite("test.jpg", gray2)
dir_list = [
    # "D:/Data/image/beijing_bohaidongdt23-day/test"
    # "D:/Data/Image/closeFarRail/closeRail1"
    # "D:/Data/Image/closeFarRail/jiugang_rail01/railpic"
     "D:/Data/farOrbitPeople-noCar"
]

dist_dir = "D:/Data/result"
if dist_dir[-1] == '/' or dist_dir[-1] == '\\':
    dist_dir = dist_dir[:-1]
if not os.path.isdir(dist_dir):
    os.makedirs(dist_dir)

num = []    #列表，保存每幅图像的轮廓数量

for dir in dir_list:
    if dir[-1] == '/' or dir[-1] == '\\':
        dir = dir[:-1]

    file_list = os.listdir(dir)
    size = len(file_list)                       #文件夹中图像的数量
    area = [[] for index in range(size + 1)]    #列表  保存每幅图像中每个轮廓的面积
    length = [[] for index in range(size + 1)]  #列表  保存每幅图像中每个轮廓的长度
    area[0].append(0)                           #列表的子项也是列表  初始化
    length[0].append(0)

    number = 1                                  #记录图像序号

    for file_name in file_list:
        contoursTemp = []                       #列表  保存符合长度条件的轮廓
        file_path = dir + '/' + file_name
        image1 = cv2.imread(file_path)
        height1, width1, channels1 = image1.shape
        image1 = image1[int(height1 / 2): height1, 0:width1]            #获取图像的下半部分
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

        #设置的标准图像减待检测图像
        tmp = np.zeros_like(gray1)
        sub1 = cv2.absdiff(gray2, gray1)                    #图像作差并取绝对值，用subtract会将小于0的像素全都置为0
        sub1 = cv2.scaleAdd(sub1, 1, tmp)                   #调整图像灰度范围  改变参数可以调整图像对比度
        _, binary1 = cv2.threshold(sub1, 100, 255, cv2.THRESH_BINARY)
        # cv2.imwrite("binary1.jpg", binary1)
        image, contours1, hierarchy = cv2.findContours(binary1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # contoursImage1 = cv2.drawContours(image1, contours1, -1, (0, 255, 0), 8)
        # cv2.imwrite("contours1.jpg", contoursImage1)

        #待检测图像减设置的标准图像
        sub2 = cv2.absdiff(gray1, gray2)
        sub2 = cv2.scaleAdd(sub2, 1, tmp)
        _, binary2 = cv2.threshold(sub2, 100, 255, cv2.THRESH_BINARY)
        # cv2.imwrite("binary2.jpg", binary2)
        image, contours2, hierarchy = cv2.findContours(binary2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # contoursImage2 = cv2.drawContours(image2, contours2, -1, (0, 255, 0), 8)
        # cv2.imwrite("contours2.jpg", contoursImage2)

        contours = np.append(contours1, contours2, axis=0)          #将两个差值图像上的轮廓合并

        # contoursImage = cv2.drawContours(image1, contours, -1, (0, 255, 0), 8)
        # cv2.imwrite("contours.jpg", contoursImage)

        count = len(contours)
        num.append(count)
        for i in range(0, count):
            contourLength = cv2.arcLength(contours[i], False)  #轮廓长度
            if contourLength > 500:
                length[number].append(contourLength)            #保存轮廓长度大于500的轮廓长度
                contoursTemp.append(contours[i])                #保存轮廓长度大于500的轮廓

        for j in range(0, len(contoursTemp)):                   #在符合长度条件的轮廓中选择符合面积条件的
            tmp = cv2.contourArea(contoursTemp[j])              #轮廓面积
            if tmp > 1000:
                area[number].append(tmp)                        #保存轮廓面积大于100的轮廓面积

        # print(area[number])
        area[number].sort(reverse=True)
        # print(file_path)
        contours = sorted(area[number], reverse=True)[:3]
        print(contours)

        if len(area[number]) == 0:  # 轮廓数目为0时表示没有车辆经过
            print("无车")
            area[number].append(0)
        # 轮廓数量大于0分为有车经过和有车静止两种情况
        elif len(area[number]) > 0:   # 当上一幅图像中轮廓的数量与当前图像轮廓数量相等且上一幅图像第一个轮廓面积与当前图像第一个轮廓面积相等，表明车是静止的
            if area[number][0] == area[number - 1][0] and len(area[number]) == len(area[number - 1]):
                print("有车静止")
            else:  #否则表示有车经过
                print("有车经过")
        number = number + 1
print(num)
