import cv2
import os

#设置标准图像，在图像上画检测区域  标准图像与待检测图像相减得到一个差值图像，待检测图像与标准图像相减的都一个差值图像，对差值图像分析，根据轮廓面积和数量判定是否有车来
#近远轨来车检测  只检测近轨道是否有车  近轨通过鼠标事件在图像上画出检测区域

path = "D:/Data/Image/tanggu_changche04/railpic/"
imagepath2 = path + "tanggu_changche04_00580.jpg"
# path = "D:/Data/Image/tanggu_liangshiche01/railpic/"
# imagepath2 = path + "tanggu_liangshiche01_01100.jpg"

ix, iy = -1, -1
xx, yy = -1, -1
def draw_rect(event, x, y, flags, param):
    global ix, iy, drawing, mode, cap, template, tempFlag
    if event == cv2.EVENT_LBUTTONDOWN:
        tempFlag = True
        drawing = True
        ix, iy = x, y                        #按下鼠标左键，用全局变量ix,iy记录下当前坐标点
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False                  #鼠标左键抬起，画出矩形框
            xx, yy = x, y
            # cv2.rectangle(image2, (ix, iy), (x, y), (255, 0, 0), 8)
            template = image2[iy:y, ix:x, :]  #截取框中的目标图像
            cv2.imshow('img', image2)         #显示画框后的图像
            img = image2[iy:yy, ix:xx, :]
            print(iy, yy, ix, xx)
            cv2.imshow("tmp", img)


# cv2.namedWindow('img', cv2.WINDOW_NORMAL)

# cv2.setMouseCallback('img', draw_rect)           #鼠标事件  在图像上画矩形检测区域
image2 = cv2.imread(imagepath2)
# cv2.waitKey(0)
iy = 944
yy = 1043
ix = 10
xx = 1863

image2 = image2[iy:yy, ix:xx, :]
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# cv2.imwrite("gray2.jpg", gray2)
# JPEGImages
dir_list = [
    # "D:/Data/image/beijing_bohaidongdt1504-day"
    "D:/Data/Image/tanggu_changche04/railpic"
    # "D:/Data/Image/closeFarRail/jiugang_rail01/railpic"
    #  "D:/Data/closeOrbitPeople-noCar"
    #  "D:/Data/farOrbitPeople-noCar"
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
        contours = []                           #列表  保存轮廓点坐标
        file_path = dir + '/' + file_name
        image1 = cv2.imread(file_path)
        height1, width1, channels1 = image1.shape
        image1 = image1[iy:yy, ix:xx, :]                        #获取图像的检测部分
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

        #设置的标准图像减待检测图像
        # tmp = np.zeros_like(gray1)
        sub1 = cv2.subtract(gray2, gray1)                       #图像作差，用subtract会将小于0的像素全都置为0
        # sub1 = cv2.scaleAdd(sub1, 1.2, tmp)                   #调整图像灰度范围  改变参数可以调整图像对比度
        _, binary1 = cv2.threshold(sub1, 50, 255, cv2.THRESH_BINARY)
        image, contours1, hierarchy = cv2.findContours(binary1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contoursImage1 = cv2.drawContours(image1, contours1, -1, (0, 0, 255), 3)
        # cv2.imwrite(dist_dir + '/' + 'contours1' + file_name, contoursImage1)
        contours.append(contours1)                              #保存  标准图像减待检测图像  得到的轮廓

        sub2 = cv2.subtract(gray1, gray2)                       #待检测图像减设置的标准图像
        # sub2 = cv2.scaleAdd(sub2, 1.2, tmp)                   #调整图像灰度范围  改变参数可以调整图像对比度
        _, binary2 = cv2.threshold(sub2, 50, 255, cv2.THRESH_BINARY)
        image, contours2, hierarchy = cv2.findContours(binary2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.append(contours2)                       #保存  待检测图像减标准图像  得到的轮廓

        count = len(contours)
        count1 = len(contours[0]) + len(contours[1])
        num.append(count1)
        print("****************")
        for i in range(0, count):
            for j in range(0, len(contours[i])):
                contourLength = cv2.arcLength(contours[i][j], False)  #轮廓长度
                if contourLength > 500:
                    length[number].append(contourLength)            #保存轮廓长度大于500的轮廓长度
                    contoursTemp.append(contours[i][j])                #保存轮廓长度大于500的轮廓

        for i in range(0, len(contoursTemp)):                   #在符合长度条件的轮廓中选择符合面积条件的
            tmp = cv2.contourArea(contoursTemp[i])              #轮廓面积
            if tmp > 1000:
                area[number].append(tmp)                        #保存轮廓面积大于100的轮廓面积

        area[number].sort(reverse=True)
        contours = sorted(area[number], reverse=True)[:3]

        print(number)
        print(file_path)
        print(contours)

        if len(contours) == 0:  # 轮廓数目为0时表示没有车辆经过
            print("无车")
            area[number].append(0)
        # 轮廓数量大于0分为有车经过和有车静止两种情况
        elif len(contours) > 0:   # 当上一幅图像中轮廓的数量与当前图像轮廓数量相等且上一幅图像第一个轮廓面积与当前图像第一个轮廓面积相等，表明车是静止的
            if area[number][0] == area[number - 1][0] and len(area[number]) == len(area[number - 1]):
                print("有车静止")
            else:
                print("有车经过")
        number = number + 1
print(num)







