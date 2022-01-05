import cv2
import os

# carPath = "D:/Data/Image/beijing_bohaidongdt23/JPEGImages/"
# imagepath1 = carPath + "beijing_bohaidongdt18_00801.jpg"

path = "D:/Data/Image/beijing_bohaidongdt23-day/railpic/"
imagepath2 = path + "beijing_bohaidongdt23_00031.jpg"
image2 = cv2.imread(imagepath2)
height2, width2, channels2 = image2.shape
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

dir_list = [
    "D:/Data/image/beijing_bohaidongdt23-day/test"
]

dist_dir = "D:/Data/result"
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
    area = [[] for index in range(size + 1)]
    area[0].append(0)
    number = 1

    length = len(file_list)
    for file_name in file_list:
    # for i in range(0, length - 1):
        file_path = dir + '/' + file_name
        # file_path1 = dir + '/' + file_list[i]
        image1 = cv2.imread(file_path)
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

        # height1, width1, channels1 = image1.shape
        # file_path2 = dir + '/' + file_list[i + 1]
        # image2 = cv2.imread(file_path2)
        # gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        sub = cv2.subtract(gray2, gray1)
        _, binary = cv2.threshold(sub, 80, 255, cv2.THRESH_BINARY)
        # binary = cv2.adaptiveThreshold(sub, 128, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -30)
        # height2, width2, channels2 = image1.shape

        cv2.imwrite(dist_dir + '/' + 'sub' +file_path, binary)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # dst = cv2.erode(binary,kernel)
        dst = cv2.dilate(binary, kernel)
        image, contours, hierarchy = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cropImage1 = cv2.drawContours(image2, contours, -1, (0, 255, 0), 8)
        cv2.imwrite(dist_dir + '/' + file_path, cropImage1)
        count = len(contours)
        num.append(count)

        for i in range(0, count - 1):
            tmp = cv2.contourArea(contours[i])
            # area[number].append(tmp)
            if tmp > 1000:
                area[number].append(tmp)

        area[number].sort(reverse=True)
        print(number)
        print(file_path)
        if len(area[number]) != 0:
            print(max(area[number]))
        print(area[number])
        contours = sorted(area[number], reverse=True)[:3]
        print(contours)

        if len(area[number]) == 0:  # 轮廓数目为0时表示没有车辆经过
            print("无车")
            area[number].append(0)
        # 当轮廓数量大于0且上一幅图像中轮廓的面积与当前图像轮廓面积相等，则表明是有车，但车是静止的
        elif (len(area[number]) > 0 and area[number][0] == area[number - 1][0] and
                len(area[number]) == len(area[number - 1])):
            print("有车静止")
        elif len(area[number]) > 0:  # 轮廓数量大于0表示有车经过
            print("有车经过")





        number = number + 1

print(num)
