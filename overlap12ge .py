from PIL import Image
import os
#https://blog.csdn.net/weixin_40446557/article/details/102913984?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control
######## 需要裁剪的图片位置#########
#path_img = 'D:/shuju/dataset363/label/'
path_img = 'E:/wlj/吉林水厂/20210628/1/10倍典型样/'
img_dir = os.listdir(path_img)
#print(img_dir)

'''
（左上角坐标(x,y)，右下角坐标（x+w，y+h）
'''

for i in range(len(img_dir)):
    #####根据图片名称提取id,方便重命名###########
    id = int((img_dir[i].split('.')[0]).split('_')[0])
    img = Image.open(path_img + img_dir[i])
    #img = img.resize((2048,2048))
    size_img = img.size
    #print(size_img)
    x = 0
    y = 0
    overlap_factor = 200
    factor = 2
    #########这里需要均匀裁剪几张，就除以根号下多少，这里我需要裁剪25张-》根号25=5（5*5）####
    w = int((size_img[0] / 4) + 150)
    h = int((size_img[1] / 3) + 133)
    img_stacks = []


    #
    #
    # for k in range(2):
    #     for v in range(2):
    #         region = img.crop((x + k * w, y + v * h, x + w * (k + 1), y + h * (v + 1)))
    #         #####保存图片的位置以及图片名称###############
    #         region.save('./1/'  + '%d' % id + '_%d%d' % (v,k)  + '.png')
#https://blog.csdn.net/weixin_30872867/article/details/97560292?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.control&dist_request_id=&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.control
    for r in range(3):
        for c in range(4):

            Lx = (c * w) - overlap_factor * c
            Ly = (r * h) - overlap_factor * r

            if (Lx <= 0):
                Lx = 0

            if (Ly <= 0):
                Ly = 0

            Rx = Lx + w
            Ry = Ly + h
            box = (Lx, Ly, Rx, Ry)
            region = img.crop(box)
            region.save('E:/wlj/data/jilin_1_10_dianxing_12/' + '%d' % id + '_%d%d' % (r, c) + '.png')