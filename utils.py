import os
import cv2
import numpy as np
# import matplotlib.pyplot as plt
from torchvision import transforms as T
from tqdm import tqdm
from PIL import Image
import torch
from torch.nn import functional as F
from dataset import Normalize,Resize,ToTensor

# trans = T.Compose([
#     T.ToPILImage(),
#     # T.Resize((224,224)),
#     T.ToTensor()
# ])

def trans(image):
    norm = Normalize()
    resize = Resize(352,352)
    toTensor = ToTensor()
    image = norm(image)
    image = resize(image)
    image = toTensor(image)
    image = image.unsqueeze(0)
    return image


def split_map(datapath):
    print(datapath)
    for name in os.listdir(datapath+'/DUTS-TE-Mask'):
        mask = cv2.imread(datapath+'/DUTS-TE-Mask/'+name,0)
        body = cv2.blur(mask, ksize=(5,5))
        body = cv2.distanceTransform(body, distanceType=cv2.DIST_L2, maskSize=5)
        body = body**0.5

        tmp  = body[np.where(body>0)]
        if len(tmp)!=0:
            body[np.where(body>0)] = np.floor(tmp/np.max(tmp)*255)

        if not os.path.exists(datapath+'/DUTS-TE-Body/'):
            os.makedirs(datapath+'/DUTS-TE-Body/')
        cv2.imwrite(datapath+'/DUTS-TE-Body/'+name, body)

        if not os.path.exists(datapath+'/DUTS-TE-Detail/'):
            os.makedirs(datapath+'/DUTS-TE-Detail/')
        cv2.imwrite(datapath+'/DUTS-TE-Detail/'+name, mask-body)

def get_edge(datapath,data_type='TR'):
    for name in os.listdir(datapath + '/DUTS-'+data_type+'-Mask'):
        name_jpg = name.split('.')[0]+'.jpg'    # for image
        image = cv2.imread(datapath+'/DUTS-'+data_type+'-Image/'+name_jpg,1)
        mask = cv2.imread(datapath+'/DUTS-'+data_type+'-Mask/'+name,1)
        mask = np.array(np.where(mask>200 , 255, 0),dtype=np.uint8)
        edge = cv2.Canny(mask,128,300)

        c = np.array((mask / 255),dtype=np.uint8)
        cover = cv2.multiply(image,c)
        cover = cv2.GaussianBlur(cover,(3,3),0)
        # check_canny(cover)
        detail = cv2.add(cv2.Canny(cover,128,300),edge)


        # cv2.imshow('image', image)
        # cv2.imshow('mask', mask)
        # cv2.imshow('cover', cover)
        # cv2.imshow('edge', edge)
        # cv2.imshow('detail', detail)
        #
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if not os.path.exists(datapath+'/DUTS-'+data_type+'-Detail/'):
            os.makedirs(datapath+'/DUTS-'+data_type+'-Detail/')
        cv2.imwrite(datapath+'/DUTS-'+data_type+'-Detail/'+name, detail)

def check_canny(image):
    lowThreshold = 0
    max_lowThreshold = 400

    maxThreshold = 0
    max_maxThreshold = 400
    kernel_size = 3

    def canny_low_threshold(intial):
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        canny = cv2.Canny(blur, intial, maxThreshold)  # x是最小阈值,y是最大阈值
        cv2.imshow('canny', canny)

    def canny_max_threshold(intial):
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        canny = cv2.Canny(blur, lowThreshold, intial)  # x是最小阈值,y是最大阈值
        cv2.imshow('canny', canny)

    # img = cv2.imread('./image_pack/shusongdai.jpg', 0)  # 后面参数0即是以灰度读取
    # img = cv2.imread(img_path, 0)  # 后面参数0即是以灰度读取
    img = image  # 后面参数0即是以灰度读取

    cv2.namedWindow('canny', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.createTrackbar('Min threshold', 'canny', lowThreshold, max_lowThreshold, canny_low_threshold)
    cv2.createTrackbar('Max threshold', 'canny', maxThreshold, max_maxThreshold, canny_max_threshold)
    canny_low_threshold(0)

    if cv2.waitKey(0) == 27:  # 27是ESC键值
        cv2.destroyAllWindows()

def get_coarse(datapath,data_type='TR'):
    for name in tqdm(os.listdir(datapath + '/DUTS-' + data_type + '-Mask')):
        mask = cv2.imread(datapath + '/DUTS-' + data_type + '-Mask/' + name, 0)
        kernel = np.ones((19, 19), np.uint8)
        coarse = cv2.dilate(mask, kernel)
        coarse = cv2.GaussianBlur(coarse, (19, 19), 8, 8)

        # cv2.imshow('mask', mask)
        # cv2.imshow('coarse', coarse)
        #
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if not os.path.exists(datapath+'/DUTS-'+data_type+'-Coarse/'):
            os.makedirs(datapath+'/DUTS-'+data_type+'-Coarse/')
        cv2.imwrite(datapath+'/DUTS-'+data_type+'-Coarse/'+name, coarse)


def count_params(model, input_size=224):
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

def apply_heatmap(image,data,name='no name',is_fuse = False, is_low = False):
    show_data = torch.max(data, dim=1)[0]
    show_data = F.sigmoid(show_data).detach().cpu().numpy().squeeze() * 200

    show_data = cv2.normalize(show_data, show_data, 0, 200, cv2.NORM_MINMAX)
    if is_low:
        show_data[show_data <= 80] = 0
    else:
        show_data[show_data <= 90] = 0
    heatmap = cv2.applyColorMap(show_data.astype(np.uint8), cv2.COLORMAP_JET)


    img = cv2.addWeighted(cv2.resize(heatmap, (352, 352)), 0.6, image, 0.4, 0)  # 将热度图覆盖到原图

    # cv2.imshow(name, img)
    # if is_fuse:
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # branch = name.split('_')[0]
    # save_dir = 'result/GFN_test/'+branch+'/'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # cv2.imwrite(save_dir+name+'.png',img)

    save_dir = 'result/CRG_test/'+'stage'+name.split('_')[2]+'/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cv2.imwrite(save_dir+name+'.png',img)



    return img

def normPRED(d):
    ma = np.max(d)
    mi = np.min(d)

    dn = (d-mi)/(ma-mi)

    return dn



if __name__=='__main__':
    # split_map('data/DUTS/DUTS-TE')
    get_edge('data/DUTS/DUTS-TR',data_type='TR')       # for TR
    # get_edge('data/DUTS/DUTS-TE',data_type='TE')       # for TE
    # net = Dnet()
    # count_params(net)
