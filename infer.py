from torch.nn import functional as F
from model.GPONet import GPONet
import torch
import os
from utils import trans
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image




net = GPONet()
net.eval()
model_dir = 'save_models/GPONet_t.pth'
ck = torch.load(model_dir, map_location='cpu')
ck_new = {k.replace('module.',''):v for k,v in ck.items()}
net.load_state_dict(ck_new)


testImg_dir = 'test_images/'
img_path_list= [testImg_dir+p for p in os.listdir(testImg_dir)]

def getPred(pred,size=(352,352)):
    pred = F.interpolate(pred, size=size, mode='bilinear')
    pred = torch.sigmoid(pred)
    pred = pred.squeeze().cpu().detach().numpy()
    return pred


def infer(img_path):
    image_name = img_path.split('/')[-1].split('.')[0]
    img = cv2.imread(img_path, 1)
    image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    h, w = img.shape[0], img.shape[1]
    img_t = trans(img).float()

    with torch.inference_mode():
        gb_pre,dt_pre,fuse_pre = net(img_t)
    gb_pre_7, gb_pre_14, gb_pre_28, gb_pre_56 = getPred(gb_pre[3],(h,w)),\
                                                getPred(gb_pre[2],(h,w)),\
                                                getPred(gb_pre[1],(h,w)),\
                                                getPred(gb_pre[0],(h,w))
    dt_pre_7, dt_pre_14, dt_pre_28, dt_pre_56 = getPred(dt_pre[3], (h, w)), \
                                               getPred(dt_pre[2], (h, w)), \
                                               getPred(dt_pre[1], (h, w)), \
                                               getPred(dt_pre[0], (h, w))
    pred = getPred(fuse_pre,(h,w))

    gb_pred_list = [gb_pre_7, gb_pre_14, gb_pre_28, gb_pre_56]
    dt_pred_list = [dt_pre_7, dt_pre_14, dt_pre_28, dt_pre_56]
    plot(image_name,image_pil, gb_pred_list, dt_pred_list, pred)

    return gb_pre_56,dt_pre_56,pred

def plot(image_name, image, gb_pred, eg_pred, final_pred, save_img=True):
    # 创建一个包含9个子图的大图，2行4列布局
    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(3, 4, figure=fig)


    for i, gb in enumerate(gb_pred):
        # 第一个子图
        ax = fig.add_subplot(gs[0, i])
        ax.set_title(f'Global Stage {4-i} Output')
        # 添加你的第一个子图内容
        ax.imshow(gb, cmap='viridis')  # cmap参数用于设置颜色映射
        ax.axis('off')  # 显示横纵坐标

    for i, eg in enumerate(eg_pred):
        # 第一个子图
        ax = fig.add_subplot(gs[1, i])
        ax.set_title(f'Detail Stage {4 - i} Output')
        # 添加你的第一个子图内容
        ax.imshow(eg, cmap='viridis')  # cmap参数用于设置颜色映射
        ax.axis('off')  # 显示横纵坐标

    # 处理最后一行，使其只包含一个子图，位于整个画布的中间位置，并等比例放大
    ax = fig.add_subplot(gs[2, :2])  # 占据整个第三行
    ax.axis('off')  # 显示横纵坐标
    ax.set_title('Original Image')  # 设置标题
    ax.imshow(image)  # 添加内容，并等比例放大

    ax = fig.add_subplot(gs[2, 2:])  # 占据整个第三行
    ax.axis('off')  # 显示横纵坐标
    ax.set_title('Final Saliency Map')  # 设置标题
    ax.imshow(final_pred, cmap='viridis')  # 添加内容，并等比例放大

    # 调整子图之间的间距
    plt.tight_layout()
    if save_img:
        save_path = 'save_infer_results'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'{image_name}.png'), bbox_inches='tight')

    plt.show()

infer(img_path='test_images/vangogh.jpeg')