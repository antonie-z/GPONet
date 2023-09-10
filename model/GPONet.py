from torch import nn
from torch.nn import functional as F
import torch
from model.backbone.Encoder_pvt import Encoder
from model.decoder import Decoder



class FFM(nn.Module):
    def __init__(self,fuse_channel=256):
        super(FFM, self).__init__()

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.fuse = nn.Sequential(nn.Linear(8, fuse_channel),
                                  nn.BatchNorm1d(fuse_channel),
                                  nn.ReLU(),
                                  nn.Linear(fuse_channel, 8))

    def forward(self,input):
        B = input.shape[0]

        # get fuse attention
        gap = self.GAP(input).squeeze(-1).squeeze(-1)
        fuse_att = self.fuse(gap).view(B, 8, 1, 1)

        # fuse from gb&dt out
        fuse = input * fuse_att.expand_as(input)

        return fuse

class GPONet(nn.Module):
    def __init__(self,backbone='pvt',d_channel=256, input_size=352):
        super(GPONet, self).__init__()
        if backbone == 'pvt':
            self.in_channel_list = [512, 320, 128, 64]
        elif backbone == 'resnet':
            self.in_channel_list = [2048, 1024, 512, 256]
        # backbone
        self.encoder = Encoder()

        # decoder
        self.decoder1 = Decoder(self.in_channel_list,d_channel)

        self.fm = FFM()

        # heads
        self.gb_head1 = nn.Conv2d(d_channel,1,3,padding=1)
        self.gb_head2 = nn.Conv2d(d_channel,1,3,padding=1)
        self.gb_head3 = nn.Conv2d(d_channel,1,3,padding=1)
        self.gb_head4 = nn.Conv2d(d_channel,1,3,padding=1)

        self.eg_head1 = nn.Conv2d(d_channel, 1, 3, padding=1)
        self.eg_head2 = nn.Conv2d(d_channel, 1, 3, padding=1)
        self.eg_head3 = nn.Conv2d(d_channel, 1, 3, padding=1)
        self.eg_head4 = nn.Conv2d(d_channel, 1, 3, padding=1)

        self.head = nn.Conv2d(8, 1, 3, padding=1)

        # loss
        self.edge_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.fuse_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.sigmoid = nn.Sigmoid()
        self.size_list_per_stage = [input_size//4, input_size//8, input_size//16, input_size//32]


    def forward(self, x):
        B = x.shape[0]
        s = self.size_list_per_stage

        f_1, f_2, f_3, f_4 = self.encoder(x)
        f_1 = f_1.permute(0, 2, 1).contiguous().view(B, self.in_channel_list[0], s[3], s[3])
        f_2 = f_2.permute(0, 2, 1).contiguous().view(B, self.in_channel_list[1], s[2], s[2])
        f_3 = f_3.permute(0, 2, 1).contiguous().view(B, self.in_channel_list[2], s[1], s[1])
        f_4 = f_4.permute(0, 2, 1).contiguous().view(B, self.in_channel_list[3], s[0], s[0])
        feat_list = [f_1, f_2, f_3, f_4]

        out_gb_feat, out_eg_feat = self.decoder1(feat_list, feat_list)
        gb_pred, eg_pred, cat_pred = self.get_stage_pred(out_gb_feat, out_eg_feat, x)

        fuse = torch.cat(cat_pred,dim=1)
        fuse = self.fm(fuse)
        fuse_pred = self.mlp(fuse, x, self.head)

        return gb_pred, eg_pred, fuse_pred

    def calcu_loss(self, x, eg, mask):
        gb_pred, eg_pred, fuse_pred = self.forward(x)
        # compute loss & mae
        loss = self.multi_loss_func(gb_pred, eg_pred, fuse_pred, eg, mask)
        mae = self.Eval_mae(fuse_pred, mask)
        return loss, mae



    def get_stage_pred(self, out_gb_feat, out_eg_feat, x):
        gb_pre_88 = self.mlp(out_gb_feat[3], x, self.gb_head1)
        gb_pre_44 = self.mlp(out_gb_feat[2], x, self.gb_head2)
        gb_pre_22 = self.mlp(out_gb_feat[1], x, self.gb_head3)
        gb_pre_11 = self.mlp(out_gb_feat[0], x, self.gb_head4)
        gb_pred = [gb_pre_88, gb_pre_44, gb_pre_22, gb_pre_11]
        eg_pre_88 = self.mlp(out_eg_feat[3], x, self.eg_head1)
        eg_pre_44 = self.mlp(out_eg_feat[2], x, self.eg_head2)
        eg_pre_22 = self.mlp(out_eg_feat[1], x, self.eg_head3)
        eg_pre_11 = self.mlp(out_eg_feat[0], x, self.eg_head4)
        eg_pred = [eg_pre_88, eg_pre_44, eg_pre_22, eg_pre_11]

        cat_pred = [gb_pre_88,gb_pre_44,gb_pre_22,gb_pre_11,eg_pre_88,eg_pre_44,eg_pre_22,eg_pre_11]

        return gb_pred, eg_pred, cat_pred

    def up_sample(self,src,target):
        size = target.shape[2:]
        out = F.interpolate(src,size=size,mode='bilinear')
        return out

    def multi_loss_func(self,gb,eg,fuse, detail,mask,ratio=[1,3,3]):
        alpha = 1+5*torch.abs(F.avg_pool2d(mask,kernel_size=3,stride=1,padding=1)-mask)
        beta = 1+5*torch.abs(F.avg_pool2d(detail,kernel_size=3,stride=1,padding=1)-detail)

        gb_loss = 0
        for i,gb_pre in enumerate(gb):
            gb_loss += self.iou_loss(gb_pre,mask,alpha)

        edge_loss = 0
        for i,eg_pre in enumerate(eg):
            edge_loss += (beta * self.edge_loss(eg_pre, detail)).mean()

        fuse_loss = (alpha * self.fuse_loss(fuse,mask)).mean()

        totoal_loss = gb_loss*ratio[0] + edge_loss*ratio[1] + fuse_loss*ratio[2]

        return totoal_loss

    def iou_loss(self,pre,mask,alpha):
        pre = torch.sigmoid(pre)
        inter = ((pre * mask)*alpha).sum(dim=(2,3))
        union = ((pre + mask)*alpha).sum(dim=(2,3))
        iou = 1 - (inter + 1) / (union - inter + 1)
        return iou.mean()

    def Eval_mae(self, pred, gt):
        totoal_mae, avg_mae, img_num, B = 0.0,0.0, 0.0, pred.shape[0]
        pred,gt = torch.sigmoid(pred).squeeze(1),gt.squeeze(1)
        with torch.no_grad():
            for b in range(B):
                totoal_mae += torch.abs(pred[b] - gt[b]).mean()
            avg_mae = totoal_mae/B

        return avg_mae

    # 各个分支的预测头
    def mlp(self,src,tar,head):
        B,_,H,W = tar.shape
        up = self.up_sample(src,tar)
        out = head(up)
        return out





