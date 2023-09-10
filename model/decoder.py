import torch
import torch.nn.functional as F
import torch.nn as nn

class GFN(nn.Module):
    def __init__(self,in_channel_list,out_channel):
        super(GFN, self).__init__()
        self.oc = out_channel
        self.squ_1 = nn.Sequential(nn.Conv2d(in_channel_list[0], out_channel, 1), nn.BatchNorm2d(out_channel),
                                      nn.ReLU(inplace=True))
        self.squ_2 = nn.Sequential(nn.Conv2d(in_channel_list[1], out_channel, 1), nn.BatchNorm2d(out_channel),
                                     nn.ReLU(inplace=True))
        self.squ_3 = nn.Sequential(nn.Conv2d(in_channel_list[2], out_channel, 1), nn.BatchNorm2d(out_channel),
                                     nn.ReLU(inplace=True))
        self.squ_4 = nn.Sequential(nn.Conv2d(in_channel_list[3], out_channel, 1), nn.BatchNorm2d(out_channel),
                                     nn.ReLU(inplace=True))

        self.CNR1 = nn.Sequential(nn.Conv2d(out_channel,out_channel,3,padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.CNR2 = nn.Sequential(nn.Conv2d(out_channel,out_channel,3,padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.CNR3 = nn.Sequential(nn.Conv2d(out_channel,out_channel,3,padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.CNR4 = nn.Sequential(nn.Conv2d(out_channel,out_channel,3,padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))

        self.GFM1 = GFM(out_channel)
        self.GFM2 = GFM(out_channel)
        self.GFM3 = GFM(out_channel)


    def forward(self,input_list):
        c1,c2,c3,c4 = input_list[0].shape[1], input_list[1].shape[1], input_list[2].shape[1], input_list[3].shape[1]
        if c1 != c2 or c3 != self.oc:
            f1, f2, f3, f4 = self.squ_1(input_list[0]), self.squ_2(input_list[1]), self.squ_3(input_list[2]), self.squ_4(input_list[3])
        else:
            f1,f2,f3, f4 = input_list[0], input_list[1], input_list[2], input_list[3]

        out1 = f1
        out2 = self.GFM1(out1,f2)
        out3 = self.GFM2(out2,f3)
        out4 = self.GFM3(out3,f4)

        out1 = self.CNR1(out1)
        out2 = self.CNR2(out2)
        out3 = self.CNR3(out3)
        out4 = self.CNR4(out4)

        return (out1,out2,out3,out4)

class GFM(nn.Module):
    def __init__(self,out_channel):
        super(GFM, self).__init__()
        self.gate1 = nn.Sequential(nn.Conv2d(out_channel,out_channel,3,padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))         # for lower feat        [B, out_channel, M, N]
        self.gate2 = nn.Sequential(nn.Conv2d(out_channel,out_channel,3,padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))         # for higher feat       [B, out_channel, M, N]
        self.convAct1 = nn.Sequential(nn.Conv2d(out_channel,out_channel,3,padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))         # for higher feat       [B, out_channel, M, N]
        self.convAct2 = nn.Sequential(nn.Conv2d(out_channel,out_channel,3,padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))         # for higher feat       [B, out_channel, M, N]
        self.convLack = nn.Sequential(nn.Conv2d(out_channel,out_channel,3,padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))         # for higher feat       [B, out_channel, M, N]
        # self.norm = nn.BatchNorm2d(out_channel)

    def forward(self, higher_feat, lower_feat):
        size = lower_feat.shape[2:]

        f1 = F.interpolate(higher_feat,size=size,mode='bilinear')
        g1 = self.gate1(f1)         # g1 for higher_feat
        act1 = self.convAct1(g1 * f1)

        g2 = self.gate2(lower_feat)          # g2 for lower_feat
        act2 = self.convAct2(g2 * lower_feat)

        lack = self.convLack((1 - g2) * act1)

        f2 = lower_feat + act2 + lack

        return f2


class CRG(nn.Module):
    def __init__(self,in_channel):
        super(CRG, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channel,in_channel,3,padding=1),
                              nn.BatchNorm2d(in_channel),
                              nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channel,in_channel,3,padding=1),
                              nn.BatchNorm2d(in_channel),
                              nn.ReLU(inplace=True))
        self.conv2_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, padding=1),
                                     nn.BatchNorm2d(in_channel),
                                     nn.ReLU(inplace=True))
        self.conv2_2 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, padding=1),
                                     nn.BatchNorm2d(in_channel),
                                     nn.ReLU(inplace=True))
        self.conv3_1 = nn.Sequential(nn.Conv2d(in_channel*2, in_channel, 3, padding=1),
                                     nn.BatchNorm2d(in_channel),
                                     nn.ReLU(inplace=True))
        self.conv3_2 = nn.Sequential(nn.Conv2d(in_channel*2, in_channel, 3, padding=1),
                                     nn.BatchNorm2d(in_channel),
                                     nn.ReLU(inplace=True))
        self.conv4_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, padding=1),
                                     nn.BatchNorm2d(in_channel),
                                     nn.ReLU(inplace=True))
        self.conv4_2 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, padding=1),
                                     nn.BatchNorm2d(in_channel),
                                     nn.ReLU(inplace=True))


    def forward(self,gb,dt):
        dt1, gb1 = self.conv1_1(dt), self.conv1_2(gb)
        dt2, gb2 = self.conv2_1(dt1), self.conv2_2(gb1)
        cro_feat = torch.cat((dt2,gb2),dim=1)
        dt3, gb3 = self.conv3_1(cro_feat), self.conv3_2(cro_feat)
        f1 = dt1 + dt3
        f2 = gb1 + gb3
        dt4 = self.conv4_1(f1)
        gb4 = self.conv4_2(f2)

        return gb4, dt4

class Decoder(nn.Module):
    def __init__(self,in_channel_list, out_channel):
        super(Decoder, self).__init__()

        # Gate融合
        self.gfn_gb = GFN(in_channel_list, out_channel)
        self.gfn_eg = GFN(in_channel_list, out_channel)

        # cross guide
        self.crg1 = CRG(out_channel)
        self.crg2 = CRG(out_channel)
        self.crg3 = CRG(out_channel)
        self.crg4 = CRG(out_channel)

    def forward(self,gb_list,eg_list):
        # feat_list = [f_7, f_14, f_28, f_56]

        # gate fuse out
        fuse_gb_7, fuse_gb_14, fuse_gb_28, fuse_gb_56 = self.gfn_gb(gb_list)  # (B, gfm_channel, 56/28/14/7, 56/28/14/7)
        fuse_eg_7, fuse_eg_14, fuse_eg_28, fuse_eg_56 = self.gfn_eg(eg_list)  # (B, gfm_channel, 56/28/14/7, 256/28/14/78)

        fuse_gb_7, fuse_eg_7 = self.crg1(fuse_gb_7, fuse_eg_7)
        fuse_gb_14, fuse_eg_14 = self.crg2(fuse_gb_14, fuse_eg_14)
        fuse_gb_28, fuse_eg_28 = self.crg3(fuse_gb_28, fuse_eg_28)
        fuse_gb_56, fuse_eg_56 = self.crg4(fuse_gb_56, fuse_eg_56)

        out_gb_feat = [fuse_gb_7,fuse_gb_14,fuse_gb_28, fuse_gb_56]
        out_eg_feat = [fuse_eg_7,fuse_eg_14,fuse_eg_28, fuse_eg_56]

        return out_gb_feat,out_eg_feat