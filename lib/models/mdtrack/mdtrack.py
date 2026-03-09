"""
MDTrack Model
"""
import torch
import math
from torch import nn
import torch.nn.functional as F
from lib.models.mdtrack.encoder import build_encoder
from .decoder import build_decoder
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed
from .neck import build_neck
from collections import OrderedDict
from .MoE_fusion import BiMixtureOfAdapters
from .neck import *
import numpy as np

class MDTrack(nn.Module):
    """ This is the base class for MDTrack """
    def __init__(self, encoder, decoder, neck,cfg,
                 num_frames=1, num_template=1, decoder_type="CENTER"):
        """ Initializes the model.
        Parameters:
            encoder: torch module of the encoder to be used. See encoder.py
            decoder: torch module of the decoder architecture. See decoder.py
        """
        super().__init__()
        self.encoder = encoder
        self.decoder_type = decoder_type
        self.neck = neck

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.decoder = decoder

        self.num_frames = num_frames
        self.num_template = num_template
        self.freeze_en = cfg.TRAIN.FREEZE_ENCODER
        self.interaction_indexes = cfg.MODEL.ENCODER.INTERACTION_INDEXES

        self.merge_injector = merge_Injector(d_model=512)

        # self.moe_fusion = BiMixtureOfAdapters(dim=cfg.MODEL.NECK.D_MODEL,
        #                                         r=16,task_num=3)

        # self.task_index = cfg.DATA.TASK_INDEX


    def forward(self, template_list=None, search_list=None, template_anno_list=None,enc_opt=None,neck_h_state=None,neck_xh_state=None, feature=None,  mode="encoder",task_index=None):
        """
        image_list: list of template and search images, template images should precede search images
        xz: feature from encoder
        seq: input sequence of the decoder
        mode: encoder or decoder.
        """
        if mode == "encoder":
            return self.forward_encoder(template_list, search_list, template_anno_list)
        elif mode == "neck":
            return self.forward_neck(enc_opt,neck_h_state,neck_xh_state,task_index)
        elif mode == "decoder":
            return self.forward_decoder(feature)
        else:
            raise ValueError

    def forward_encoder(self, template_list, search_list, template_anno_list):
        # Forward the encoder
        num_template = len(template_list)
        num_search = len(search_list)

        rgb_template_list = []
        x_template_list = []
        rgb_search_list = []
        x_search_list = []

        for i in range(num_template):
            rgb_template_list.append(template_list[i][:,:3,:,:])
            x_template_list.append(template_list[i][:,3:,:,:])
        
        # for i in range(num_search):
        #     if np.random.rand() < 0.2:
        #         rgb_search_list.append(search_list[i][:,:3,:,:])
        #     x_search_list.append(search_list[i][:,3:,:,:])

        p_missing = 0.0          # 触发“缺失注入”的概率
        p_rgb_black = 0.5        # 在触发时，RGB置黑的概率；否则X置黑

        for i in range(num_search):
            # search_list[i]: Tensor on GPU, shape [B, C, H, W], C>=4
            rgb = search_list[i][:, :3, :, :].clone()
            x   = search_list[i][:, 3:, :, :].clone()

            # # 用 torch.rand，保证在同一 device 上生成随机数
            if torch.rand((), device=rgb.device) < p_missing:
                if torch.rand((), device=rgb.device) < p_rgb_black:
                    # rgb.zero_()   # RGB 全黑
                    rgb = x.clone()
                else:
                    # x.zero_()     # X(TIR等) 全黑
                    x = rgb.clone()

            rgb_search_list.append(rgb)
            x_search_list.append(x)


        xz = self.encoder(rgb_template_list, rgb_search_list, template_anno_list)
        x_xz = self.encoder(x_template_list, x_search_list,  template_anno_list)
        xzxz = torch.cat([xz, x_xz], dim=1)
        return xzxz
    def forward_neck(self,enc_out,neck_h_state,neck_xh_state,task_index):
        x = enc_out
        b,l,c = x.shape
        l = l//2
        xs = x[:, 0:self.num_patch_x]
        x_xs = x[:, l:l+self.num_patch_x]

        x,xs,xxs,h,xh = self.neck(x,xs,x_xs,neck_h_state,neck_xh_state,self.encoder.body.blocks,self.interaction_indexes)
        x = self.encoder.body.fc_norm(x)
        
        xs = xs + x[:, 0:self.num_patch_x]
        xxs = xxs + x[:, l:l+self.num_patch_x]
        xs = xs + xxs
        # xs = self.merge_injector(xs + xxs,xs,xxs)
        # xs,mloss = self.moe_fusion(xs,x_xs,task_index)
        return x,xs,h,xh

    def forward_decoder(self, feature, gt_score_map=None):
        # feature = feature[0]
        # feature = feature[:,0:self.num_patch_x * self.num_frames] # (B, HW, C)
        bs, HW, C = feature.size()
        if self.decoder_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
        if self.decoder_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.decoder(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.decoder_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        elif self.decoder_type == "MLP":
            # run the mlp head
            score_map, bbox, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

def build_mdtrack(cfg):
    encoder = build_encoder(cfg)
    neck = build_neck(cfg,encoder)
    decoder = build_decoder(cfg, neck)
    model = MDTrack(
        encoder,
        decoder,
        neck,
        cfg,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
    )

    # pretrained_file = '/disk3/wsl/disk1/wsl_workspace/MDTrack/pretrained/vitb224/MCITRACK_ep0300.pth.tar'
    # checkpoint = torch.load(pretrained_file, map_location="cpu")
    # missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
    # print(missing_keys, unexpected_keys)
    pretrained_file = '/disk1/wsl_workspace/MDTrack/pretrained/vitb224/MCITRACK_ep0300.pth.tar'
    checkpoint = torch.load(pretrained_file, map_location="cpu")
    state_dict = checkpoint["net"]

    # ❌ 直接丢掉 pos_embed
    # state_dict.pop("encoder.body.pos_embed", None)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    print("✅ Loaded pretrained weights without pos_embed")

    # print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
