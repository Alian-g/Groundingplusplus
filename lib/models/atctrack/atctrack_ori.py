"""
ATCTrack  Model
"""
import os

import torch
import math
from torch import nn
import torch.nn.functional as F

from lib.utils.misc import NestedTensor

# from .language_model import build_bert
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_iou
### aqatrack
from lib.models.aqatrack.hivit import hivit_small, hivit_base
from lib.models.aqatrack.itpn import itpn_base_3324_patch16_224
from lib.models.aqatrack.fast_itpn import fast_itpn_base_3324_patch16_224,fast_itpn_large_2240_patch16_256

from lib.models.transformers.transformer import build_rgb_det_decoder
from lib.models.layers.transformer_dec import build_transformer_dec,build_transformer_dec_with_mask

from torch.nn.modules.transformer import _get_clones
from lib.models.layers.head import build_box_head

import torch.nn.functional as F
from lib.models.layers.frozen_bn import FrozenBatchNorm2d
from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizerFast
from lib.models.transformers import build_decoder, VisionLanguageFusionModule, PositionEmbeddingSine1D,build_text_prompt_decoder
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

from pathlib import Path
from torchvision.utils import save_image
from typing import List
def expand_exps_cycle(exps: List[str], N: int) -> List[str]:
    """逐帧循环：第 i 张用 exps[i % len(exps)]"""
    return [exps[i % len(exps)] for i in range(N)]
def save_search_batch(search: torch.Tensor, out_root=".vis",
                      img_mean=(0.485, 0.456, 0.406),
                      img_std=(0.229, 0.224, 0.225),
                      assume_imagenet_norm=True,
                      prefix="search"):
    """
    search: [N, 3, H, W] in RGB, CHW
    out_root: output folder
    assume_imagenet_norm: 若为 True，则按 ImageNet 均值方差做反归一化；否则仅 clamp 到 [0,1]
    """
    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = search.detach().clone().cpu()

    if assume_imagenet_norm:
        mean = torch.tensor(img_mean).view(1, 3, 1, 1)
        std  = torch.tensor(img_std).view(1, 3, 1, 1)
        imgs = imgs * std + mean  # 反归一化到 [0,1] 左右
    else:
        # 如果你的张量本来就在 [0,1]，或者在 [0,255]，可视情况处理：
        # 若在 [0,255]，请先 imgs = imgs / 255.0
        pass

    imgs = imgs.clamp(0.0, 1.0)

    # 逐张保存 & 再额外存一个九宫格图
    for i in range(imgs.size(0)):
        save_image(imgs[i], out_dir / f"{prefix}_{i:04d}.png")

    # 可选：保存一个拼图（例如一行 10 张）
    try:
        save_image(imgs, out_dir / f"{prefix}_grid.png", nrow=10)
    except Exception:
        pass

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
class ConfidencePred(nn.Module):
    def __init__(self):
        super(ConfidencePred, self).__init__()
        self.feat_sz = 24
        self.stride = 1
        self.img_sz = self.feat_sz * self.stride
        freeze_bn = False

        # CNN
        self.conv1_ctr = conv(5, 16, freeze_bn=freeze_bn)
        self.conv2_ctr = conv(16, 16 // 2, freeze_bn=freeze_bn)
        self.conv3_ctr = conv(16 // 2, 16 // 4, freeze_bn=freeze_bn)
        self.conv4_ctr = conv(16 // 4, 16 // 8, freeze_bn=freeze_bn)
        self.conv5_ctr = nn.Conv2d(16 // 8, 1, kernel_size=1)

        # 定义全连接层
        self.fc1 = nn.Linear(256, 512)

        ## cross attn 交互层
        # self.multihead_attn = nn.MultiheadAttention(512, 4, dropout=0.1)
        # # Implementation of Feedforward model
        # self.dropout = nn.Dropout(0.1)
        # self.norm1 = nn.LayerNorm(512)


        self.fc2 = nn.Linear(512, 1)

        # 定义激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x,xz_feature=None, gt_score_map=None):
        """ Forward pass with input x. """

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map_ctr = self.conv5_ctr(x_ctr4)

        # 展平输入
        x = score_map_ctr.flatten(1)
        x = self.relu(self.fc1(x))

        x = self.sigmoid(self.fc2(x))

        return x

class SubjectIndexPred(nn.Module):
    def __init__(self,dim):
        super(SubjectIndexPred, self).__init__()

        # 定义全连接层
        self.fc1 = nn.Linear(dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass with input x. """

        # 全连接层前向传播
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))

        return x


class ATCTrack(nn.Module):
    """ This is the base class for ATCTrack"""
    def __init__(self, transformer,  box_head, tokenizer, text_encoder, aux_loss=False, head_type="CORNER",dim=512,cfg=None):
        """ Initializes the model.
        Parameters:
            encoder: torch module of the encoder to be used. See encoder.py
            decoder: torch module of the decoder architecture. See decoder.py
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        self.dim = dim

        self.query_len = 1
        self.cls_prompts_pos = nn.Embedding(num_embeddings=self.query_len, embedding_dim=self.dim )  # pos for cur query
        # self.cls_initial= nn.Embedding(num_embeddings=self.query_len, embedding_dim=self.dim )  # pos for cur query
        self.confidence_pred = ConfidencePred()

        ### visual temporal
        self.visual_temporal_fusion = build_transformer_dec_with_mask(cfg, self.dim )
        self.temporal_len = 4
        self.dy_template_pos_embed = nn.Embedding(num_embeddings=self.temporal_len,
                                                  embedding_dim=self.dim )  # pos for cur query

        ## invlove_text
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.text_adj = nn.Sequential(
            nn.Linear(768, self.dim , bias=True),
            nn.LayerNorm(self.dim , eps=1e-12),
            nn.Dropout(0.1),
        )

        self.language_adjust = build_transformer_dec(cfg, self.dim )
        self.vl_fusion = VisionLanguageFusionModule(dim=self.dim , num_heads=8, attn_drop=0.1, proj_drop=0.1,
                                                    num_vlfusion_layers=2,
                                                    vl_input_type='separate')

        self.text_pos = PositionEmbeddingSine1D(self.dim , normalize=True)

        self.text_sub_idnex_classifier = SubjectIndexPred(self.dim)

        dargs = SLConfig.fromfile("/data/ATCTrack/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        dargs.device = "cuda" 
        model_d = build_model(dargs)
        checkpoint = torch.load("/data/ATCTrack/groundingdino/weights/groundingdino_swint_ogc.pth", map_location="cpu")
        load_res = model_d.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        self.gdino = model_d.eval()
        for p in self.gdino.parameters():
            p.requires_grad = False
    def forward_backbone(self, template, search, cls_token,soft_token_template_mask,x_pos):
        # template b, 12, h,w
        # search b,6,h,w
        template = [template[:,:3],template[:,3:]]
        soft_token_template_mask = [soft_token_template_mask[:, :64], soft_token_template_mask[:, 64:]]

        x, token_type_infor = self.backbone.forward_features_pe(z=template, x=search, soft_token_template_mask =soft_token_template_mask)
        x, aux_dict = self.backbone.forward_features_stage3(x, cls_token,x_pos)
        return x, aux_dict

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                soft_token_template_mask=None,
                exp_str=None,
                exp_subject_mask=None,
                temporal_infor=[],
                first_frame_flag=False,
                training=True):

        b0, num_search = template[0].shape[0], len(search)
        if training:
            search = torch.cat(search, dim=0)
            template = torch.cat(template, dim=1)  # (bs,6(rgb0;rgb1),w,h)
            #save_search_batch(search, out_root=".vis", assume_imagenet_norm=True, prefix="search")
            with torch.no_grad():
                out = self.gdino(search,captions = expand_exps_cycle(exp_str,b0*num_search))
            soft_token_template_mask = torch.cat(soft_token_template_mask,
                                                              dim=1)  # (bs,128(mask0;mask1),1)
            template_temporal = []
            soft_token_template_mask_temporal = []
            for _ in range(num_search):
                template_temporal.append(template)
                soft_token_template_mask_temporal.append(soft_token_template_mask)
            template_temporal = torch.cat(template_temporal, dim=0)
            soft_token_template_mask_temporal = torch.cat(soft_token_template_mask_temporal,dim=0)

        else:
            b0 = 1
            template_temporal = torch.cat(template, dim=1)
            soft_token_template_mask_temporal = torch.cat(soft_token_template_mask, dim=1)

        # x, aux_dict = self.backbone(z=template, x=search,
        #                             soft_token_template_mask = soft_token_template_mask )
        cls_prompts_pos = self.cls_prompts_pos.weight.unsqueeze(0)
        x_pos_0 = torch.cat([cls_prompts_pos, self.backbone.pos_embed_z, self.backbone.pos_embed_x], dim=1)
        # pos_embed = x_pos.transpose(0, 1).repeat(1, b0, 1)
        x_pos = x_pos_0.repeat(b0*num_search, 1, 1)
        x, aux_dict = self.forward_backbone(template_temporal, search, None, soft_token_template_mask_temporal,
                                                 x_pos)
        # forward Language branch
        if training:
            if exp_str:
                text_features, text_subject_features, subject_infor_mask_pred, subject_infor_mask_gt  = self.forward_text(
                    exp_str, num_search, exp_subject_mask, device=search.device)  # text_subject_features, subject_infor_mask_pred, subject_infor_mask_gt
        else:
            text_features = exp_str
            text_subject_features = exp_subject_mask
            subject_infor_mask_pred = None
            subject_infor_mask_gt = None
        batch_size = text_features.tensors.shape[0]
        text_pos = self.text_pos(text_features) # [batch_size, length, c]
        text_pos_0 = text_pos[:b0]
        x_s_pos_item = x_pos_0.repeat(b0, 1, 1)[:, -self.feat_len_s:]
        pre_temporal_pos = self.dy_template_pos_embed.weight.unsqueeze(1)
        pre_temporal_pos = pre_temporal_pos.repeat(b0, 1, self.query_len)
        pre_temporal_pos = pre_temporal_pos.view(b0, self.temporal_len * self.query_len, self.dim).contiguous()

        # Forward temporal
        xt_data = []
        for temporal_index in range(num_search):
            x_item = x[temporal_index * b0:(temporal_index + 1) * b0]

            visual_prompts_token = x_item[:, :self.query_len, :]

            ## heatmap by backbone feat
            ## by attn
            # attn_xz = attn[:, :, :-self.feat_len_s, -self.feat_len_s:]  #  b,h,l,l
            # attn_xz_1 = attn_xz.mean(1).mean(1)
            # # attn_xz = attn_xz.view(16, 16)
            # # attn_weights_debug = attn_xz.detach().cpu().numpy()
            x_f = x_item[:, -256:]
            x_f1 = torch.matmul(x_f, x_f.permute(0, 2, 1).contiguous())
            x_f = torch.matmul(x_f1, x_f)

            z_f = x_item[:, :-256]

            x_z = torch.matmul(x_f, z_f.permute(0, 2, 1).contiguous())
            att_map = x_z.mean(-1)

            tensor_min = torch.min(att_map)
            tensor_max = torch.max(att_map)
            # normalized_tensor = (s_vl_1 - tensor_min) / (tensor_max - tensor_min)
            normalized_tensor = (tensor_max - att_map) / (tensor_max - tensor_min)

            attn_xz = normalized_tensor.view(-1, 256,1).contiguous()

            ### initialize & update memory
            if training:
                if temporal_index == 0:
                    temporal_infor = []
                    for _ in range(self.temporal_len):
                        temporal_infor.append(visual_prompts_token)
            else:
                if first_frame_flag:
                    temporal_infor = []
                    for _ in range(self.temporal_len):
                        temporal_infor.append(visual_prompts_token)

            temporal_infor_data = torch.cat(temporal_infor, dim=1)

            #### vl fusion  ############
            ## L adjust
            l_item_initial = text_features.tensors[temporal_index * b0:(temporal_index + 1) * b0]
            l_item_subject = text_subject_features.tensors[temporal_index * b0:(temporal_index + 1) * b0]
            l_mask_item_0 = text_features.mask[temporal_index * b0:(temporal_index + 1) * b0]
            temporal_mask = torch.ones((l_mask_item_0.shape[0],self.temporal_len)).bool().to(l_mask_item_0.device)
            l_mask_item = torch.cat([l_mask_item_0, temporal_mask],dim=1)

            l_subject_temporal = torch.cat([l_item_subject,temporal_infor_data],dim=1)
            l_subject_temporal_pos = torch.cat([text_pos_0,pre_temporal_pos ],dim=1)

            l_item_update,_ = self.language_adjust([l_item_initial,l_subject_temporal],None,
                                          text_pos_0,l_subject_temporal_pos,l_mask_item)
            l_all = torch.cat([ l_item_initial,l_item_update ],dim=1)
            x_s_item = x_item[:, -self.feat_len_s:]
            x_s_item = self.vl_fusion(x_s_item,
                                 l_all,
                                 query_pos=x_pos_0[:, -self.feat_len_s:],
                                 memory_pos=torch.cat([text_pos_0,text_pos_0],dim=1),
                                 memory_key_padding_mask=torch.cat([l_mask_item_0,l_mask_item_0],dim=1),
                                 need_weights=False)


            #### cross_attention with temporal_infor
            temporal_infor_update = self.visual_temporal_fusion(temporal_infor_data, x_s_item, attn_xz,pre_temporal_pos ,kv_pos= x_s_pos_item )
            temporal_item = temporal_infor_update[:,-1,:].unsqueeze(1)

            # STM
            enc_opt = x_s_item
            dec_opt = temporal_item.transpose(1, 2)
            att = torch.matmul(enc_opt, dec_opt)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

            xt_data.append(opt_feat)

            ### update temporal infor
            if training:
                if temporal_index == 0:
                    temporal_infor = []
                    for _ in range(self.temporal_len):
                        temporal_infor.append(temporal_item)
                else:
                    temporal_infor[:-1] = temporal_infor[1:]
                    temporal_infor[-1] = temporal_item
            else:
                if first_frame_flag:
                    temporal_infor = []
                    for _ in range(self.temporal_len):
                        temporal_infor.append(temporal_item)

                else:
                    temporal_infor[:-1] = temporal_infor[1:]
                    temporal_infor[-1] = temporal_item


        # Forward head
        xt_data = torch.cat(xt_data,dim=0)
        out = self.forward_head(xt_data, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        out['subject_infor_mask_pred'] = subject_infor_mask_pred
        out['subject_infor_mask_gt'] = subject_infor_mask_gt

        if training == False:
            out["temporal_infor"] = temporal_infor

        return out

    def forward_head(self, opt_feat, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """

        # enc_opt = cat_feature #[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        # bs, Nq, C, HW = opt.size()
        # opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s).contiguous()

        bs = opt_feat.shape[0]
        Nq = 1
        # Head
        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4).contiguous()
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)

            score_map = torch.cat([score_map_ctr, size_map, offset_map], dim=1)
            confidence_pred = self.confidence_pred(score_map)

            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4).contiguous()
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map,
                   "confidence_pred": confidence_pred}
            return out
        else:
            raise NotImplementedError

    def forward_text(self, captions, num_search, exp_subject_mask, device):
        tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt").to(device)
        encoded_text = self.text_encoder(**tokenized)

        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        # text_attention_mask: [batch_size, length]

        text_features = encoded_text.last_hidden_state
        text_features = self.text_adj(text_features)

        encodings_infor = tokenized.encodings

        subject_infor_mask_gt = None
        if exp_subject_mask is not None:
            # train: given the exp_subject_mask, used for generating  sub_index_gt
            subject_infor_mask_gt = torch.zeros(text_attention_mask.shape[0], text_attention_mask.shape[1]).to(
                text_features.device)

            for item_index, item in enumerate(encodings_infor):
                word_ids_item = item.word_ids
                exp_subject_mask_item = exp_subject_mask[item_index]
                text_index_list = []
                for word_index, word_item in enumerate(word_ids_item):
                    if word_item in exp_subject_mask_item:
                        text_index_list.append(word_index)

                subject_infor_mask_gt[item_index, text_index_list] = 1

        subject_infor_mask_pred = self.text_sub_idnex_classifier(text_features)
        subject_infor_mask_pred_1 = subject_infor_mask_pred.expand_as(text_features)

        subject_infor = text_features * subject_infor_mask_pred_1

        # (B,L,D) to (T,B,L,D)
        text_features_t = []
        text_attention_mask_t = []
        text_subject_infor_t = []
        for i in range(num_search):
            text_features_t.append(text_features)
            text_attention_mask_t.append(text_attention_mask)
            text_subject_infor_t.append(subject_infor)

        text_features = torch.cat(text_features_t, dim=0)
        text_attention_mask = torch.cat(text_attention_mask_t, dim=0)
        text_features = NestedTensor(text_features, text_attention_mask)
        subject_infor = torch.cat(text_subject_infor_t, dim=0)
        subject_infor = NestedTensor(subject_infor, text_attention_mask)

        return text_features, subject_infor, subject_infor_mask_pred, subject_infor_mask_gt


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def build_atctrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../resource/pretrained_models')

    if cfg.MODEL.PRETRAIN_FILE  and training and ("ATCTrack" not in cfg.MODEL.PRETRAIN_FILE) :
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''


    if cfg.MODEL.BACKBONE.TYPE == 'hivit_base_adaptor':
        backbone = hivit_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'itpn_base':  # by this
        backbone = fast_itpn_base_3324_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'itpn_large':  # by this
        backbone = fast_itpn_large_2240_patch16_256(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg,dim=hidden_dim, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    # Build Text Encoder
    tokenizer = RobertaTokenizerFast.from_pretrained(
        os.path.join(pretrained_path, 'roberta-base'))  # load pretrained RoBERTa Tokenizer
    text_encoder = RobertaModel.from_pretrained(
        os.path.join(pretrained_path, 'roberta-base'))  # load pretrained RoBERTa model


    model = ATCTrack(
        backbone,
        box_head,
        tokenizer,
        text_encoder,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        dim = hidden_dim,
        cfg=cfg
    )

    if  ("ATCTrack" in cfg.MODEL.PRETRAINED_PATH) and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAINED_PATH, map_location="cpu")
        ckpt = checkpoint["net"]
        model_weight = {}
        for k, v in ckpt.items():
            model_weight[k] = v

        missing_keys, unexpected_keys = model.load_state_dict(model_weight, strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)


    return model

def load_pretrained(model, pretrained_path, strict=False):

    model_ckpt = torch.load(pretrained_path, map_location="cpu")
    state_dict = model_ckpt['net']
    pos_st = state_dict['encoder.body.pos_embed']
    pos_s = pos_st[:,:(pos_st.size(1) // 2)]
    pos_t = pos_st[:,(pos_st.size(1) // 2):]
    state_dict['encoder.body.pos_embed_search'] = pos_s
    state_dict['encoder.body.pos_embed_template'] = pos_t
    state_dict['encoder.body.patch_embed_interface.proj.weight'] = state_dict['encoder.body.patch_embed.proj.weight']
    state_dict['encoder.body.patch_embed_interface.proj.bias'] = state_dict['encoder.body.patch_embed.proj.bias']
    state_dict['decoder.embedding.prompt_embeddings.weight'] = model.state_dict()['decoder.embedding.prompt_embeddings.weight']
    state_dict['decoder.embedding.prompt_embeddings.weight'][:] = state_dict['decoder.embedding.word_embeddings.weight'][-1]
    del state_dict['encoder.body.pos_embed']
    model.load_state_dict(state_dict, strict=strict)
