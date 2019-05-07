import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLossClassify(nn.Module):
    def __init__(self, num_classes, background=0, cuda=1):
        super(FocalLossClassify, self).__init__()
        self.num_cls = num_classes + background
        self.one_hot = torch.eye(num_classes + background)
        if cuda:
            self.one_hot = self.one_hot.cuda()

    def focal_loss(self, x, y):
        alpha = 0.25
        gamma = 2
        t = self.one_hot[y.data, :]
        t = Variable(t)
        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        w = w.detach()
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)
    
    def forward(self, cls_pred, cls_truth):
        batch, C, L, W = cls_pred.size()
        cls_pred = cls_pred.permute(0,2,3,1).contiguous().view(batch,-1,self.num_cls)
        cls_truth = cls_truth.permute(0,2,3,1).contiguous().view(batch,-1)
        
        pos = cls_truth > 0
        num_obj = pos.data.sum()

        cls_loss = self.focal_loss(cls_pred, cls_truth)

        self.cls_loss = cls_loss.data
        self.loss = cls_loss/num_obj.float()
        return self.loss

class FocalLoss(nn.Module):
    def __init__(self, num_classes, box_len, num_anchor, background=1, cuda=1):
        super(FocalLoss, self).__init__()
        self.num_cls = num_classes + background
        self.box_len = box_len
        self.num_anchor = num_anchor
        self.one_hot = torch.eye(num_classes + background)
        if cuda:
            self.one_hot = self.one_hot.cuda()

    def focal_loss(self, x, y):
        alpha = 0.25
        gamma = 2
        t = self.one_hot[y.data, :]
        t = Variable(t)
        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        w = w.detach()
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)
    
    def forward(self, cls_pred, cls_truth, box_pred, box_truth):
        batch, C, L, W = cls_pred.size()
        box_pred = box_pred.permute(0,2,3,1).contiguous().view(batch,-1,self.box_len) 
        cls_pred = cls_pred.permute(0,2,3,1).contiguous().view(batch,-1,self.num_cls)
        box_truth = box_truth.permute(0,2,3,1).contiguous().view(batch,-1,self.box_len) 
        cls_truth = cls_truth.permute(0,2,3,1).contiguous().view(batch,-1)
        
        pos = cls_truth > 0
        num_obj = pos.data.sum()

        # box_loss = SmoothL1Loss(pos_box_pred, pos_box_targets)
        mask = pos.unsqueeze(2).expand_as(box_pred)      # [batch, anchors, 8]
        masked_box_pred = box_pred[mask].view(-1,self.box_len)      # [#pos,8]
        masked_box_truth = box_truth[mask].view(-1,self.box_len)    # [#pos,8]
        box_loss = F.smooth_l1_loss(masked_box_pred, masked_box_truth, size_average=False)

        # cls_loss = FocalLoss(loc_preds, loc_targets)
        pos_neg = cls_truth > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_pred)
        masked_cls_pred = cls_pred[mask].view(-1,self.num_cls)
        cls_loss = self.focal_loss(masked_cls_pred, cls_truth[pos_neg])

        self.cls_loss = cls_loss.data
        self.box_loss = box_loss.data
        self.loss = (cls_loss + box_loss)/num_obj.float()
        return self.loss