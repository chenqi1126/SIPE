import torch
import torch.nn as nn
import torch.nn.functional as F
from tool import torchutils
from network import resnet50


class Net(nn.Module):

    def __init__(self, num_cls=21):
        super(Net, self).__init__()

        self.num_cls = num_cls

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1), dilations=(1, 1, 1, 1))

        self.stage0 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool)
        self.stage1 = nn.Sequential(self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.side1 = nn.Conv2d(256, 128, 1, bias=False)
        self.side2 = nn.Conv2d(512, 128, 1, bias=False)
        self.side3 = nn.Conv2d(1024, 256, 1, bias=False)
        self.side4 = nn.Conv2d(2048, 256, 1, bias=False)
        self.classifier = nn.Conv2d(2048, self.num_cls-1, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage0, self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier, self.side1, self.side2, self.side3, self.side4])

    def get_seed(self, norm_cam, label, feature):
        
        n,c,h,w = norm_cam.shape

        # iou evalution
        seeds = torch.zeros((n,h,w,c)).cuda()
        feature_s = feature.view(n,-1,h*w)
        feature_s = feature_s/(torch.norm(feature_s,dim=1,keepdim=True)+1e-5)
        correlation = F.relu(torch.matmul(feature_s.transpose(2,1), feature_s),inplace=True).unsqueeze(1) #[n,1,h*w,h*w]
        # correlation = correlation/torch.max(correlation, dim=-1)[0].unsqueeze(-1) #[n,1,h*w,h*w]
        cam_flatten = norm_cam.view(n,-1,h*w).unsqueeze(2) #[n,21,1,h*w]
        inter = (correlation * cam_flatten).sum(-1)
        union = correlation.sum(-1) + cam_flatten.sum(-1) - inter
        miou = (inter/union).view(n,self.num_cls,h,w) #[n,21,h,w]
        miou[:,0] = miou[:,0]*0.5
        probs = F.softmax(miou, dim=1)
        belonging = miou.argmax(1)
        seeds = seeds.scatter_(-1, belonging.view(n,h,w,1), 1).permute(0,3,1,2).contiguous()
        
        seeds = seeds * label
        return seeds, probs
    
    def get_prototype(self, seeds, feature):
        n,c,h,w = feature.shape
        seeds = F.interpolate(seeds, feature.shape[2:], mode='nearest')
        crop_feature = seeds.unsqueeze(2) * feature.unsqueeze(1)  # seed:[n,21,1,h,w], feature:[n,1,c,h,w], crop_feature:[n,21,c,h,w]
        prototype = F.adaptive_avg_pool2d(crop_feature.view(-1,c,h,w), (1,1)).view(n, self.num_cls, c, 1, 1) # prototypes:[n,21,c,1,1]
        return prototype

    def reactivate(self, prototype, feature):
        IS_cam = F.relu(torch.cosine_similarity(feature.unsqueeze(1), prototype, dim=2)) # feature:[n,1,c,h,w], prototypes:[n,21,c,1,1], crop_feature:[n,21,h,w]
        IS_cam = F.interpolate(IS_cam, feature.shape[2:], mode='bilinear', align_corners=True)
        return IS_cam

    def forward(self, x, valid_mask):

        N, C, H, W = x.size()

        # forward
        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        side1 = self.side1(x1.detach())
        side2 = self.side2(x2.detach())
        side3 = self.side3(x3.detach())
        side4 = self.side4(x4.detach())

        hie_fea = torch.cat([F.interpolate(side1/(torch.norm(side1,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side2/(torch.norm(side2,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side3/(torch.norm(side3,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'),
                              F.interpolate(side4/(torch.norm(side4,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear')], dim=1)

        sem_feature = x4
        cam = self.classifier(x4)
        score = F.adaptive_avg_pool2d(cam, 1)

        # initialize background map
        norm_cam = F.relu(cam)
        norm_cam = norm_cam/(F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1-torch.max(norm_cam,dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
        norm_cam = F.interpolate(norm_cam, side3.shape[2:], mode='bilinear', align_corners=True)*valid_mask

        seeds, probs = self.get_seed(norm_cam.clone(), valid_mask.clone(), sem_feature.clone())
        prototypes = self.get_prototype(seeds, hie_fea)
        IS_cam = self.reactivate(prototypes, hie_fea)
        

        return {"score": score, "cam": norm_cam, "seeds": seeds, "prototypes": prototypes, "IS_cam": IS_cam, "probs":probs}

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class CAM(Net):

    def __init__(self, num_cls):
        super(CAM, self).__init__(num_cls=num_cls)
        self.num_cls = num_cls

    def forward(self, x, label):

        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        side1 = self.side1(x1.detach())
        side2 = self.side2(x2.detach())        
        side3 = self.side3(x3.detach())
        side4 = self.side4(x4.detach())

        hie_fea = torch.cat([F.interpolate(side1/(torch.norm(side1,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side2/(torch.norm(side2,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side3/(torch.norm(side3,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'),
                              F.interpolate(side4/(torch.norm(side4,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear')], dim=1)

        cam = self.classifier(x4)
        cam = (cam[0] + cam[1].flip(-1)).unsqueeze(0)
        hie_fea = (hie_fea[0] + hie_fea[1].flip(-1)).unsqueeze(0)
        
        norm_cam = F.relu(cam)
        norm_cam = norm_cam/(F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1-torch.max(norm_cam,dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
        norm_cam = F.interpolate(norm_cam, side3.shape[2:], mode='bilinear', align_corners=True)

        seeds, _ = self.get_seed(norm_cam.clone(), label.unsqueeze(0).clone(), hie_fea.clone())
        prototypes = self.get_prototype(seeds, hie_fea)
        IS_cam = self.reactivate(prototypes, hie_fea)


        return norm_cam[0], IS_cam[0]
