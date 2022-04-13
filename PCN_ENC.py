import torch
import torch.nn as nn

class PCN_ENC(nn.Module):
    def __init__(self,input_points):
        super(PCN_ENC, self).__init__()
        self.input_points = input_points
        self.MLP1 = nn.Sequential(nn.Linear(3,128),nn.ReLU(inplace=True),nn.Linear(128,128),nn.ReLU(inplace=True),nn.Linear(128,128))
        self.MLP2 = nn.Sequential(nn.Linear(128,128),nn.ReLU(inplace=True),nn.Linear(128,128),nn.ReLU(inplace=True),nn.Linear(128,128))

        self.MLP3 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Linear(128, 128), nn.ReLU(inplace=True),
                              nn.Linear(128, 256))

        self.MLP4 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Linear(256, 256), nn.ReLU(inplace=True),
                              nn.Linear(256, 256))
        self.MLP5 = nn.Sequential(nn.Linear(256,256),nn.ReLU(inplace=True),nn.Linear(256,256),nn.ReLU(inplace=True),nn.Linear(256,256))
        self.MLP6 = nn.Sequential(nn.Linear(256,512),nn.ReLU(inplace=True),nn.Linear(512,512),nn.ReLU(inplace=True),nn.Linear(512,512))

    def forward(self,input):
        batch = input.shape[1]
        F = self.MLP1(input)
        g = torch.max(F,dim=2,keepdim=True)[0]
        F_head = torch.cat([g.expand(-1,-1,batch),F],dim=1)
        point_feature1 = self.MLP2(F_head)
        F1 = self.MLP3(point_feature1)
        g1 = torch.max(F1, dim=2,keepdim=True)[0]
        F_head1 = torch.cat([g1.expand(-1, -1, batch), F1], dim=1)
        point_feature2 = self.MLP4(F_head1)
        F2 = self.MLP5(point_feature2)
        g2 = torch.max(F2, dim=2,keepdim=True)[0]
        F_head2 = torch.cat([g2.expand(-1, -1, batch), F2], dim=1)
        point_feature3 = self.MLP6(F_head2)
        global_feature = torch.max(point_feature3,dim=2,keepdim=False)[0]
        return point_feature1,point_feature2,global_feature

