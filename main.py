from utils import *
# from model import *
import torch
from models.pointnet2_cls_Github import pointnet2_cls_msg
# from scipy.stats import wasserstein_distance
# from chamfer_distance import ChamferDistance


def train(model, train_loader):
    model.train()
    total_loss = 0
    step = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.L1Loss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_points = None
    for train_x, train_y in train_loader:
        print(train_x.size)
        step += 1
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        optimizer.zero_grad()
        y, train_points = model(train_x, train_points)
        #dist1, dist2 = criterion(decoded.reshape(-1,2048,3), data.y.reshape(-1,2048,3))
        loss = criterion(y, train_y)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return total_loss / len(train_loader)



if __name__ == '__main__':
    train_dataset = load_data("./data/Completion3D/raw/", "train", "Airplane")
    val_dataset = load_data("./data/Completion3D/raw/", "val", "Airplane")

    for batch in val_dataset:
        x = batch[0][0].shape
        y = batch[1][0].shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = pointnet2_cls_msg(3, 3).to(device)
    print(model)
    print('Training started:')
    epoch_num = 400
    for epoch in range(epoch_num):
        loss = train(model, train_dataset)
        print('Epoch {:03d}, Loss: {:.4f}'.format(epoch, loss))

