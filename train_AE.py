import torch
from torch_geometric.data import DataLoader
from Compeletion3D import Completion3D
from model_modified import SA_net
import kaolin
from emd_func import *


# from test import cal_chamfer_distance

def train():
    model.train()
    total_loss = 0
    step = 0
    for data in train_loader:
        if data.batch[-1] != 15:
            continue
        step += 1
        data = data.to(device)
        optimizer.zero_grad()
        decoded = model(data, data.batch)
        loss = torch.sum(emd(decoded.reshape(-1, 2048, 3), data.y.reshape(-1, 2048, 3), 0.002, 1000)[0]) + 10 * torch.sum(kaolin.metrics.pointcloud.chamfer_distance(decoded.reshape(-1, 2048, 3), data.y.reshape(-1, 2048, 3)))
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(dataset)


if __name__ == '__main__':

    dataset = Completion3D('../data/Completion3D', split='train', categories='Airplane')
    print(dataset[0])
    train_loader = DataLoader(
        dataset, batch_size=16, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = SA_net().to(device)
    # model.load_state_dict(torch.load('C:/Users/Gsh_1/OneDrive/Documents/GitHub/674-final-project/trained/SA_net_Ch_Airplane_674_new20.pt', map_location=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    emd = emdModule()
    print('Training started:')
    for epoch in range(1, 401):
        loss = train()
        print('Epoch {:03d}, Loss: {:.4f}'.format(
            epoch, loss))
        if epoch % 20 == 0:
            torch.save(model.state_dict(), './trained/SA_net_Ch_Airplane_674_modified' + '{}'.format(epoch) + '.pt')
