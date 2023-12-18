import torch
import numpy as np
from torch_geometric.loader import DataLoader
from scipy.stats import pearsonr
import random
import os
from layer.model import *
from layer.util import *
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Training graphBAR2 model")
parser.add_argument('--gpu', '-gpu', default='0', help='gpu')
parser.add_argument('--feature_type', '-feature', default=1, help='size of features(0: 13, 1: 22)')
parser.add_argument('--num_layers', '-layers', default=3, help='number of layer in graph convolution')
parser.add_argument('--val_size', '-v', default=1944, help='size of validation set') #refined set recommendation 512
parser.add_argument('--val_type', '-vt', default='stratified', help='type of validation set selection (stratified, random)')
parser.add_argument('--batch_size', '-b', default=32, help='batch size')
parser.add_argument('--output', '-o', help='directory for saving output models')
parser.add_argument('--patience', '-p', default=10, help='patience for early stopping')
parser.add_argument('--fa_layer', '-fa', default=False, help='fully adjacency layer for graph convolution bottleneck')
parser.add_argument('--n_epoch', '-epoch', default=1000, help='number of epochs')
parser.add_argument('--model', '-m', default='GCN_model', help='GCN_model, GAT_model, TransConv_model')
parser.add_argument('--seed', '-seed', default='42', help='seed')
parser.add_argument('--use_dock', '-dock', default=False, help='docking data')
parser.add_argument('--docking_file', '-df', default='docking_dict', help='docking data file')
parser.add_argument('--kinase', '-k', default=False, help='use kinase only (True/False)')

args=parser.parse_args()

gpu_num = args.gpu
feature_type = int(args.feature_type)
num_layers = int(args.num_layers)
val_num = int(args.val_size)
val_type = args.val_type
batch_size = int(args.batch_size)
output = "result/" + args.output
patience = int(args.patience)
fa_layer = str2bool(args.fa_layer)
n_epoch = int(args.n_epoch)
model = args.model
docking_file = 'database/%s.pickle' %args.docking_file
kinase = str2bool(args.kinase)
device = torch.device("cuda:" + gpu_num) if torch.cuda.is_available() else torch.device("cpu")

seed = int(args.seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

use_dock = str2bool(args.use_dock)
bias = False
feature_size = {0 : 18, 1 : 27}

if not os.path.isdir(output):
    os.system("mkdir " + output)


for iter_ in range(5):
    print("iter : %d" %iter_)
    general_file = "database/general_data.pickle" 
    refined_file = "database/refined_data.pickle" 
    general_data = load_data(general_file, feature_type=feature_type)
    refined_data = load_data(refined_file, feature_type=feature_type)
    refined_data = sorted(refined_data, key=lambda value: value.y)

# validation set sampling
    if val_type == "stratified":
        # stratified sampling
        range_val = 15
        val_set = []
        temp_set = []
        refined_sub = [[] for _ in range(range_val)]
        for i in range(len(refined_data)):
            refined_sub[i%range_val].append(refined_data[i])
        for i in range(val_num):
            val_set.append(refined_sub[i%range_val].pop(random.randrange(len(refined_sub[i%range_val]))))
        for i in range(range_val):
            temp_set = temp_set + refined_sub[i]
        refined_data = temp_set

    else:
        # random sampling
        val_set = [refined_data.pop(random.randrange(len(refined_data))) for _ in range(val_num)]

    train_set = general_data + refined_data
    train_list = [tr.pdbid for tr in train_set]

    if use_dock == True:
        docking_data = load_data(docking_file, feature_type=feature_type)
        if not kinase:
            for dock_data in docking_data:
                if dock_data.pdbid in train_list:
                    train_set.append(dock_data)
        else:
            with open("../docking/pdb_kinase.csv", 'r', encoding='utf-8-sig') as f:
                kinase_set = f.read().splitlines()
            kinase_set = set(kinase_set)

            for dock_data in docking_data:
                if (dock_data.pdbid in train_list) or (dock_data.pdbid in kinase_set):
                    train_set.append(dock_data)

    print('train_set, val_set', len(train_set), len(val_set))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    if model=="GCN_model":
        net = GCN_model(input_dim=feature_size[feature_type], num_adjs=2, num_layers=num_layers, fa_layer=fa_layer, bias=bias).to(device)
    elif model=="GAT_model":
        net = GAT_model(input_dim=feature_size[feature_type], num_adjs=2, num_layers=num_layers, fa_layer=fa_layer, bias=bias).to(device)
    elif model=="TransConv_model":
        net = TransConv_model(input_dim=feature_size[feature_type], num_adjs=2, num_layers=num_layers, fa_layer=fa_layer, bias=bias).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-03)

    best_loss = np.inf
    last_saving = 0
    for e in range(1, n_epoch+1):
        train_loss = 0
        val_loss = 0

        net.train()
        for tr_data in train_loader:
            t_data = tr_data.to(device)
            optimizer.zero_grad()
            out = net(t_data)
            loss = criterion(out, torch.unsqueeze(t_data.y, 1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        net.eval()
        for val_data in val_loader:
            v_data = val_data.to(device)
            out = net(v_data)
            val_loss += criterion(out, torch.unsqueeze(v_data.y, 1))
        
        print("epoch %d : train_loss %f val_loss %f" %(e, train_loss, val_loss))

        if val_loss < best_loss:
            torch.save(net, "%s/epoch_%d_%d.pkg" %(output, iter_, e))
            best_loss = val_loss
            last_saving = e
        if (e-last_saving) == patience:
            break

    os.system("cp %s/epoch_%d_%d.pkg %s/epoch_best_%d.pkg" %(output, iter_, last_saving, output, iter_))

#test part
test_file = "database/core_data.pickle" 
test_set = load_data(test_file, feature_type=feature_type)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
rmse_total = []
mae_total = []
corr_total = []
for iter_ in range(5):
    net=torch.load("%s/epoch_best_%d.pkg" %(output, iter_))
    net.eval()
    pred = []
    label = []
    for test_data in test_loader:
        t_data = test_data.to(device)
        out = net(t_data)
        pred.extend(out.cpu().detach().numpy())
        label.extend(t_data.y.cpu().detach().numpy())
    pred = np.squeeze(pred)
    rmse = ((pred-label)**2).mean()**0.5
    mae = (np.abs(pred-label)).mean()
    corr = pearsonr(pred, label)
    rmse_total.append(rmse)
    mae_total.append(mae)
    corr_total.append(corr[0])
rmse_total = np.array(rmse_total)
mae_total = np.array(mae_total)
corr_total = np.array(corr_total)
print(rmse_total)
print("rmse : %f, %f" %(np.mean(rmse_total), np.std(rmse_total)))
print(mae_total)
print("mae : %f, %f" %(np.mean(mae_total), np.std(mae_total)))
print(corr_total)
print("corr : %f, %f" %(np.mean(corr_total), np.std(corr_total)))
result_file = output + "/result.txt"
with open(result_file, 'w') as f:
    f.write('2016 core\n')
    f.write(str(rmse_total))
    f.write('\n')
    f.write("rmse : %f, %f" %(np.mean(rmse_total), np.std(rmse_total)))
    f.write('\n')
    f.write(str(mae_total))
    f.write('\n')
    f.write("mae : %f, %f" %(np.mean(mae_total), np.std(mae_total)))
    f.write('\n')
    f.write(str(corr_total))
    f.write('\n')
    f.write("corr : %f, %f" %(np.mean(corr_total), np.std(corr_total)))
    f.write('\n')

test_file = "database/core2013_data.pickle" 
test_set = load_data(test_file, feature_type=feature_type)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
rmse_total = []
mae_total = []
corr_total = []
for iter_ in range(5):
    net=torch.load("%s/epoch_best_%d.pkg" %(output, iter_))
    net.eval()
    pred = []
    label = []
    for test_data in test_loader:
        t_data = test_data.to(device)
        out = net(t_data)
        pred.extend(out.cpu().detach().numpy())
        label.extend(t_data.y.cpu().detach().numpy())
    pred = np.squeeze(pred)
    rmse = ((pred-label)**2).mean()**0.5
    mae = (np.abs(pred-label)).mean()
    corr = pearsonr(pred, label)
    rmse_total.append(rmse)
    mae_total.append(mae)
    corr_total.append(corr[0])
rmse_total = np.array(rmse_total)
mae_total = np.array(mae_total)
corr_total = np.array(corr_total)
print(rmse_total)
print("rmse : %f, %f" %(np.mean(rmse_total), np.std(rmse_total)))
print(mae_total)
print("mae : %f, %f" %(np.mean(mae_total), np.std(mae_total)))
print(corr_total)
print("corr : %f, %f" %(np.mean(corr_total), np.std(corr_total)))
result_file = output + "/result.txt"
with open(result_file, 'a') as f:
    f.write('2016 core\n')
    f.write(str(rmse_total))
    f.write('\n')
    f.write("rmse : %f, %f" %(np.mean(rmse_total), np.std(rmse_total)))
    f.write('\n')
    f.write(str(mae_total))
    f.write('\n')
    f.write("mae : %f, %f" %(np.mean(mae_total), np.std(mae_total)))
    f.write('\n')
    f.write(str(corr_total))
    f.write('\n')
    f.write("corr : %f, %f" %(np.mean(corr_total), np.std(corr_total)))
    f.write('\n')
