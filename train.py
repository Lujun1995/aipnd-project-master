import argparse
import load_data
import model_fuc

parser = argparse.ArgumentParser(description = 'Train a deep neural network on flower datasets')
parser.add_argument('data_dir', nargs = '?', action = "store", default = "./flowers/")
parser.add_argument('--save_dir', dest = 'save_dir', nargs = '?', action = 'store', default = './checkpoint.pth')
parser.add_argument('--arch', dest = 'arch', nargs = '?', action = "store", default = 'vgg16')
parser.add_argument('--learning_rate', dest = 'lr', nargs='?', action="store", type = int, default=0.001)
parser.add_argument('--hidden_units', dest = 'hidden_units', nargs='?', action="store", type = int, default=500)
parser.add_argument('--epochs', dest = 'epochs', nargs='?', action="store", type = int, default=5)
parser.add_argument('--gpu', dest = 'gpu', nargs='?', action="store", default='GPU')

pa = parser.parse_args()
data_dir = pa.data_dir
save_dir = pa.save_dir
arch = pa.arch
lr = pa.lr
hidden_units = pa.hidden_units
epochs = pa.epochs
gpu = pa.gpu

train_dataloaders, vaild_dataloaders, test_dataloaders, class_to_idx = load_data.loaddata(data_dir)
model, criterion, optimizer = model_fuc.setup_model(structure = arch, dropout = 0.5, lr=lr, power = gpu, hidden_layer = hidden_units)
model_fuc.train_model(model, criterion, optimizer, train_dataloaders, vaild_dataloaders, power = gpu, epochs = epochs)
model_fuc.save_model(class_to_idx, save_dir, model, arch, optimizer)



