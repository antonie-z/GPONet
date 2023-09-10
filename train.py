import torch
from torch import nn
from dataset import TrainDataset
from model.GPONet import GPONet
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import os
import time
from torch.cuda.amp import autocast,GradScaler
from tqdm import tqdm

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True

# opt
epoch_num = 60
batch_size_train = 8
data_root = "data/DUTS"
model_name = 'GPONet'
model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)
from_checkpoint = False

# define dataset
sal_dataset = TrainDataset(root=data_root, train=True, aug=True)
sal_dataloader = DataLoader(sal_dataset, batch_size=batch_size_train, num_workers=0, drop_last=True)

train_num = len(sal_dataset)

# define net
net = GPONet()
net = net.to(device)

# load checkpoint
if from_checkpoint:
    saved_models = 'save_models/GPONet_t.pth'
    ck = torch.load(saved_models)
    ck_new = {k.replace('module.', ''): v for k, v in ck.items()}
    net.load_state_dict(ck_new)


# define optimizer
optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-3, weight_decay=0)

scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# training
print("---start training...")
ite_num = 0
running_loss = 0.0
running_mae = 0.0
save_frq = 5
min_loss = 1.0
prt_frq = 50

scaler = GradScaler()
for epoch in range(epoch_num):
    net.train()
    start_time = time.time()

    for i, (img, mask, dt) in enumerate(tqdm(sal_dataloader)):
        ite_num = ite_num + 1

        img, mask, dt = img.type(torch.FloatTensor), mask.type(torch.FloatTensor), dt.type(torch.FloatTensor)

        img, mask, dt = img.to(device), mask.to(device), dt.to(device)

        optimizer.zero_grad()
        with autocast():
            loss,mae = net.calcu_loss(img, dt, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()

        running_loss += loss.data.item()
        running_mae += mae.data.item()

        # print
        if ite_num % prt_frq == 0:
            print(f'[epoch:{epoch+1}/{epoch_num}, batch:{(i + 1) * batch_size_train}/{train_num}, ite:{ite_num}] train loss:{(running_loss / ite_num):.3f}, train_mae:{(running_mae / ite_num):.3f}')

    # save model
    cur_loss = running_loss / ite_num
    if (epoch+1) % save_frq == 0 and cur_loss < min_loss:
        min_loss = cur_loss
        torch.save(net.state_dict(), model_dir + model_name+"_bce_epoch_%d_train_%3f_mae_%3f.pth" % (epoch+1, running_loss / ite_num, running_mae / ite_num))
        net.train()  # resume train

    end_time = time.time()
    use_time_m = (end_time-start_time) // 60
    use_time_s = (end_time-start_time) % 60
    print(f'epoch:{epoch + 1} complete in {use_time_m}m {use_time_s}s')













