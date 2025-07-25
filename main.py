from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as io
import os
import random
import time
import socket

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from model import S3RNet
from data import get_patch_training_set, get_test_set
from torch.autograd import Variable
from psnr import MPSNR
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--custom_test', action='store_true', help='Run test on custom .npy datacube')
parser.add_argument('--upscale_factor', type=int, default=4)
parser.add_argument('--batchSize', type=int, default=8)
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--testBatchSize', type=int, default=1)
parser.add_argument('--ChDim', type=int, default=31)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--nEpochs', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--threads', type=int, default=2)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--save_folder', default='TrainedNet/')
parser.add_argument('--outputpath', type=str, default='result/')
parser.add_argument('--mode', default='test', help='train or test')
opt = parser.parse_args()
print(opt)

# Reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_random_seed(opt.seed)

# Datasets
if opt.mode == 'train':
    train_set = get_patch_training_set(opt.upscale_factor, opt.patch_size)
    training_data_loader = DataLoader(
        dataset=train_set, num_workers=opt.threads,
        batch_size=opt.batchSize, shuffle=True, pin_memory=True
    )

test_set = get_test_set(opt.upscale_factor)
testing_data_loader = DataLoader(
    dataset=test_set, num_workers=opt.threads,
    batch_size=opt.testBatchSize, shuffle=False, pin_memory=True
)

# Model
print('===> Building model')
model = S3RNet().cuda()
print('# network parameters:', sum(p.numel() for p in model.parameters()))
model = torch.nn.DataParallel(model).cuda()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = MultiStepLR(optimizer, milestones=[10, 30, 60, 120], gamma=0.5)

if opt.nEpochs != 0:
    checkpoint = torch.load(opt.save_folder + "_epoch_{}.pth".format(opt.nEpochs))
    opt.lr = checkpoint['lr']
    model.load_state_dict(checkpoint['param'])
    optimizer.load_state_dict(checkpoint['adam'])

criterion = nn.L1Loss()

# TensorBoard
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
CURRENT_DATETIME_HOSTNAME = '/' + current_time + '_' + socket.gethostname()
tb_logger = SummaryWriter(log_dir='./tb_logger/' + 'unfolding2' + CURRENT_DATETIME_HOSTNAME)
current_step = 0

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("--- Created folder:", path)

mkdir(opt.save_folder)
mkdir(opt.outputpath)

def train(epoch, optimizer, scheduler):
    global current_step
    model.train()
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        W, Y, Z, X = [b.cuda().float() for b in batch[:4]]
        optimizer.zero_grad()
        HX, HY, HZ, listX, listY, listZ = model(W)

        alpha = opt.alpha
        loss = criterion(HX, X) + alpha * criterion(HY, Y) + alpha * criterion(HZ, Z)
        for i in range(len(listX) - 1):
            loss += 0.5 * alpha * (
                criterion(X, listX[i]) + criterion(Y, listY[i]) + criterion(Z, listZ[i])
            )

        loss.backward()
        optimizer.step()
        tb_logger.add_scalar('total_loss', loss.item(), current_step)
        current_step += 1
        epoch_loss += loss.item()

        if iteration % 100 == 0:
            print(f"===> Epoch[{epoch}]({iteration}/{len(training_data_loader)}): Loss: {loss.item():.4f}")

    print(f"===> Epoch {epoch} Complete: Avg. Loss: {epoch_loss / len(training_data_loader):.4f}")
    return epoch_loss / len(training_data_loader)

def test():
    avg_psnr = 0
    avg_ssim = 0
    avg_time = 0
    model.eval()

    with torch.no_grad():
        for batch in testing_data_loader:
            W, X = batch[0].cuda(), batch[1].cuda()
            W, X = W.float(), X.float()
            torch.cuda.synchronize()
            start_time = time.time()
            HX, _, _, _, _, _ = model(W)
            torch.cuda.synchronize()
            end_time = time.time()

            X_np = torch.squeeze(X).permute(1, 2, 0).cpu().numpy()
            HX_np = torch.squeeze(HX).permute(1, 2, 0).cpu().numpy()

            # Normalize
            X_np = np.clip(X_np, 0, 1)
            HX_np = np.clip(HX_np, 0, 1)

            psnr = MPSNR(HX_np, X_np)
            ssim_val = ssim(X_np, HX_np, data_range=1.0, multichannel=True)

            im_name = batch[2][0]
            print(im_name)
            print(f"Inference time: {end_time - start_time:.4f} s")

            io.savemat(opt.outputpath + os.path.basename(im_name), {'HX': HX_np})

            avg_psnr += psnr
            avg_ssim += ssim_val
            avg_time += end_time - start_time

    print(f"===> Avg. PSNR: {avg_psnr / len(testing_data_loader):.4f} dB")
    print(f"===> Avg. SSIM: {avg_ssim / len(testing_data_loader):.4f}")
    print(f"===> Avg. Time: {avg_time / len(testing_data_loader):.4f} s")
    return avg_psnr / len(testing_data_loader)

# Test on indian pines datacube
def test_indian_pines():
    import numpy as np
    import torch
    import scipy.io as sio
    from model import S3RNet
    from psnr import MPSNR
    from skimage.metrics import structural_similarity as ssim
    from skimage.transform import resize

    print("===> Running test on Indian Pines")

    # Load trained model
    model = S3RNet(in_channels=3, out_channels=31, ratio=4).cuda()
    model = nn.DataParallel(model)
    checkpoint = torch.load("TrainedNet/_epoch_152.pth")
    model.load_state_dict(checkpoint["param"])
    model.eval()

    # Load datacube
    cube = np.load("indian_pine_array.npy")  
    print("Loaded shape:", cube.shape)

    # Prepare input RGB bands
    rgb_bands = cube[:, :, [30, 20, 10]]  
    rgb_input = np.transpose(rgb_bands, (2, 0, 1))  
    rgb_input = rgb_input / np.max(rgb_input)
    rgb_input = np.expand_dims(rgb_input, 0)  
    input_tensor = torch.from_numpy(rgb_input).float().cuda()

    # Run model
    with torch.no_grad():
        HX, _, _, _, _, _ = model(input_tensor) 

    output_np = HX.squeeze().cpu().numpy().transpose(1, 2, 0) 
    sio.savemat("indian_pines_super_resolution.mat", {"HX": output_np})
    print("Saved to indian_pines_super_resolution.mat")

    gt = cube / np.max(cube)       
    gt = gt[:, :, :31]               

    pred = np.clip(output_np, 0, 1) 

    from skimage.transform import resize
    pred_resized = resize(pred, gt.shape, order=1, preserve_range=True, anti_aliasing=True)

    print("===> Computing metrics...")

    psnr = MPSNR(pred_resized, gt)
    ssim_val = ssim(gt, pred_resized, data_range=1.0, multichannel=True)

    print(f"PSNR: {psnr:.4f} dB")
    print(f"SSIM: {ssim_val:.4f}")

# Main
if __name__ == "__main__":
    if opt.custom_test:
        test_indian_pines()
    elif opt.mode == 'train':
        for epoch in range(opt.nEpochs + 1, 161):
            avg_loss = train(epoch, optimizer, scheduler)
            checkpoint(epoch)
            scheduler.step()
    else:
        test()
