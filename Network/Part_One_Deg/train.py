import os
import random
import argparse
import numpy as np
from timeit import default_timer as timer
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import PatchSet, load_image_pair, transform_image_test
from utils import AverageMeter
from sewar import rmse, ssim, sam
from Deg_net import BlindNet
from Deg_loss import GeneratorLoss


def uiqi(im1, im2, block_size=64, return_map=False):
    if len(im1.shape) == 3:
        return np.array(
            [uiqi(im1[:, :, i], im2[:, :, i], block_size, return_map=return_map) for i in range(im1.shape[2])])
    delta_x = np.std(im1, ddof=1)
    delta_y = np.std(im2, ddof=1)
    delta_xy = np.sum((im1 - np.mean(im1)) * (im2 - np.mean(im2))) / (im1.shape[0] * im1.shape[1] - 1)
    mu_x = np.mean(im1)
    mu_y = np.mean(im2)
    q1 = delta_xy / (delta_x * delta_y)
    q2 = 2 * mu_x * mu_y / (mu_x ** 2 + mu_y ** 2)
    q3 = 2 * delta_x * delta_y / (delta_x ** 2 + delta_y ** 2)
    q = q1 * q2 * q3
    return q

def test(opt, model, test_dates, IMAGE_SIZE, PATCH_SIZE):
    global final_sam
    cur_result_MS2 = {}
    cur_result_OLCI = {}
    model.eval()

    PATCH_STRIDE = PATCH_SIZE // 2
    end_h = (IMAGE_SIZE[0] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
    end_w = (IMAGE_SIZE[1] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
    h_index_list = [i for i in range(0, end_h, PATCH_STRIDE)]
    w_index_list = [i for i in range(0, end_w, PATCH_STRIDE)]
    if (IMAGE_SIZE[0] - PATCH_STRIDE) % PATCH_STRIDE != 0:
        h_index_list.append(IMAGE_SIZE[0] - PATCH_SIZE)
    if (IMAGE_SIZE[1] - PATCH_STRIDE) % PATCH_STRIDE != 0:
        w_index_list.append(IMAGE_SIZE[1] - PATCH_SIZE)

    for cur_data in test_dates:
        cur_day = int(cur_data.split('_')[1])
        if cur_day == 3:
            images = load_image_pair(opt.root_dir, cur_data)
            output_image_OLCI = np.zeros(images[1].shape)
            image_mask_OLCI = np.ones(images[1].shape)
            inf_mask_OLCI = np.where(images[1] > 1000)
            image_mask_OLCI[inf_mask_OLCI] = 0
            output_image_MS2 = np.zeros(images[1].shape)
            image_mask = np.ones(images[1].shape)
            inf_mask = np.where(images[1] > 1000)
            image_mask[inf_mask] = 0

            for i in range(len(h_index_list)):
                for j in range(len(w_index_list)):
                    h_start = h_index_list[i]
                    w_start = w_index_list[j]
                    OLCI = images[0][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
                    MSI = images[1][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
                    flip_num = 0
                    rotate_num0 = 0
                    rotate_num = 0
                    OLCI, im_mask1 = transform_image_test(OLCI, flip_num, rotate_num0, rotate_num)
                    MSI, im_mask2 = transform_image_test(MSI, flip_num, rotate_num0, rotate_num)
                    OLCI = OLCI.unsqueeze(0).cuda()
                    MSI = MSI.unsqueeze(0).cuda()
                    output = model(OLCI, MSI)
                    output_MS2 = output[1].squeeze()
                    output_OLCI = output[0].squeeze()
                    h_end = h_start + PATCH_SIZE
                    w_end = w_start + PATCH_SIZE
                    cur_h_start = 0
                    cur_h_end = PATCH_SIZE
                    cur_w_start = 0
                    cur_w_end = PATCH_SIZE
                    if i != 0:
                        h_start = h_start + PATCH_SIZE // 4
                        cur_h_start = PATCH_SIZE // 4
                    if i != len(h_index_list) - 1:
                        h_end = h_end - PATCH_SIZE // 4
                        cur_h_end = cur_h_end - PATCH_SIZE // 4
                    if j != 0:
                        w_start = w_start + PATCH_SIZE // 4
                        cur_w_start = PATCH_SIZE // 4
                    if j != len(w_index_list) - 1:
                        w_end = w_end - PATCH_SIZE // 4
                        cur_w_end = cur_w_end - PATCH_SIZE // 4
                    output_image_MS2[:, h_start: h_end, w_start: w_end] = \
                        output_MS2[:, cur_h_start: cur_h_end, cur_w_start: cur_w_end].cpu().detach().numpy()
                    output_image_OLCI[:, h_start: h_end, w_start: w_end] = \
                        output_OLCI[:, cur_h_start: cur_h_end, cur_w_start: cur_w_end].cpu().detach().numpy()

            real_im_MS2 = output_image_MS2
            real_output = output_image_OLCI
            for real_predict in [real_output]:
                cur_result_MS2['rmse'] = []
                cur_result_MS2['ssim'] = []
                cur_result_MS2['cc'] = []
                cur_result_MS2['uiqi'] = []
                cur_result_MS2['ergas'] = 0
                for i in range(9):
                    cur_result_MS2['rmse'].append(rmse(real_im_MS2[i], real_predict[i]))
                    cur_result_MS2['ssim'].append(ssim(real_im_MS2[i], real_predict[i], MAX=1.0)[0])
                    cur_result_MS2['uiqi'].append(uiqi(real_im_MS2[i], real_predict[i]))
                    cur_cc_MS2 = np.sum(
                        (real_im_MS2[i] - np.mean(real_im_MS2[i])) * (real_predict[i] - np.mean(real_predict[i]))) / \
                                 np.sqrt((np.sum(np.square(real_im_MS2[i] - np.mean(real_im_MS2[i])))) * np.sum(
                                     np.square(real_predict[i] - np.mean(real_predict[i]))) + 1e-100)
                    cur_result_MS2['cc'].append(cur_cc_MS2)
                    cur_result_MS2['ergas'] += rmse(real_im_MS2[i], real_predict[i]) ** 2 / (
                            np.mean(real_im_MS2[i]) ** 2 + 1e-100)
                cur_result_MS2['ergas'] = np.sqrt(cur_result_MS2['ergas'] / 6.) * 6
                cur_im_MS2 = real_im_MS2 * 10000.
                cur_predict_MS2 = real_predict * 10000.
                cur_result_MS2['sam'] = sam(cur_im_MS2.transpose(1, 2, 0),
                                            cur_predict_MS2.transpose(1, 2, 0)) * 180 / np.pi
                print('[%s] 1_RMSE: %.4f SSIM: %.4f UIQI: %.4f CC: %.4f ERGAS: %.4f SAM: %.4f' % (
                    cur_data, np.mean(np.array(cur_result_MS2['rmse'])),
                    np.mean(np.array(cur_result_MS2['ssim'])), np.mean(np.array(cur_result_MS2['uiqi'])),
                    np.mean(np.array(cur_result_MS2['cc'])), cur_result_MS2['ergas'], cur_result_MS2['sam']))
                final_sam = cur_result_MS2['sam']
    return final_sam

def train(opt, train_dates, test_dates, IMAGE_SIZE, PATCH_SIZE):
    train_set = PatchSet(opt.train_dir, train_dates, IMAGE_SIZE, PATCH_SIZE)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=8, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BlindNet(ms_bands=9, ratio=15)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('There are %d trainable parameters for generator.' % n_params)
    model.to(device)
    cri_pix = GeneratorLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-5)
    scheculer = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)
    best_sam = 20
    best_epoch = -1
    save_dir = 'J:\ALL\Test'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    Loss_list = []
    for epoch in range(opt.num_epochs):
        print("Epoch:{}  Lr:{:.2E}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        model.train()
        g_loss, batch_time = AverageMeter(), AverageMeter()
        batches = len(train_loader)
        total_train_loss = 0
        for item, (OLCI, MSI, Mask, gt_mask1, gt_mask2) in enumerate(train_loader):
            t_start = timer()
            target = OLCI.to(device)
            input = MSI.to(device)
            maskOLCI = gt_mask1.float().to(device)
            maskMSI = gt_mask2.float().to(device)
            Lr_LOLCI, Lr_LMS1 = model(target, input)
            optimizer.zero_grad()
            loss = cri_pix(Lr_LOLCI*maskMSI, Lr_LMS1*maskMSI, is_ds=False)
            loss.backward()
            optimizer.step()
            g_loss.update(loss.cpu().item())
            t_end = timer()
            batch_time.update(round(t_end - t_start, 4))
            if item % 200 == 199:
                print('[%d/%d][%d/%d] G-Loss: %.4f Batch_Time: %.4f' % (
                    epoch + 1, opt.num_epochs, item + 1, batches, g_loss.avg, batch_time.avg,
                ))
        print('[%d/%d][%d/%d] G-Loss: %.4f Batch_Time: %.4f' % (
            epoch + 1, opt.num_epochs, batches, batches, g_loss.avg, batch_time.avg,
        ))
        final_sam = test(opt, model, test_dates, IMAGE_SIZE, PATCH_SIZE)
        scheculer.step(final_sam)
        if final_sam < best_sam:
            best_sam = final_sam
            best_epoch = epoch
            torch.save(model.state_dict(), save_dir + '/epoch_best.pth')
        torch.save(model.state_dict(), save_dir + '/epoch_%d.pth' % (epoch + 1))
        print('Best Epoch is %d' % (best_epoch + 1), 'sam is %.4f' % best_sam)
        length = len(train_loader)
        train_loss1 = total_train_loss / length
        Loss_list.append(f'{train_loss1:.2f}')
        print("{:.4f}".format(train_loss1))
        print('------------------')

def main():
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    torch.backends.cudnn.deterministic = True
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('--image_size', default=[1800, 1800], type=int, help='the image size (height, width)')
    parser.add_argument('--patch_size', default=128, type=int, help='training images crop size')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
    parser.add_argument('--root_dir', default='J:\SSR\ALL\Root_dir_all', help='Datasets root directory')
    parser.add_argument('--train_dir', default='J:\SSR\ALL\Data_NPY_ALL', help='Datasets train directory')
    parser.add_argument("--step", type=int, default=40, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning Rate. Default=0.1")
    parser.add_argument("--clip", type=float, default=0.1, help="Clipping Gradients. Default=0.4")
    opt = parser.parse_args()
    IMAGE_SIZE = opt.image_size
    PATCH_SIZE = opt.patch_size
    train_dates = []
    test_dates = []
    for dir_name in os.listdir(opt.root_dir):
        cur_day = int(dir_name.split('_')[1])
        if cur_day not in [3]:
            train_dates.append(dir_name)
        else:
            test_dates.append(dir_name)
    train(opt, train_dates, test_dates, IMAGE_SIZE, PATCH_SIZE)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
if __name__ == '__main__':
    main()