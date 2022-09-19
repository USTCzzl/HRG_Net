import datetime
import os
import sys
import argparse
import logging
import cv2
import torch
import torch.utils.data
import torch.optim as optim
from torchsummary import summary
from sklearn.model_selection import KFold
#from traning import train, validate
from utils.data import get_dataset
from models.common import post_process_output
from utils.dataset_processing import evaluation
#from models.swin import SwinTransformerSys
# from models.Swin_without_skipconcetion import SwinTransformerSys
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='TF-Grasp')

    # Network


    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str,default="jaquard", help='Dataset Name ("cornell" or "jaquard or multi")')
    parser.add_argument('--dataset-path', type=str,default="/home/zzl/Pictures/cornell" ,help='Path to dataset')
    parser.add_argument('--dataset-path1', type=str,default="/home/zzl/Pictures/cornelltest" ,help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for training (0/1)')
    parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=2000, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=500, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=201, help='Validation Batches')
    # Logging etc.
    parser.add_argument('--description', type=str, default='', help='Training description')
    parser.add_argument('--outdir', type=str, default='output/models/', help='Training Output Directory')

    args = parser.parse_args()
    return args

def validate(net, device, val_data, batches_per_epoch):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param batches_per_epoch: Number of batches to run
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }

    ld = len(val_data)

    with torch.no_grad():
        batch_idx = 0
        while batch_idx < batches_per_epoch:
            for x, y, didx, rot, zoom_factor in val_data:
                batch_idx += 1
                if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
                    break

                xc = x.to(device)
                yc = [yy.to(device) for yy in y]
                lossd = net.compute_loss(xc, yc)

                loss = lossd['loss']

                results['loss'] += loss.item()/ld
                for ln, l in lossd['losses'].items():
                    if ln not in results['losses']:
                        results['losses'][ln] = 0
                    results['losses'][ln] += l.item()/ld

                q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                            lossd['pred']['sin'], lossd['pred']['width'])

                s = evaluation.calculate_iou_match(q_out, ang_out,
                                                   val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                                                   no_grasps=1,
                                                   grasp_width=w_out,
                                                   )

                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1

    return results


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False,scheduler=None):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()

    batch_idx = 0
    index=0
    count=1
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    # while batch_idx < batches_per_epoch:
    while index < count:
        index=index+1
        batch_idx=0
        for x, y, _, _, _ in train_data:
            # print("shape:",x.shape)
            batch_idx += 1
            # if batch_idx >= batches_per_epoch:
            #     break

            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc)

            loss = lossd['loss']

            if batch_idx % 100 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # Display the images
            if vis:
                imgs = []
                n_img = min(4, x.shape[0])
                for idx in range(n_img):
                    imgs.extend([x[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
                        x[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in lossd['pred'].values()])
                gridshow('Display', imgs,
                         [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (0.0, 1.0)] * 2 * n_img,
                         [cv2.COLORMAP_BONE] * 10 * n_img, 10)
                cv2.waitKey(2)

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results
def run():
    args = parse_args()
    # Set-up output directories
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))

    save_folder = os.path.join(args.outdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)


    dataset = Dataset(args.dataset_path, start=0.0, end=0.9, ds_rotate=args.ds_rotate,
                                              random_rotate=True, random_zoom=True,
                                              include_depth=args.use_depth, include_rgb=args.use_rgb)

    # dataset1 = Dataset("/home/zzl/Pictures/cornelltest", start=0.0, end=1.0, ds_rotate=args.ds_rotate,
    #                                           random_rotate=True, random_zoom=True,
    #                                           include_depth=args.use_depth, include_rgb=args.use_rgb)                                          
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    logging.info('Done')
    logging.info('Loading Network...')
    input_channels = 1*args.use_depth + 3*args.use_rgb
    
    from models.parm import config
    from models.HEHERnet_official import HRNet
    
    net=HRNet(input_channels=input_channels,cfg=config)

    #net = SwinTransformerSys(in_chans=input_channels,embed_dim=48,num_heads=[1,2,4,8])
    device = torch.device("cuda:0")
    net = net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=1e-4)
    listy = [x *30 for x in range(1,1000,3)]
    schedule=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=listy,gamma=0.9)
    logging.info('Done')
    
    best_iou = 0.0
    for epoch in range(args.epochs):
        accuracy=0.
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            trainloader = torch.utils.data.DataLoader(
                           dataset,
                           batch_size=args.batch_size,num_workers=args.num_workers, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                           dataset,
                           batch_size=1,num_workers=args.num_workers, sampler=test_subsampler)


            logging.info('Beginning Epoch {:02d}'.format(epoch))
            print("lr:",optimizer.state_dict()['param_groups'][0]['lr'])
            train_results = train(epoch, net, device, trainloader, optimizer, args.batches_per_epoch, )
            schedule.step()

        # Run Validation
            logging.info('Validating...')
            test_results = validate(net, device, testloader, args.val_batches)
            logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct']/(test_results['correct']+test_results['failed'])))


            iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
            accuracy+=iou
            if iou > best_iou or epoch == 0 or (epoch % 50) == 0:
                #torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
                torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.4f' % (epoch, iou)), _use_new_zipfile_serialization=False)
                # torch.save(net.state_dict(), os.path.join(save_folder, 'epoch_%02d_iou_%0.2f_statedict.pt' % (epoch, iou)))
                best_iou = iou
        schedule.step()
        print("the accuracy:",accuracy/k_folds)


if __name__ == '__main__':
    run()
