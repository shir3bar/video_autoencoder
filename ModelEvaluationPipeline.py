from Net import Net
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import cv2
from AuxiliaryFunctions import save_checkpoint, save_recon, error_by_frame,write_gif_fish,write_movie
from VideoDataset import VideoDataset
from VideoTransforms import *
from torch.utils.data import DataLoader
from torchvision import transforms
from timeit import default_timer as timer
from sklearn.metrics import auc
from moviepy.editor import ImageSequenceClip
import shutil
import seaborn as sns
from PIL import Image, ImageSequence
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import os
from pathlib import Path, PureWindowsPath
from collections import OrderedDict
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import pickle

class ModelEvaluationPipeline:
    def __init__(self, model, train_folder,test_folder,feed_dir,save_dir,hyperparams):#num_frames,batch_size,learning_rate,train_transforms,test_transforms,match_hist=False):
        """
        
        :param model: model to train
        :param train_folder: directory of training (and validation) videos
        :param test_folder: directory of test videos
        :param feed_dir: directory of test feed
        :param save_dir: directory to save results
        :param hyperparams: dictionary containing hyperparameters for training and dataset configuration - batch size,
                            number of epochs, learning rate, train transforms, test transforms, model type, number of frames
                            match histogram (boolean), loss function, weight decay, schedule (boolean)
        """
        self.train_dir = train_folder
        self.test_dir = test_folder
        self.feed_dir = feed_dir
        ds = VideoDataset(self.train_dir, num_frames=hyperparams['num_frames'],
                          transform=transforms.Compose(hyperparams['train_transforms']),
                          match_hists=hyperparams['match_hists'])
        num_valid = int(0.15 / 0.85 * len(ds)) # set the validation to be 15% of original dataset size
        self.train_ds, self.val_ds = torch.utils.data.random_split(ds, (len(ds) - num_valid, num_valid))
        self.train_loader = DataLoader(self.train_ds, batch_size=hyperparams['batch_size'],
                                       shuffle=True, num_workers=12)
        self.val_loader = DataLoader(self.val_ds, batch_size=hyperparams['batch_size'],
                                       shuffle=False, num_workers=4)
        self.test_ds = VideoDataset(self.test_dir, num_frames=hyperparams['num_frames'],
                                    transform=transforms.Compose(hyperparams['test_transforms']),
                                    match_hists=hyperparams['match_hists'])
        self.feed_ds = VideoDataset(self.feed_dir, num_frames=hyperparams['num_frames'],
                                    transform=transforms.Compose(hyperparams['test_transforms']),
                                    match_hists=hyperparams['match_hists'])
        self.test_loader = DataLoader(self.test_ds, batch_size=1,
                                      shuffle=False, num_workers=4)
        self.feed_loader = DataLoader(self.feed_ds, batch_size=1,
                                      shuffle=False, num_workers=4)
        self.model = model
        self.save_dir = save_dir
        self.make_subdirs()
        self.hyperparams = hyperparams
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.train_losses = {}
        self.val_losses = {}

    def make_subdirs(self):
        if os.path.exists(self.save_dir):
            self.save_dir += self.hyperparams['model_name']+datetime.now().strftime("%d%m%y")
        os.mkdir(self.save_dir)
        self.results_dir = os.path.join(self.save_dir, 'reconstructions')
        os.mkdir(os.path.join(self.save_dir,'checkpoints'))
        os.mkdir(self.results_dir)
        os.mkdir(os.path.join(self.results_dir,'feed'))
        os.mkdir(os.path.join(self.results_dir, 'test'))
        os.mkdir(os.path.join(self.results_dir,'feed','sum_error_plots'))
        os.mkdir(os.path.join(self.results_dir, 'test','sum_error_plots'))

    def load_checkpoint(self,checkpoint,optimizer,scheduler):
        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer.lr = self.hyperparams['learning_rate']
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except KeyError:
            print('scheduler not found')
        start_idx = checkpoint['epoch']
        return optimizer,scheduler,start_idx

    def train(self,checkpoint=[],verbose=True):
        torch.manual_seed(42)
        directory = os.path.join(self.save_dir,'checkpoints')
        if self.hyperparams['loss_func'] == 'L1':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparams['learning_rate'],
                                     weight_decay=self.hyperparams['weight_decay'])
        if self.hyperparams['schedule']:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=15,
                                                                   min_lr=1e-7, verbose=verbose)
        start_idx = 0
        if isinstance(checkpoint, dict):
            optimizer, scheduler, start_idx = self.load_checkpoint(checkpoint,optimizer,scheduler)
        for epoch in range(start_idx, self.hyperparams['num_epochs']):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                optimizer.zero_grad()
                clips = data['clip'].to(self.device)
                reconstruction = self.model(clips)
                loss = criterion(clips, reconstruction)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * clips.size(0)
                if (i % 500 == 0) and verbose:  # print every 500 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, loss.item()))

            self.train_losses[epoch] = running_loss / len(self.train_loader.dataset)
            # evaluate the validation data set:
            with torch.no_grad():
                val_loss = 0.0
                for j, vdata in enumerate(self.val_loader):
                    val_clips = vdata['clip'].to(self.device)
                    val_recon = self.model(val_clips)
                    val_loss += criterion(val_clips, val_recon) * val_clips.size(0)
                self.val_losses[epoch] = val_loss / len(self.val_loader.dataset)
            if verbose:
                print(f'Train loss: {self.train_losses[epoch]}, Validation loss: {self.val_losses[epoch]}')
            # step the scheduler
            scheduler.step(self.val_losses[epoch])
            save_checkpoint(self.model, optimizer, epoch, train_loss=self.train_losses[epoch],
                            scheduler_state_dict=scheduler.state_dict(),
                            val_loss=self.val_losses[epoch], directory=directory,
                            name=self.hyperparams['model_name'])
            save_recon(reconstruction, self.hyperparams['model_name'], epoch, directory)

    def plot_loss(self,train_time):
        with plt.style.context('seaborn-poster'):
            plt.figure(figsize=(15,10))
            plt.plot(list(self.train_losses.keys()),(list(self.train_losses.values())))
            plt.plot(list(self.val_losses.keys()),(list(self.val_losses.values())))
            plt.legend(['train','validation'])
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title(f'{self.hyperparams["loss_func"]} Loss Adam Optimizer With Weight Decay - \n '
                      f'lr:{self.hyperparams["learning_rate"]}, {self.hyperparams["num_frames"]} frames, '
                      f'training time: {train_time/3600} hrs')
            plt.savefig(os.path.join(self.save_dir,'ds_lossvsepoch.png'))
            plt.close()

    def optic_flow_auxiliary(self,clip,prediction):
        tmp = np.ones((1, 3, clip.shape[2], clip.shape[3], clip.shape[4])) * 0.5
        tmp[:, :-1, :, :, :] = clip
        clip = tmp.copy()
        tmp = np.ones((1, 3, clip.shape[2], clip.shape[3], clip.shape[4])) * 0.5
        tmp[:, :-1, :, :, :] = prediction
        prediction = tmp.copy()
        return clip, prediction

    def eval_category(self,category,dataloader,movie):
        column_names = ['samp_id', 'category', 'avg_error', 'var_error', 'max_error']
        df = pd.DataFrame({column: [] for column in column_names})
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                clip = batch['clip']
                prediction = self.model(clip)
                vidpath = dataloader.dataset.file_paths[i]
                vidname = os.path.basename(vidpath)
                error = np.sqrt((prediction - clip) ** 2)
                if clip.shape[1] == 2:
                    clip, prediction = self.optic_flow_auxiliary(clip,prediction)
                d = {'samp_id': vidname, 'category': category, 'avg_error': error.mean().item(),
                     'var_error': error.var().item(), 'max_error': error.max().item()}
                df = df.append(d, ignore_index=True)
                errorname = f"{''.join(vidname.split('.')[:-1])}.jpg"
                gifname = f"{''.join(vidname.split('.')[:-1])}.gif"
                error_by_frame(error, os.path.join(self.results_dir, category, 'sum_error_plots', errorname))
                if movie:
                    write_movie(prediction,os.path.join(self.results_dir, category),vidname)
                else:
                    #write_gif_fish(clip, os.path.join(self.results_dir, category, 'original'), gifname)
                    write_gif_fish(prediction, os.path.join(self.results_dir, category), gifname)
            df.to_csv(os.path.join(self.results_dir, f'errors{category}.csv'))
        return df

    def roc_plots(self):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.fpr, self.tpr)
        plt.ylabel('tpr')
        plt.xlabel('fpr')
        plt.title('ROC Curve')
        plt.text(0.2, 0.8, f'AUC score: {self.auc_score}')
        plt.savefig(os.path.join(self.save_dir, 'ROC.jpg'), dpi=200)
        plt.subplot(1, 2, 2)
        plt.plot(self.tpr, self.precision)
        plt.ylabel('Recall')
        plt.xlabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig(os.path.join(self.save_dir, 'ROC-PR.jpg'), dpi=200)
        plt.close()

    def roc_curve(self,errors, labels):
        # a label of 1 is anomaly, a label of 0 is normal,
        # errors are the average reconstruction error for each sample, they'll be our normality score for now
        max_error = max(errors)  # get max error
        self.tpr = np.zeros(len(errors))
        self.fpr = np.zeros(len(errors))
        self.precision = np.zeros(len(errors))
        self.threshes = np.zeros(len(errors))
        for i, thres in enumerate(np.linspace(0, max_error, len(errors))):
            # iterate over the space of possible thresholds
            new_labels = errors > thres  # get the new labels
            tp = sum((new_labels + labels) == 2)
            fn = new_labels[(new_labels != labels) & (new_labels == 0)].size
            fp = new_labels[(new_labels != labels) & (new_labels == 1)].size
            tn = sum((new_labels + labels) == 0)
            self.precision[i] = tp / (tp + fp)
            # Recall = tpr
            self.tpr[i] = tp / (tp + fn)
            self.fpr[i] = fp / (fp + tn)
            self.threshes[i] = thres
        self.auc_score = auc(self.fpr, self.tpr)
        self.roc_plots()

    def evaluate_performance(self,movie=False):
        self.model.eval()  # put the model in evaluation mode
        test_df = self.eval_category('test',self.test_loader,movie)
        feed_df = self.eval_category('feed',self.feed_loader,movie)
        plt.figure(figsize=(10, 8))
        plt.hist(feed_df.avg_error, bins=10, alpha=0.5, label='feed')
        plt.hist(test_df.avg_error, bins=10, alpha=0.5, label='test')
        plt.legend()
        plt.savefig(os.path.join(self.results_dir,'error_dist.png'))
        feed_error = feed_df['avg_error'].to_numpy()
        swim_error = test_df['avg_error'].to_numpy()
        feed_labels = np.ones(len(feed_error))
        swim_labels = np.zeros(len(swim_error))
        labels = np.concatenate([feed_labels, swim_labels])
        errors = np.concatenate([feed_error, swim_error])
        self.roc_curve(errors, labels)
        print(f'AUC Score: {self.auc_score}')

    def __call__(self,evaluate,checkpoint=[],verbose=True):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, map_location=self.device)
        start = timer()
        self.train(checkpoint, verbose)
        end = timer()
        train_time = end - start
        print(f'elapsed training time {train_time} sec')
        self.plot_loss(train_time)
        if evaluate:
            self.evaluate_performance()
        df = pd.DataFrame(self.hyperparams)
        df.to_csv(os.path.join(self.save_dir,'hyperparameters.csv'))
        file = open(os.path.join(self.save_dir,'validation_indexs.txt'),mode='+w')
        file.write(self.val_ds.dataset.filepaths)
        file.close()



if __name__ == '__main__':
    parameters = ['num_epochs', 'learning_rate', 'weight_decay', 'batch_size', 'loss_func','schedule',
                        'model_name', 'num_frames', 'match_hists','color_channels']

    # [Rescale(256), RandomHorizontalFlip(0.5), RandomVerticalFlip(0.3), ToTensor()]),
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir',help='enter train dataset path')
    parser.add_argument('test_dir', help='enter test dataset path')
    parser.add_argument('feed_dir', help='enter feed dataset path')
    parser.add_argument('save_dir', help='enter save dataset path')
    parser.add_argument('-epochs', '--num_epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('-lr','--learning_rate',type=float, default=1e-3, help='optimizer learning rate')
    parser.add_argument('-weight_decay', type=float, default=1e-5, help='weight decay for optimization')
    parser.add_argument('-bs','--batch_size',type=int,default=4, help='mini-batch size')
    parser.add_argument('-loss','--loss_func',type=str, default='L1',
                        choices=['L1','L2'], help='Choose the loss function')
    parser.add_argument('-model_name', type=str, default=f'ae_{datetime.now().strftime("%d%m%y")}',help='model name')
    parser.add_argument('-frms','--num_frames',type=int,default=100,help='number of frames in video clips in the ds')
    parser.add_argument('-rescale', type=int, default=256, help='Apply rescaling to samples, specify dimensions')
    parser.add_argument('-h_flip', action="store_true",help='Apply horizontal flip to train samples')
    parser.add_argument('-v_flip', action="store_true",help='Apply vertical flip to train samples')
    parser.add_argument('-match_hists', action="store_true",help='Apply histogram matching to samples')
    parser.add_argument('-schedule', action="store_true", help='Apply learning rate schedule')
    parser.add_argument('-color_channels', type=int, default=1, choices=[1,2,3],
                        help='choose #color channels in input clips, 2 is for optic flow data')
    parser.add_argument('-no_eval','--dont_evaluate',action='store_true',help='raise flag to avoid evaluating the model')
    parser.add_argument('-chkpt','--checkpoint', default=[], help='enter checkpoint path for loading')
    parser.add_argument("-v", '--verbose', action='store_true')
    args = parser.parse_args()
    train_transforms = []
    test_transforms = []
    if args.rescale > 0:
        train_transforms.append(Rescale(args.rescale))
        test_transforms.append(Rescale(args.rescale))
    if args.h_flip:
        train_transforms.append(RandomHorizontalFlip(0.5))
    if args.v_flip:
        train_transforms.append(RandomVerticalFlip(0.5))
    train_transforms.append(ToTensor())
    test_transforms.append(ToTensor())
    hyperparameters = {}
    for param in parameters:
        hyperparameters[param] = vars(args)[param]
    hyperparameters['train_transforms'] = train_transforms
    hyperparameters['test_transforms'] = test_transforms
    print(args)
    model = Net(color_channels=args.color_channels)
    pipeline = ModelEvaluationPipeline(model,args.train_dir,args.test_dir,args.feed_dir,args.save_dir,hyperparameters)
    pipeline(evaluate=(not args.dont_evaluate), checkpoint=args.checkpoint, verbose=args.verbose)





