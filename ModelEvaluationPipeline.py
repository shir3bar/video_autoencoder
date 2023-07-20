from Net import Autoencoder, GANomaly, BN_Autoencoder,BN_GANomaly, AutoencoderB
from GANomaly_model import GAN_Autoencoder as GANomaly_autoencoder
import argparse
import torch.nn as nn
import pandas as pd
from AuxiliaryFunctions import save_recon, error_by_frame,write_gif_fish,write_movie
from VideoDataset import VideoDataset
from VideoTransforms import *
from torch.utils.data import DataLoader
from torchvision import transforms
from timeit import default_timer as timer
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from datetime import datetime
import matplotlib.pyplot as plt
import json
import os
import sys
sys.path.append('/media/shirbar/DATA/codes/SlowFast')
from slowfast.models.ptv_model_builder import PTVResNetAutoencoder
from slowfast.utils.parser import load_config
from slowfast.config.defaults import assert_and_infer_cfg

class Args:
    def __init__(self, cfg_file):
        self.cfg_file = cfg_file
        self.shard_id = 0
        self.num_shards = 1
        self.init_method = 'tcp://localhost:9999'
        self.opts = None




class BaseAE():
    def __init__(self,model,dataloaders,optimizer,criterion,save_dir,scheduler=False,verbose=False,save_every=10):
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_every
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.model = Autoencoder(color_channels=color_channels)
        self.model = model
        self.model.to(self.device)
        self.recon_loss = criterion
        self.save_dir = save_dir
        self.verbose=verbose
        self.train_losses = {}
        self.val_losses = {}
        self.best_loss = np.inf
        self.best_epoch = 0

    def calc_loss(self,clips,validate=False,test=False):
        reconstruction = self.model(clips)
        loss = self.recon_loss(reconstruction, clips)
        if not validate:
            loss.backward()
        running_loss = loss.item() * clips.size(0)
        if not test:
            return running_loss
        else:
            return running_loss, reconstruction

    def train_one_epoch(self):
        running_loss = 0.0
        for i, data in enumerate(self.dataloaders['train'], 0):
            self.optimizer.zero_grad()
            clips = data['clip'].to(self.device)
            running_loss += self.calc_loss(clips)
            self.optimizer.step()
        return running_loss

    def validate(self):
        print('Validating')
        self.model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for j, vdata in enumerate(self.dataloaders['validation']):
                val_clips = vdata['clip'].to(self.device)
                val_loss += self.calc_loss(val_clips,validate=True)
        return val_loss

    def save_checkpoint(self,name,epoch,best=False):
        directory = self.save_dir + '/checkpoints/'
        if best:
            model_name = f'{name}.pt'
        else:
            model_name = f'{name}_epoch{epoch}.pt'
        path = os.path.join(directory, model_name)
        if self.scheduler:
            scheduler_state_dict = self.scheduler.load_state_dict()
        else:
            scheduler_state_dict = None
        if not os.path.exists(directory):
            os.mkdir(directory)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': scheduler_state_dict,
            'train_loss': self.train_losses[epoch],
            'val_loss': self.val_losses[epoch],
        }, path)

    def train(self,num_epochs,start_idx=0,evaluate=True):
        print('Training')
        for epoch in range(start_idx,num_epochs):
            self.model.train()
            epoch_loss = self.train_one_epoch()/len(self.dataloaders['train'].dataset)
            self.train_losses[epoch] = epoch_loss
            if evaluate:
                # judge best by the val loss if validating:
                epoch_loss = self.validate()/len(self.dataloaders['validation'].dataset)
                self.val_losses[epoch] = epoch_loss
            if self.scheduler:
                self.scheduler.step(epoch_loss)
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.best_epoch = epoch
                self.save_checkpoint(name=self.model._get_name()+'_best', epoch=epoch, best=True)
            if self.verbose:
                print(f'Epoch {epoch} Training loss: {self.train_losses[epoch]:.4f}, Validation loss: {self.val_losses[epoch]:.4f}')
            if epoch%self.save_every == 0:
                self.save_checkpoint(name=self.model._get_name(),epoch=epoch)
        self.save_checkpoint(name=self.model._get_name()+'_final',epoch=epoch)

class GanomalyAE(BaseAE):
    def __init__(self,net,dataloaders,optimizer,recon_loss,save_dir,loss_weights={'l_enc':1,'l_recon':50},scheduler=False,verbose=False,save_every=10):
        super().__init__(net,dataloaders,optimizer,recon_loss,save_dir,scheduler,verbose,save_every)
        self.enc_loss = nn.MSELoss()
        self.loss = lambda img_in,z_in,img_out,z_out: \
            loss_weights['l_enc']*self.enc_loss(z_out,z_in) + \
            loss_weights['l_recon']*self.recon_loss(img_out, img_in)
        self.loss_weights = loss_weights


    def calc_loss(self, clips,validate=False,test=False):
        img_out, z_in,  z_out = self.model(clips)
        loss = self.loss(clips,z_in,img_out,z_out)
        if not validate:
            loss.backward()
        running_loss = loss.item() * clips.size(0)
        if not test:
            return running_loss
        else:
            return running_loss,img_out

class ResNetAE(BaseAE):
    def __init__(self,net,dataloaders,optimizer,recon_loss,save_dir,loss_weights={'l_recon':1,'l_lap':500},scheduler=False,verbose=False,save_every=10):
        super().__init__(net,dataloaders,optimizer,recon_loss,save_dir,scheduler,verbose,save_every)
        self.lap_loss = nn.MSELoss()
        seven_pt_stencil = torch.tensor([[[0, 0, 0],
                                          [0, 1, 0],
                                          [0, 0, 0]],
                                         [[0, 1, 0],
                                          [1, -6, 1],
                                          [0, 1, 0]],
                                         [[0, 0, 0],
                                          [0, 1, 0],
                                          [0, 0, 0]]])
        self.laplacian = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.laplacian.weight = nn.Parameter(seven_pt_stencil.unsqueeze(0).unsqueeze(0).float())
        self.laplacian.to('cuda')
        self.laplacian.eval()
        self.loss = lambda img_in,img_out,lap_in,lap_out: \
            loss_weights['l_lap']*self.lap_loss(lap_out,lap_in) + \
            loss_weights['l_recon']*self.recon_loss(img_out, img_in)
        self.loss_weights = loss_weights


    def calc_loss(self, clips,validate=False,test=False):
        img_out = self.model([clips])
        with torch.no_grad():
            lap_in = self.laplacian(img_out)
            lap_out = self.laplacian(clips)
        loss = self.loss(clips, img_out,lap_in,lap_out)
        if not validate:
            loss.backward()
        running_loss = loss.item() * clips.size(0)
        if not test:
            return running_loss
        else:
            return running_loss,img_out

class ModelEvaluationPipeline:
    def __init__(self, model, ds_dir,feed_dir,save_dir,hyperparams,checkpoint=[],load=False,
                 load_weights=True,weight_dir=''):#num_frames,batch_size,learning_rate,train_transforms,test_transforms,match_hist=False):
        """
        
        :param model: model to train
        :param ds_dir: directory of training, validation and test videos
        :param feed_dir: directory of test feed
        :param save_dir: directory to save results
        :param hyperparams: dictionary containing hyperparameters for training and dataset configuration - batch size,
                            number of epochs, learning rate, train transforms, test transforms, model type, number of frames
                            match histogram (boolean), loss function, weight decay, schedule (boolean)
        """
        self.model = model
        self.dataset_dir = ds_dir
        self.feed_dir = feed_dir
        self.hyperparams = hyperparams
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prep_loaders()
        print(f'Train: {len(self.train_ds)}, Validation:{len(self.val_ds)}, '
              f'Test: {len(self.test_ds)}, Feed: {len(self.feed_ds)}')
        if self.hyperparams['loss_func'] == 'L1':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        if hyperparams['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparams['learning_rate'],
                                         weight_decay=self.hyperparams['weight_decay'])
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hyperparams['learning_rate'],
                                             weight_decay=self.hyperparams['weight_decay'], momentum=0.9)
        if hyperparams['schedule']:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=15,
                                                                   min_lr=1e-7, verbose=self.hyperparams['verbose'])
        else:
            self.scheduler = False
        self.start_idx=0
        if isinstance(checkpoint, str):
            print('Loading checkpoint...')
            self.load_checkpoint(checkpoint)
        self.save_dir = save_dir
        if not load:
            self.make_subdirs()
        else:
            self.results_dir = os.path.join(self.save_dir, 'reconstructions')
        if load_weights:
            self.load_weight_init(weight_dir)
        if 'ganomaly' in self.hyperparams['model_type']:
            print('Loading a GANomaly type pipeline')
            self.pipeline = GanomalyAE(self.model,{'train':self.train_loader,'validation':self.val_loader},
                                       self.optimizer,criterion,self.save_dir,
                                       loss_weights=self.hyperparams['loss_weights'],
                                       scheduler=self.scheduler,
                                       verbose=self.hyperparams['verbose'], save_every=10)

        elif 'resnet' in self.hyperparams['model_type']:
            print('Loading a ResNet type pipeline')
            self.pipeline = ResNetAE(self.model,{'train':self.train_loader,'validation':self.val_loader},
                                       self.optimizer,criterion,self.save_dir,
                                       loss_weights=self.hyperparams['loss_weights'],
                                       scheduler=self.scheduler,
                                       verbose=self.hyperparams['verbose'], save_every=10)
        else:
            self.pipeline = BaseAE(self.model,{'train':self.train_loader,'validation':self.val_loader},
                                       self.optimizer, criterion, self.save_dir,
                                       scheduler=self.scheduler,
                                       verbose=self.hyperparams['verbose'], save_every=10)

    def prep_loaders(self):
        train_dir = os.path.join(self.dataset_dir,'train')
        val_dir = os.path.join(self.dataset_dir, 'val')
        test_dir = os.path.join(self.dataset_dir,'test')
        self.train_ds = VideoDataset(train_dir, num_frames=self.hyperparams['num_frames'],
                     transform=transforms.Compose(self.hyperparams['train_transforms']),
                     match_hists=self.hyperparams['match_hists'], color_channels=self.hyperparams['color_channels'])
        self.val_ds = VideoDataset(val_dir, num_frames=self.hyperparams['num_frames'],
                     transform=transforms.Compose(self.hyperparams['train_transforms']),
                     match_hists=self.hyperparams['match_hists'], color_channels=self.hyperparams['color_channels'])
        self.train_loader = DataLoader(self.train_ds, batch_size=self.hyperparams['batch_size'],
                                       shuffle=True, num_workers=4)
        self.val_loader = DataLoader(self.val_ds, batch_size=self.hyperparams['batch_size'],
                                     shuffle=False, num_workers=4)
        self.test_ds = VideoDataset(test_dir, num_frames=self.hyperparams['num_frames'],
                                    transform=transforms.Compose(self.hyperparams['test_transforms']),
                                    match_hists=self.hyperparams['match_hists'],
                                    color_channels=self.hyperparams['color_channels'])
        self.feed_ds = VideoDataset(self.feed_dir, num_frames=self.hyperparams['num_frames'],
                                    transform=transforms.Compose(self.hyperparams['test_transforms']),
                                    match_hists=self.hyperparams['match_hists'],
                                    color_channels=self.hyperparams['color_channels'])
        self.test_loader = DataLoader(self.test_ds, batch_size=1,
                                      shuffle=False, num_workers=4)
        self.feed_loader = DataLoader(self.feed_ds, batch_size=1,
                                      shuffle=False, num_workers=4)

    def make_subdirs(self):
        if os.path.exists(self.save_dir):
            self.save_dir += datetime.now().strftime("%H%M%S")
        os.mkdir(self.save_dir)

        self.results_dir = os.path.join(self.save_dir, 'reconstructions')
        os.mkdir(os.path.join(self.save_dir,'checkpoints'))
        os.mkdir(self.results_dir)
        os.mkdir(os.path.join(self.results_dir, 'feed'))
        os.mkdir(os.path.join(self.results_dir, 'test'))
        os.mkdir(os.path.join(self.results_dir, 'feed', 'sum_error_plots'))
        os.mkdir(os.path.join(self.results_dir, 'test', 'sum_error_plots'))
        torch.save({'model_state_dict': self.model.state_dict()},
                   os.path.join(self.save_dir, 'checkpoints','weight_initialization.pt'))

    def load_weight_init(self,checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        print(f'Loading weights for file {checkpoint_path}')
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def load_checkpoint(self,checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.optimizer.lr = self.hyperparams['learning_rate']
        try:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            self.scheduler=False
            print('scheduler not found')
        self.start_idx = checkpoint['epoch']


    def plot_loss(self,train_time):
        with plt.style.context('seaborn-poster'):
            plt.figure(figsize=(15,10))
            plt.plot(list(self.pipeline.train_losses.keys()),(list(self.pipeline.train_losses.values())))
            plt.plot(list(self.pipeline.val_losses.keys()),(list(self.pipeline.val_losses.values())))
            plt.legend(['train','validation'])
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title(f'{self.hyperparams["model_name"]}'
                      f'{self.hyperparams["loss_func"]} Loss \n'
                      f'{self.hyperparams["optimizer"]} Optimizer With Weight Decay '
                      f'{self.hyperparams["weight_decay"]}- \n '
                      f'lr:{self.hyperparams["learning_rate"]}, {self.hyperparams["num_frames"]} frames, '
                      f'training time: {train_time/3600:.2f} hrs')
            plt.savefig(os.path.join(self.save_dir,f'{self.hyperparams["model_name"]}_ds_lossvsepoch.png'))
            plt.close()

    @staticmethod
    def optic_flow_auxiliary(self,clip,prediction):
        tmp = np.ones((1, 3, clip.shape[2], clip.shape[3], clip.shape[4])) * 0.5
        tmp[:, :-1, :, :, :] = clip
        clip = tmp.copy()
        tmp = np.ones((1, 3, clip.shape[2], clip.shape[3], clip.shape[4])) * 0.5
        tmp[:, :-1, :, :, :] = prediction
        prediction = tmp.copy()
        return clip, prediction

    def eval_category(self,category,dataloader,movie,write_results=True,model_type=None):
        column_names = ['samp_id', 'category', 'avg_error', 'var_error', 'max_error','samp_loss']
        df = pd.DataFrame({column: [] for column in column_names})
        self.pipeline.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                clip = batch['clip'].to(self.device)
                loss, prediction = self.pipeline.calc_loss(clip,validate=True,test=True)
                running_loss += loss
                vidpath = dataloader.dataset.file_paths[i]
                vidname = os.path.basename(vidpath)
                error = torch.sqrt((prediction - clip) ** 2)
                if clip.shape[1] == 2:
                    clip, prediction = self.optic_flow_auxiliary(clip.cpu().numpy(),prediction.cpu().numpy())
                d = {'samp_id': vidname, 'category': category, 'avg_error': error.mean().item(),
                     'var_error': error.var().item(), 'max_error': error.max().item(), 'samp_loss':loss}
                df = df.append(d, ignore_index=True)
                if write_results:
                    errorname = f"{''.join(vidname.split('.')[:-1])}.jpg"
                    gifname = f"{''.join(vidname.split('.')[:-1])}.gif"
                    error_by_frame(error, os.path.join(self.results_dir, category, 'sum_error_plots', errorname))
                    if movie:
                        write_movie(prediction.cpu(),os.path.join(self.results_dir, category),vidname)
                    else:
                        #write_gif_fish(clip, os.path.join(self.results_dir, category, 'original'), gifname)
                        write_gif_fish(prediction.cpu(), os.path.join(self.results_dir, category), gifname)
            df.to_csv(os.path.join(self.results_dir, f'errors{category}.csv'))
            running_loss = running_loss/len(dataloader.dataset)
        return df,running_loss

    def roc_plots(self):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
       # plt.plot(self.best_model_metrics['fpr'], self.best_model_metrics['tpr'], label='lowest loss model')
        plt.plot(self.last_model_metrics['fpr'], self.last_model_metrics['tpr'], label='last epoch model')
        plt.ylabel('tpr')
        plt.xlabel('fpr')
        plt.title('ROC Curve')
       # plt.text(0.55, 0.2, f'Lowest Loss Model AUC score: {self.best_model_metrics["auc"]:.3f} \n '
       #                    f'Last Model AUC score: {self.last_model_metrics["auc"]:.3f} ')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'ROC.jpg'), dpi=200)
        plt.subplot(1, 2, 2)
       # plt.plot(self.best_model_metrics['recall'], self.best_model_metrics['precision'],label='lowest loss model')
        plt.plot(self.last_model_metrics['recall'], self.last_model_metrics['precision'], label='last epoch model')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, f'{self.hyperparams["model_name"]}ROC-PR.jpg'), dpi=200)
        plt.close()

    def calc_metrics(self,errors, labels):
        # a label of 1 is anomaly, a label of 0 is normal,
        # errors are the average reconstruction error for each sample, they'll be our normality score for now
        # max_error = max(errors)  # get max error
        # self.tpr = np.zeros(len(errors))
        # self.fpr = np.zeros(len(errors))
        # self.precision = np.zeros(len(errors))
        # self.threshes = np.zeros(len(errors))
        # for i, thres in enumerate(np.linspace(0, max_error, len(errors))):
        #     # iterate over the space of possible thresholds
        #     new_labels = errors > thres  # get the new labels
        #     tp = sum((new_labels + labels) == 2)
        #     fn = new_labels[(new_labels != labels) & (new_labels == 0)].size
        #     fp = new_labels[(new_labels != labels) & (new_labels == 1)].size
        #     tn = sum((new_labels + labels) == 0)
        #     self.precision[i] = tp / (tp + fp)
        #     # Recall = tpr
        #     self.tpr[i] = tp / (tp + fn)
        #     self.fpr[i] = fp / (fp + tn)
        #     self.threshes[i] = thres
        self.auc_score = roc_auc_score(labels, errors)
        self.fpr, self.tpr, thres = roc_curve(labels, errors)
        self.precision, self.recall, thres = precision_recall_curve(labels, errors)

    def evaluate_performance(self,movie=False, write_results=False):
        test_df,test_loss = self.eval_category('test',self.test_loader,movie, write_results=write_results)
        feed_df,feed_loss = self.eval_category('feed',self.feed_loader,movie, write_results=write_results)
        print(f'Test loss was {test_loss}, feed loss was {feed_loss}')
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
        self.calc_metrics(errors, labels)
        print(f'AUC Score: {self.auc_score}')

    def log_session(self,train_time):
        model_params = self.hyperparams.copy()
        model_params['dataset_name'] = os.path.dirname(self.dataset_dir)
        model_params['training_time'] = train_time/3600
        model_params['datetime'] = datetime.now().strftime('%d-%m-%y %H:%M:%S')
        model_params['train_size'] = len(self.train_ds)
        model_params['val_size'] = len(self.val_ds)
        model_params['test_size'] = len(self.test_ds)
        model_params['loss_weights'] = self.hyperparams['loss_weights']
        #model_params['best_epoch'] = self.pipeline.best_epoch
       # model_params['best_model_auc'] = self.best_model_metrics['auc']
        model_params['last_model_auc'] = self.last_model_metrics['auc']
        df = pd.DataFrame([model_params])
        df = df[['model_name','datetime','loss_weights','dataset_name','train_size','val_size', 'test_size',
                 'last_model_auc', #'best_model_auc','best_epoch',
                 'num_epochs', 'training_time','loss_func','learning_rate',
                 'weight_decay', 'batch_size','schedule', 'num_frames', 'match_hists','color_channels']]
        df.to_csv(os.path.join(self.save_dir,f'{self.hyperparams["model_name"]}_train_log.csv'))
        file = open(os.path.join(self.save_dir,'architecture.txt'),'w+')
        file.write(self.model.__repr__())
        file.close()

    def __call__(self,evaluate,checkpoint=[],verbose=True):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        start = timer()
        torch.manual_seed(42)
        self.pipeline.train(self.hyperparams['num_epochs'], evaluate=evaluate, start_idx=self.start_idx)
        end = timer()
        train_time = end - start
        print(f'elapsed training time {train_time} sec, '
              f'best model loss {self.pipeline.best_loss} at epoch {self.pipeline.best_epoch}')
        self.plot_loss(train_time)
        if evaluate:
            self.evaluate_performance(write_results=False)
            self.last_model_metrics = {'auc':self.auc_score, 'precision':self.precision,
                                       'fpr': self.fpr,'tpr':self.tpr, 'recall':self.recall}
            # best_checkpoint_path = os.path.join(self.save_dir,
            #                                     'checkpoints',
            #                                     f'{self.model._get_name()}_best.pt')
            # if self.pipeline.best_epoch != (self.hyperparams['num_epochs']-1):
            #     torch.cuda.empty_cache()
            #     with torch.no_grad():
            #         self.load_checkpoint(best_checkpoint_path)
            #     print('Evaluating model.... ' + best_checkpoint_path)
            #     self.evaluate_performance(write_results=False)
            #     self.best_model_metrics = {'auc':self.auc_score, 'precision':self.precision,
            #                                'fpr': self.fpr,'tpr':self.tpr,'recall':self.recall}
            # else:
            #     self.best_model_metrics = self.last_model_metrics
            #
            # if self.last_model_metrics['auc']>self.auc_score:
            #     best_checkpoint_path =  os.path.join(self.save_dir,
            #                         'checkpoints',
            #                         f'{self.model._get_name()}_final_epoch{self.hyperparams["num_epochs"]-1}.pt')
            #     torch.cuda.empty_cache()
            #     with torch.no_grad():
            #         self.load_checkpoint(best_checkpoint_path)
            #     new_name =  os.path.join(self.save_dir,
            #                         'checkpoints', f'{self.model._get_name()}_final_best.pt')
            #     os.rename(best_checkpoint_path,new_name)
            # print('Writing results with model.... ' + best_checkpoint_path)
            self.evaluate_performance(write_results=True)
            self.roc_plots()
        self.log_session(train_time)



if __name__ == '__main__':
    parameters = ['num_epochs', 'learning_rate', 'weight_decay', 'batch_size', 'loss_func','schedule','loss_weights',
                        'model_type','model_name', 'num_frames', 'match_hists','color_channels','verbose','optimizer',
                  'cfg']

    # [Rescale(256), RandomHorizontalFlip(0.5), RandomVerticalFlip(0.3), ToTensor()]),
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir',help='enter dataset path')
    parser.add_argument('feed_dir', help='enter feed dataset path')
    parser.add_argument('save_dir', help='enter save dataset path')
    parser.add_argument('-model_type', type=str, default='autoencoder', choices=['autoencoder','ganomaly',
                                                                                 'bn_autoencoder','bn_ganomaly',
                                                                                 'resnet'])
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
    parser.add_argument('-optim', '--optimizer', default='Adam', choices=['Adam','SGD'], help='enter checkpoint path for loading')
    parser.add_argument("-v", '--verbose', action='store_true')
    parser.add_argument("-loss_weights", type=json.loads, default={'l_enc':1,'l_recon':50}, help='loss weights for ganomaly default {l_enc:1,l_rec:50}')
    parser.add_argument('-load_weights', action='store_true',help='should weight initialization be loaded')
    parser.add_argument('-weight_dir', type=str, default='', help='path for weights to initialize')
    parser.add_argument('-cfg', type=str, default='', help='path to cfg for ResNet Autoencdoer')
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
    #print(args)
    if hyperparameters['model_type'] == 'autoencoder':
        model = Autoencoder(color_channels=args.color_channels)
        #model = GANomaly_autoencoder(color_channels=args.color_channels)
        #model = GANomaly_real(color_channels=args.color_channels, num_frames=hyperparameters['num_frames'],
        #                      batchnorm=True)
        hyperparameters['loss_weights'] = np.NaN
    elif hyperparameters['model_type'] == 'bn_autoencoder':
        model = BN_Autoencoder(color_channels=args.color_channels)
        hyperparameters['loss_weights'] = np.NaN
        hyperparameters['model_name'] = f'bn_autoencoder_{datetime.now().strftime("%d%m%y")}'
    elif hyperparameters['model_type'] == 'bn_ganomaly':
        model = BN_GANomaly(color_channels=args.color_channels)
        hyperparameters['model_name'] = f'bn_ganomaly{datetime.now().strftime("%d%m%y")}'
    #elif hyperparameters['model_type'] == 'ganomaly':
        #model = GANomaly_real(color_channels=args.color_channels,num_frames=hyperparameters['num_frames'],batchnorm=False)
    #    if hyperparameters['model_name'].startswith('ae'):
    #        hyperparameters['model_name'] = f'ganomaly_{datetime.now().strftime("%d%m%y")}'
    elif hyperparameters['model_type'] == 'resnet':
        arg = Args(hyperparameters['cfg'])
        cfg = load_config(arg)
        cfg = assert_and_infer_cfg(cfg)
        model = PTVResNetAutoencoder(cfg)
        hyperparameters['model_name'] = f'resnet_ae_{datetime.now().strftime("%d%m%y")}'
    else:
        raise Exception('Model type not supported')
    print(hyperparameters)
    pipeline = ModelEvaluationPipeline(model, args.dataset_dir, args.feed_dir, args.save_dir,hyperparameters,
                                       load_weights=args.load_weights, weight_dir=args.weight_dir)
    pipeline(evaluate=(not args.dont_evaluate), checkpoint=args.checkpoint, verbose=args.verbose)





