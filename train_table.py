from dataloaders.dataset_utils import LimitDataset
import torch
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import pickle
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os 
from tqdm import tnrange, tqdm_notebook, tqdm
from PIL import Image
import copy
import time
from utils import set_up_dir
from sklearn.metrics import confusion_matrix
import numpy as np
    
def get_patch_pred(mask, patch):
    patch_flat = patch.view((patch.shape[0],patch.shape[1], -1))
    mask_flat = mask.view((mask.shape[0],mask.shape[1], -1))
    X = patch_flat*mask_flat
    Y_pred = torch.argmax(X.sum(2), dim=1)
    return Y_pred.tolist()

def get_true_labels(y):
    y_flat = y.view((y.shape[0],y.shape[1], -1))
    Y_pred = torch.argmax(y_flat.sum(2), dim=1)
    adv_mask = y_flat.sum(2).sum(1)==0.
    Y_pred[adv_mask] = -1
    return Y_pred.tolist()

def get_confusion_fig(y1,y2):
    cm = confusion_matrix(y1,y2)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, axes = plt.subplots(1,1,figsize=(8,8))
    pcm = axes.imshow(cm)
    fig.colorbar(pcm, ax=axes)
    return fig
    
class TableTrainer(object):

    """
    Train a digit detection model (see train_table.py for a use case).
    Arguments:
        savedir (str):  Directory to save results in.
        descr_name (str):  Additional descriptor string to specifiy the training run.
        model (torch.nn.Module): Model architecture to be used, here: TableModel (see models/e2cnn_models.py).
        epochs (int): Number of training epochs.
        weight_dir (str, optional): File to initilaize the model from pretrained weights.
        jobdict (str, optional):   
        seed (int, optional):  Seed initilization.
        lr_params (tuple, optional):  Tuple with additional learning rate parameters (lr, step_size, gamma).
        default_init_lr (str, optional):  Specify the initial learning rate. 

    """
    
    def __init__(self, savedir, descr_name, model, epochs, jobdict=None, weight_dir=None, seed=0, lr_params=None, default_init_lr = 0.001):
                 
        if weight_dir:
            self.start_epoch =  int(os.path.split(weight_dir)[1].split('_ep')[1].replace('.pt','')) + 1
        else:
            self.start_epoch = 0
        self.num_epochs =  self.start_epoch + epochs
        self.save_dir_name = savedir
        self.weight_dir = weight_dir
        self.descr_name = descr_name
        set_up_dir(self.save_dir_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        if jobdict:
            pfile = os.path.join(self.save_dir_name, 'jobdict.p')
            pickle.dump(jobdict, open(pfile, 'wb'))

        self.lr_params = lr_params
        self.default_init_lr = default_init_lr
        torch.manual_seed(seed)

    def train(self, dataloader):

        if self.weight_dir:
            #Load weights
            self.model.load_state_dict(torch.load(self.weight_dir))
                 
        self.model.to(self.device)
        params = self.model.parameters()                                         
        criterion = torch.nn.MSELoss()
   
        if self.lr_params is not None:
            lr, step_size, gamma = self.lr_params[0], self.lr_params[1], self.lr_params[2]
            print('lr params', lr, step_size, gamma)
            optimizer = torch.optim.Adam(params, lr = lr)
            lr_scheduler_train = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            optimizer = torch.optim.Adam(params, lr = self.default_init_lr)
            lr_scheduler_train = None
    
        model_trained, res_dict = self.train_model(self.model, dataloader, criterion, optimizer, scheduler=lr_scheduler_train)
    
        weight_file = '{}/final_model.pt'.format(self.save_dir_name)
        
        torch.save(model_trained.state_dict(), weight_file)
        pickle.dump(res_dict, open(os.path.join(self.save_dir_name, 'res.p'), 'wb'))
        return weight_file
            
    def train_model(self, model, dataloader, criterion, optimizer, scheduler=None):

        dataset_sizes = {'train':  len(dataloader['train']), 
                'test': len(dataloader['test'])}

        print(dataset_sizes)
        writer = SummaryWriter(os.path.join(self.save_dir_name, self.descr_name))

        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())

        res_dict = {'train': {'loss':[], 'acc' :[]},
                    'test': {'loss':[],  'acc' :[]},
                     'dataset_sizes': dataset_sizes}

        for epoch in tnrange(self.start_epoch, self.num_epochs, desc='Epochs'):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                Y_pred, Y_true = [],[]

                if phase == 'train':
                    if scheduler:
                        print('LR Scheduler step')
                        scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                # Iterate over data.
                running_loss = 0.0
                it = 0
                tsamples = 0.
                datal =  dataloader[phase]

                for patch, Y, mask, number  in tqdm(datal, desc='Sample Loop'):

                    if len(patch.shape)==3:
                        patch = patch.unsqueeze(0)
                    if len(Y.shape)==3:
                        Y = Y.unsqueeze(0)

                    X_ = patch

                    A = X_.cpu().detach().numpy()
                    B = Y.cpu().detach().numpy()

                    inputs = X_.to(self.device)
                    y_map = Y.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):

                        outputs = model(inputs)

                        C = outputs.cpu().detach().numpy()

                        acc, preds = 1.0, 1.0

                        # Masking of loss depending on context 
                        # In this case context is ignored for the loss computation 
                        # (loss only based on annotation box)
                        mask_ = torch.Tensor(mask).to(self.device)
                        masked_outputs = mask_*outputs

                        loss = criterion(masked_outputs, y_map)
                        
                        y_pred = get_patch_pred(mask_, outputs)
                        y_true = get_true_labels(y_map) #[int(i) for int in number.tolist()]
                       
                        Y_pred.extend(y_pred)
                        Y_true.extend(y_true)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                           
                    # statistics
                    running_loss += loss.item() #* inputs.size(0)
                    
                    tsamples += inputs.size(0)

                    it += 1

                epoch_loss = running_loss / tsamples #dataset_sizes[phase]
                res_dict[phase]['loss'].append(epoch_loss)

                #### Tensorboard ####
                # write current loss
                writer.add_scalar(phase+'/epoch_loss', epoch_loss, epoch)

                # Create plot of predictions to visualize learning
                fig, axes = plt.subplots(11,5,figsize=(14,20))
                with torch.no_grad():
                    datal = dataloader[phase]

                    for i, (image, bboxes, labels, num) in enumerate(datal):
                        assert len(image.shape) == 4
                        #if REMOVE_MEAN:
                            #image = image - torch.mean(image, dim=(2,3), keepdim=True)

                        out = model(image.to(self.device))
                        h = axes[0][i].imshow(image[0][0])
                        axes[0][i].get_xaxis().set_visible(False)
                        axes[0][i].get_yaxis().set_visible(False)
                        
                        plt.colorbar(h,ax=axes[0,i])

                        for j in range(10):
                            h = axes[j+1][i].imshow(out[0][j].cpu().detach(), vmin=out[0].min(), vmax=out[0].max())
                            axes[j+1][i].get_xaxis().set_visible(False)
                            axes[j+1][i].get_yaxis().set_visible(False)

                            plt.colorbar(h,ax=axes[j+1])
                        if i == 4: break

                plt.suptitle("vmin={:.2f} - vmax={:.2f}".format(out[0].min(), out[0].max()), y=0.9)
                writer.add_figure('Pred ' + phase, fig, global_step=epoch)


                #Write labels
                fig, axes = plt.subplots(11,5,figsize=(14,20))
                with torch.no_grad():
                    datal = dataloader[phase]
                    for i, (image, labels, bboxes, num) in enumerate(datal):

                        assert len(image.shape) == 4

                       # out = model(image.to(self.device))
                        axes[0][i].imshow(image[0][0])
                        axes[0][i].get_xaxis().set_visible(False)
                        axes[0][i].get_yaxis().set_visible(False)
                        for j in range(10):
                            h = axes[j+1][i].imshow(labels[0][j].cpu().detach(), vmin=labels[0].min(), vmax=labels[0].max())
                            axes[j+1][i].get_xaxis().set_visible(False)
                            axes[j+1][i].get_yaxis().set_visible(False)
                            plt.colorbar(h,ax=axes[j+1])
                        if i == 4: break
                        i+=1

                plt.suptitle("vmin={:.2f} - vmax={:.2f}".format(labels[0].min(), labels[0].max()), y=0.9)
                writer.add_figure('Labels ' + phase, fig, global_step=epoch)
                plt.close()
                
                # Show confusion matrix for training/test samples
                Y_pred = np.array(Y_pred)
                Y_true = np.array(Y_true)
                adv_mask = Y_true!=-1
                
                acc_patches = np.mean(Y_pred[adv_mask]==Y_true[adv_mask])

                res_dict[phase]['acc'].append(acc_patches)
                                
                fig = get_confusion_fig(Y_pred[adv_mask],Y_true[adv_mask])
                writer.add_figure('Patch Confusion ' + phase, fig, global_step=epoch)
                writer.add_scalar(phase+'/acc_patches', acc_patches, epoch)

                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

                # deep copy the model
                if epoch == self.start_epoch:
                    best_loss = epoch_loss 

                if phase == 'test' and epoch_loss < best_loss and epoch>self.start_epoch:
                    best_loss = epoch_loss 
                    best_model_wts = copy.deepcopy(model.state_dict())

                if phase == 'test':
                    res_dict['test']['loss'].append(epoch_loss)
                    torch.save(model.state_dict(), '{}/model_train.pt'.format(self.save_dir_name))
                   # if epoch%1==0:
                    torch.save(model.state_dict(), '{}/model_train_ep{}.pt'.format(self.save_dir_name, epoch))


        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, res_dict
