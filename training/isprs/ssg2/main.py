# Based on multiprocessing example from
# https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html

from datetime import datetime
import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from ssg2.nn.loss.ftnmt_loss import ftnmt_loss
from ssg2.utils.classification_metric import Classification 


# Debugging flag 
DEBUG=True 

# Here NClasses = 6+1 for ISPRS
def mtsk_loss(preds, labels,criterion, NClasses):
    # Multitasking loss,    segmentation / boundaries/ distance

    pred_segm  = preds[:,:NClasses]                                                                                           
    pred_bound = preds[:,NClasses:2*NClasses]                                                                            
    pred_dists = preds[:,2*NClasses:3*NClasses]                                                                          
                                                                                                                               
                                                                                                                               
                                                                                                                               
    # Multitasking loss                                                                                                            
    label_segm  = labels[:,:NClasses]                                                                                         
    label_bound = labels[:,NClasses:2*NClasses]                                                                          
    label_dists = labels[:,2*NClasses:3*NClasses]                                                                        
                                                                                                                               
                                                                                                                               
    loss_segm  = criterion(pred_segm,   label_segm)                                                   
    loss_bound = criterion(pred_bound, label_bound) 
    loss_dists = criterion(pred_dists, label_dists) 
                                                                                                                               
                                                                                                                               
    return (loss_segm+loss_bound+loss_dists)/3.0                                                                                   



def monitor_epoch(epoch, datagen_valid, NClasses):
    # Computes various classification metrics                                                                                                                  
    metric_target   = Classification(num_classes=6+1).cuda() #                                        
                                                                                                                                         
                                                                                                                                         
    dist.barrier() # Make sure all operations finish until this point                                                                        
                                                                                                                                         
    for idx, data in enumerate(datagen_valid): 
        img1, lst_img_2, labels_lst_inter, labels_lst_union,  labels_lst_diff, labels1_wth_off   = data                                                                                                  
        img1                = img1.cuda(non_blocking=True)
        lst_img_2           = lst_img_2.cuda(non_blocking=True)
        labels1_wth_off     = labels1_wth_off.cuda(non_blocking=True) 

        # preds_target is the output of the sequence model 
        with torch.inference_mode():
            _ , _,_ , preds_target, _, _, _ = model(img1,lst_img_2) 


        # XXXXXXXXXXXXXXXXXXXXX TARGET metrics XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                                      
        pred_segm  = preds_target[:,:NClasses]  # Segmentation only output                                       
        label_segm = labels1_wth_off[:,:NClasses]                                                                
        val_loss_target   += criterion(pred_segm, label_segm).mean() 

        # Update Metric                                                                                               
        metric_target(pred_segm,torch.argmax(label_segm,dim=1) )                                                   

       # DEBUGGING OPTION
       if DEBUG and idx > 5 : 
           break 

    # Evaluate statistics for all predictions 
    metric_kwargs_target = metric_target.compute()

    kwargs{'epoch':epoch}
    for k,v in metric_kwargs_target.items():                               
        kwargs[k+"_target_vV"]=v.cpu().numpy() # Pass to cpu and numpy format 


    return kwargs



def train(args):
    num_epochs = args.epochs
    batch_size = args.batch_size


    dist.init_process_group(backend='nccl')

    torch.manual_seed(0)
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)


    SequenceLength=4 
    NClasses=6+1 # +1 refers tp the NULL class that results from set operations. 
    nf=96
    verbose = dist.get_rank() == 0  # print only on global_rank==0 
    model_config = {'in_channels':5,
                   'spatial_size_init':128,         
                   'depths':[2,2,5,2],              
                   'nfilters_init':nf,              
                   'nfilters_embed':nf,             
                   'nheads_start':nf//4,            
                   'NClasses':NClasses,   
                   'verbose':verbose,               
                   'segm_act':'sigmoid',            
                   'nresblocks':1}                  
 
    # UNet-like model 
    model = ptavit(**model_config)

    # Fractal Tanimoto with complement loss 
    criterion = ftnmt_loss()
    criterion_features = ftnmt_loss(axis=[-3,-2,-1])
    
    # You might need eps = 1.e-6 if you train with mixed precision. It avoids NANs due to overflow. 
    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3, eps=1.e-6)

    model = DistributedDataParallel(model, device_ids=[local_rank])


    transform_train = TrainingTransform(NClasses=NClasses,mode='train')
    transform_valid = TrainingTransform(NClasses=NClasses,mode='valid')

    train_dataset = = RocksDBDataset_SSG2(                                                                                                    
                 flname_db= '../../data/isprs/TRAINING_DBs/F128/train.db', # Change here with location of your database
                 sequence_length = SequenceLength, 
                 transform=transform_train,
                 num_workers=4) # 4 is a good choice for RocksDB                                       
                                                                                                                                                      


    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True,
                              sampler=train_sampler)



    valid_dataset = = RocksDBDataset_SSG2(                                                                                                    
                 flname_db= '../../data/isprs/TRAINING_DBs/F128/valid.db', # Change here with location of your database
                 sequence_length = SequenceLength, 
                 transform=transform_valid,
                 num_workers=4) # 4 is a good choice for RocksDB                                       
                                                                                                                                                      


    valid_sampler = DistributedSampler(valid_dataset)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True,
                              sampler=valid_sampler)


    start = datetime.now()
    for epoch in range(num_epochs):
        tot_loss = 0
        for i, data in enumerate(train_loader):

            # #######################################
            # DEBUGGING OPTION
            if DEBUG and  i > 5 : 
                break 
            # ########################################

            img1, lst_img_2, labels_lst_inter, labels_lst_union, labels_lst_diff, labels1_wth_off = data 

            img1                = img1.cuda(non_blocking=True)
            lst_img_2           = lst_img_2.cuda(non_blocking=True)
            labels_lst_inter    = labels_lst_inter.cuda(non_blocking=True)
            labels_lst_union    = labels_lst_union.cuda(non_blocking=True)
            labels_lst_diff     = labels_lst_diff.cuda(non_blocking=True) 
            labels1_wth_off     = labels1_wth_off.cuda(non_blocking=True) 



            self.opt.zero_grad(set_to_none=True)                                                                                                               
            lst_of_preds_inter, lst_of_preds_union, lst_of_preds_diff, preds_target, preds_target_fz,  preds_null, features = model(img1,lst_img_2)     
                                                                                                                                                   
            tloss = 0.0                                                                                                                                    
            seq_len = lst_of_preds_inter.shape[2]                                                                                                          
            tfactor = 1./(5*seq_len+1.+1.)                                                                                                                 
                                                                                                                                                   
                                                                                                                                                  
            for idx_seq in range(seq_len):                                                                                                                 
                # Intersection losses                                                                                                    
                tpreds = lst_of_preds_inter[:,:,idx_seq]                                                                                                   
                tlabels = labels_lst_inter[:,idx_seq]                                                                                                      
                tloss = tloss +  mtsk_loss(tpreds,tlabels)                                                                     
                                                                                                                                                   
                # Union losses                                                                                                                             
                tpreds  = lst_of_preds_union[:,:,idx_seq]                                                                                                  
                tlabels = labels_lst_union[:,idx_seq]                                                                                                      
                tloss = tloss +  mtsk_loss(tpreds,tlabels)                                                                     
                                                                                                                                                   
                # Diff losses                                                                                                                              
                tpreds  = lst_of_preds_diff[:,:,idx_seq]                                                                                                   
                tlabels = labels_lst_diff[:,idx_seq]                                                                                                       
                tloss = tloss +  mtsk_loss(tpreds,tlabels) 
                                                                                                                                                   
                                                                                                                                                   
                # FZ Correlation loss - Segmentation only                                                                                                  
                tpreds  = preds_target_fz[:,:,idx_seq]                                                                                                     
                tlabels = labels1_wth_off[:,:NClasses]                                                                                                
                tloss = tloss + criterion(tpreds,tlabels)
                                                                                                                                                   
                                                                                                                                                   
                # FZ Correlation loss - Segmentation only                                                                                                  
                tpreds  = preds_null[:,:,idx_seq] # Result of intersection of diff with inter                                                              
                tlabels = torch.zeros_like( labels1_wth_off[:,:NClasses]) 
                tloss   = tloss + criterion(tpreds, tlabels) 
                                                                                                                                                   
                                                                                                                                                   
                                                                                                                                                   
                                                                                                                                                   
            # Target loss -- Predictions of sequence model  
            tloss = tloss + mtsk_loss(preds_target, labels1_wth_off)                                                                
                                                                                                                                                   
                                                                                                                                                   
            # Equal features per time split                                                                                                                
            # Enforces abelian symmetry. 
            features = torch.sigmoid(features) # B x C x T x H x W                                                                                         
            sequence_dim = 2                                                                                                                               
            indices = torch.randperm(features.size(sequence_dim)).to(features.device)                                                                      
            features2 = features.index_select(sequence_dim, indices) # random permutation                                                                                     
            # enforces abelian symmetry, at the expense of memory.                                                                                         
            tloss = tloss +  criterion_features(features, features2, torch.ones_like(features2) ).squeeze().mean(dim=-1)                                   
                                                                                                                                                   
                                                                                                                                                   
            loss = tloss * tfactor                                                                                                                            

            loss.backward()
            optimizer.step()

            tot_loss += loss.item()

        kwargs = monitor_epoch(epoch, valid_loader, NClasses)
        kwargs['tot_train_loss'] = tot_loss
        if verbose:
            output_str = ', '.join(f'{k}:: {v}, |===|, ' for k, v in kwargs.items())
            print(output_str)

    if verbose:
        print("Training completed in: " + str(datetime.now() - start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=2, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch-size', default=2, type=int, metavar='batch', help='batch-size')
    args = parser.parse_args()

    train(args.epochs)


if __name__ == '__main__':
    main()

