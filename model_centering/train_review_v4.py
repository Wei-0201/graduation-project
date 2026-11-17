"""Group-held-out correction of human-reviewed curve targets."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from train_local_centering_v2 import CenteringDataset, stratified_split, save_jetson
from train_local_round2 import valid_images

ROOT=Path(__file__).resolve().parent


class ReviewDataset(Dataset):
    def __init__(self,records,augment=False,horizontal_crop=False):
        self.records=records
        self.augment=augment
        self.horizontal_crop=horizontal_crop
        self.jitter=transforms.ColorJitter(.15,.15,.12,.025)

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        r=self.records[index]
        with Image.open(r['path']) as source:image=source.convert('RGB')
        x,y=float(r['x']),float(r['y'])
        if self.augment:
            if random.random()<.5:
                image=ImageOps.mirror(image);x=319-x
            if self.horizontal_crop:
                left=random.randint(0,min(24,int(x)))
                right=random.randint(0,min(24,int(319-x)))
                if left or right:
                    image=image.crop((left,0,320-right,240))
                    x=(x-left)*320/(320-left-right)
            image=self.jitter(image)
        image=transforms.functional.resize(image,(224,224))
        tensor=transforms.functional.to_tensor(image)
        tensor=torch.from_numpy(tensor.numpy()[::-1].copy())
        tensor=transforms.functional.normalize(tensor,[.485,.456,.406],[.229,.224,.225])
        return tensor,torch.tensor([(x-160)/160,(y-120)/120],dtype=torch.float32)


def predictions(model,loader,device):
    model.eval();pred=[];target=[]
    with torch.no_grad():
        for images,labels in loader:
            pred.append(model(images.to(device)).cpu().numpy());target.append(labels.numpy())
    return np.concatenate(pred),np.concatenate(target)


def metrics(pred,target):
    error=(pred[:,0]-target[:,0])*160
    clear=np.abs(target[:,0]*160)>20
    return dict(n=len(error),x_mae_px=float(np.abs(error).mean()),
        x_p90_px=float(np.percentile(np.abs(error),90)),x_mse=float(np.mean((error/160)**2)),
        clear_targets=int(clear.sum()),wrong_direction=int(np.sum(
            (pred[clear,0]*target[clear,0])<0)),
        left_mae_px=float(np.abs(error[target[:,0]<0]).mean()),
        right_mae_px=float(np.abs(error[target[:,0]>=0]).mean()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'best_steering_model_xy_review_v4_20261025.pth')
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--review-repeat',type=int,default=6)
    parser.add_argument('--height-matched-labels',action='store_true')
    parser.add_argument('--horizontal-crop',action='store_true')
    args=parser.parse_args()
    if args.output.exists():parser.error('output already exists')
    random.seed(20261025);np.random.seed(20261025);torch.manual_seed(20261025)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():parser.error('CUDA unavailable')
    device=torch.device('cuda');torch.backends.cudnn.benchmark=True
    checkpoint=ROOT/'best_steering_model_xy_centering_y003.pth'
    records=json.loads((ROOT/'curve_review_20261025/labeled_manifest.json').read_text(encoding='utf-8'))
    holdout={('bend_exit_v2.avi',2),('bend_exit_v2.avi',4)}
    validation=[r for r in records if (r['source'],r['run']) in holdout]
    train=[r for r in records if (r['source'],r['run']) not in holdout]
    train_hashes={hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() for r in train}
    assert not any(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() in train_hashes
                   for r in validation),'identical images cross review split'
    old_paths,invalid=valid_images(ROOT/'dataset_xy_full_20261021')
    old_train,old_val=stratified_split(old_paths)
    if args.height_matched_labels:
        old_train=[p for p in old_train if 105<=int(Path(p).stem.split('_')[2])<=135]
        old_val=[p for p in old_val if 105<=int(Path(p).stem.split('_')[2])<=135]
    bins=[[] for _ in range(5)]
    for path in old_train:bins[min(4,int(Path(path).stem.split('_')[1])//64)].append(path)
    rng=random.Random(43);anchors=[]
    for group in bins:anchors.extend(rng.sample(group,min(103,len(group))))
    model=models.resnet18(pretrained=False);model.fc=nn.Linear(512,2)
    model.load_state_dict(torch.load(str(checkpoint),map_location='cpu'));model.to(device)
    val_loader=DataLoader(ReviewDataset(validation),batch_size=32,shuffle=False,num_workers=0)
    old_loader=DataLoader(CenteringDataset(old_val),batch_size=32,shuffle=False,num_workers=0)
    val_pred,val_target=predictions(model,val_loader,device)
    old_pred,old_target=predictions(model,old_loader,device)
    baseline=metrics(val_pred,val_target);baseline_old=metrics(old_pred,old_target)
    print('GPU:',torch.cuda.get_device_name(0),flush=True)
    print('Review train={} held-out={} old anchors={} old validation={}'.format(
        len(train),len(validation),len(anchors),len(old_val)),flush=True)
    print('Baseline review:',baseline,flush=True);print('Baseline old:',baseline_old,flush=True)
    if args.height_matched_labels:
        anchor_records=[dict(path=p,x=int(Path(p).stem.split('_')[1]),
                             y=int(Path(p).stem.split('_')[2])) for p in anchors]
    else:
        teacher_loader=DataLoader(CenteringDataset(anchors),batch_size=32,shuffle=False,num_workers=0)
        teacher,_=predictions(model,teacher_loader,device)
        anchor_records=[dict(path=p,x=float(np.clip((xy[0]+1)*160,0,319)),
                             y=float(np.clip((xy[1]+1)*120,0,239)))
                        for p,xy in zip(anchors,teacher)]
    loader=DataLoader(ReviewDataset(train*args.review_repeat+anchor_records,True,args.horizontal_crop),
                      batch_size=16,shuffle=True,num_workers=0,pin_memory=True)
    for name,param in model.named_parameters():
        param.requires_grad=name.startswith(('layer3.','layer4.','fc.'))
    optimizer=torch.optim.AdamW([
        {'params':model.layer3.parameters(),'lr':.00002},
        {'params':model.layer4.parameters(),'lr':.00004},
        {'params':model.fc.parameters(),'lr':.00008}],weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    scaler=torch.cuda.amp.GradScaler()
    report=dict(checkpoint=str(checkpoint),baseline=baseline,baseline_old=baseline_old,
        review_train_ids=[r['id'] for r in train],review_validation_ids=[r['id'] for r in validation],
        anchors=anchors,old_validation=old_val,epochs=[],best_epoch=None,
        height_matched_labels=args.height_matched_labels,horizontal_crop=args.horizontal_crop,
        notes='Whole driving runs held out; mirrors are made only after splitting; BN statistics frozen.')
    best_score=baseline['x_mae_px'];best=None;stale=0;started=time.monotonic()
    baseline_path=args.output.with_suffix('.baseline_predictions.json')
    baseline_path.write_text(json.dumps(dict(ids=[r['id'] for r in validation],
        target=val_target.tolist(),baseline_prediction=val_pred.tolist()),indent=2),encoding='utf-8')
    for epoch in range(1,args.epochs+1):
        model.train()
        for module in model.modules():
            if isinstance(module,nn.BatchNorm2d):module.eval()
        loss_sum=0.;count=0
        for images,labels in loader:
            images=images.to(device,non_blocking=True);labels=labels.to(device,non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred=model(images)
                loss=F.smooth_l1_loss(pred[:,0],labels[:,0],beta=.1)
                loss=loss+.03*F.smooth_l1_loss(pred[:,1],labels[:,1],beta=.1)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            loss_sum+=loss.item()*len(labels);count+=len(labels)
        scheduler.step()
        vp,vt=predictions(model,val_loader,device);op,ot=predictions(model,old_loader,device)
        current=metrics(vp,vt);old=metrics(op,ot)
        keep_old=old['x_mae_px']<=baseline_old['x_mae_px']+max(2.,baseline_old['x_mae_px']*.20)
        keep_sides=(current['left_mae_px']<=baseline['left_mae_px']+2 and
                    current['right_mae_px']<=baseline['right_mae_px']+2)
        score=current['x_mae_px']+.5*max(0.,old['x_mae_px']-baseline_old['x_mae_px'])
        accepted=keep_old and keep_sides and score<best_score
        log=dict(epoch=epoch,train_loss=loss_sum/count,review=current,old=old,
                 acceptable_retention=keep_old,acceptable_sides=keep_sides,saved=accepted)
        report['epochs'].append(log)
        print('Epoch {}/{} loss={:.5f} review_MAE={:.2f}px L={:.2f} R={:.2f} wrong={}/{} old_MAE={:.2f}px accept={} elapsed={:.0f}s'.format(
            epoch,args.epochs,loss_sum/count,current['x_mae_px'],current['left_mae_px'],current['right_mae_px'],
            current['wrong_direction'],current['clear_targets'],old['x_mae_px'],accepted,time.monotonic()-started),flush=True)
        if accepted:
            save_jetson(model,args.output);best_score=score;best=current;stale=0
            report['best_epoch']=epoch;report['best_review']=current;report['best_old']=old
            np.savez(str(args.output)+'.predictions.npz',prediction=vp,target=vt,old_prediction=op,old_target=ot)
        else:stale+=1
        args.output.with_suffix('.report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if best is not None and best['x_mse']<=.0075 and best['x_mae_px']<=baseline['x_mae_px']*.8:
            print('Reached validation X MSE <=0.0075 with retention and >=20% MAE improvement.',flush=True);break
        if stale>=8:print('Early stop: no acceptable improvement in 8 epochs.',flush=True);break
    if best is None:raise RuntimeError('No checkpoint passed review improvement and old-data retention gates')
    print('BEST',args.output,'epoch',report['best_epoch'],report['best_review'],flush=True)


if __name__=='__main__':main()
