import torch
from model.utils import norm
from model.utils.utils import AverageMeter
from torchvision.utils import save_image
from config.config import config
from model.seg.MUnet import MUnet
from model.seg.UNet import UNet
import torch.nn as nn


stages = config.MODEL.STAGES
midc = config.MODEL.MIDCHANNEL

class Client(object):

    def __init__(self, conf, train_dataset, criterion, scales, config, logger, c, device, train_size=1):
        self.conf = conf

        #self.local_model = MUnet(inc=4, outc=4, midc=midc, stages=stages)
        self.local_model = UNet(inc=4, outc=4, midc=midc, stages=stages)
        self.local_model = nn.DataParallel(self.local_model, config.TRAIN.DEVICES).to(device)

        self.train_dataset = train_dataset

        self.criterion = criterion
        self.scales = scales
        self.config = config
        self.logger = logger
        self.index = c
        self.train_size = train_size

    def _train_step(self, image, labels, optim_seg):
        outs = self.local_model(image)
        loss = self.criterion(outs, labels)
        optim_seg.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), 12)
        optim_seg.step()
        return loss, outs

    def _apply_flat_clip(self, global_model):
        if not self.conf.get("FlatClip", False):
            return
        model_norm = norm.model_norm(global_model, self.local_model)
        if model_norm == 0:
            return
        norm_scale = min(1, self.conf['C'] / model_norm)
        for name, layer in self.local_model.named_parameters():
            global_param = global_model.state_dict()[name].to(layer.device)
            clipped_difference = norm_scale * (layer.data - global_param)
            layer.data.copy_(global_param + clipped_difference)

    def local_train(self, model, optim ,schedule, device):
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone().to(self.local_model.state_dict()[name].device))

        optim_seg = optim
        sched_seg = schedule

        print_freq = 10
        lsegs = AverageMeter()
        lcfs = AverageMeter()
        self.local_model.train()
        method = self.conf.get("method", "FedAvg")
        cfs_weight = self.conf.get("cfs_weight", 1.0)

        for e in range(self.conf["local_epochs"]):
            self.logger.info('learning rate : {}'.format(optim_seg.param_groups[0]['lr']))
            num_iter = self.config.TRAIN.NUM_BATCHES
            for i in range(num_iter):
                data_dict = next(self.train_dataset)
                image = data_dict['data'].to(device)
                cfs_image = data_dict['cfs_data'].to(device)
                labels = data_dict['label']

                labels = [label.to(device) for label in labels]

                if method == "EpisodicCFS":
                    lseg, outs = self._train_step(image, labels, optim_seg)
                    self._apply_flat_clip(model)
                    lcfs_loss, _ = self._train_step(cfs_image, labels, optim_seg)
                    loss = lseg + cfs_weight * lcfs_loss
                else:
                    outs = self.local_model(image)
                    lseg = self.criterion(outs, labels)
                    if method == "CFS":
                        cfs_outs = self.local_model(cfs_image)
                        lcfs_loss = self.criterion(cfs_outs, labels)
                        loss = lseg + cfs_weight * lcfs_loss
                    else:
                        lcfs_loss = torch.zeros_like(lseg)
                        loss = lseg

                    optim_seg.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), 12)
                    optim_seg.step()

                lsegs.update(lseg.item(), self.config.TRAIN.BATCH_SIZE)
                lcfs.update(lcfs_loss.item(), self.config.TRAIN.BATCH_SIZE)
                if i % print_freq == 0:
                    msg = 'client {0} epoch: [{1}][{2}/{3}]\t' \
                          'lseg {lseg.val:.3f} ({lseg.avg:.3f})\t' \
                          'lcfs {lcfs.val:.3f} ({lcfs.avg:.3f})\t'.format(
                         self.index, e, i, num_iter,
                         lseg=lsegs, lcfs=lcfs,
                     )
                    self.logger.info(msg)
                    bs = image.shape[0]
                    image = torch.cat(torch.split(image, 1, 1))
                    label = torch.cat(torch.split(labels[-1], 1, 1))
                    out = torch.argmax(torch.softmax(outs[-1], 1), dim=1, keepdim=True)
                    out = torch.cat(torch.split(out, 1, 1))
                    save_image(torch.cat([image, label, out], dim=0).data.to(device), f'tmp/train.png', nrow=bs,
                               scale_each=True, normalize=True)
                self._apply_flat_clip(model)
            sched_seg.step()
            print("Client {0} Epoch {1} done." .format(self.index, e))
        if self.conf.get("FlatClip", False):
            diff = dict()
            for name, data in self.local_model.state_dict().items():
                diff[name] = (data - model.state_dict()[name].to(data.device)) * self.conf["lambda"]
            return {k: v.cpu().clone() for k, v in model.state_dict().items()}, diff
        else:
            diff = {k: torch.zeros_like(v).cpu() for k, v in self.local_model.state_dict().items()}
            return {k: v.cpu().clone() for k, v in self.local_model.state_dict().items()}, diff
