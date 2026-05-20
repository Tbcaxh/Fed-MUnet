import torch
import os, pickle
import numpy as np
import torch.nn as nn
from medpy.metric.binary import dc
from model.utils.utils import AverageMeter
import random
from utils.dataloader import crop_or_pad_2d

def dice_binary(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    pred_sum = np.count_nonzero(pred)
    target_sum = np.count_nonzero(target)
    if pred_sum == 0 and target_sum == 0:
        return 1.0
    if pred_sum == 0 or target_sum == 0:
        return 0.0
    return 2.0 * np.count_nonzero(pred & target) / float(pred_sum + target_sum)

@torch.no_grad()
def inference(model, logger, config, dataset, device):
    print("-----------------inference------------------")
    model.eval().to(device)
    perfs = {'WT': AverageMeter(), 'ET': AverageMeter(), 'TC': AverageMeter()}
    nonline = nn.Softmax(dim=1)
    with open(os.path.join(config.SPLIT.ROOT, 'split_data.pkl'), 'rb') as f:
        splits = pickle.load(f)

    valids = splits[dataset]
    eval_clients = set(getattr(config, 'EVAL_CLIENTS', []))
    if eval_clients:
        valids = [item for item in valids if isinstance(item, dict) and item['client'] in eval_clients]
    valids = random.sample(valids, min(50, len(valids)))
    logger.info('evaluating split %s clients %s cases %d', dataset, sorted(eval_clients) if eval_clients else 'all', len(valids))

    for item in valids:
        if isinstance(item, dict):
            data_path = os.path.join(config.DATASET.ROOT, item['client'], item['case'] + '.npy')
        else:
            data_path = os.path.join(config.DATASET.ROOT, item + '.npy')
        data = np.load(data_path)
        slices = []
        for slice_idx in range(data.shape[1]):
            slice_data = data[:, slice_idx]
            center = np.array(slice_data.shape[1:]) // 2
            slices.append(crop_or_pad_2d(slice_data, config.TRAIN.PATCH_SIZE, center=center))
        data = np.stack(slices, axis=1)
        # run inference
        image = torch.from_numpy(data[:-1]).permute(1, 0, 2, 3).to(device)
        label = data[-1]
        out_list = [model(torch.tensor(np.expand_dims(image[i].cpu(), 0)).to(device)) for i in range(image.shape[0])]
        out_list = torch.cat(out_list, dim=0)
        out_list = nonline(out_list)
        pred = torch.argmax(out_list, dim=1).cpu().numpy()
        # quantitative analysis
        perfs['WT'].update(dice_binary(pred > 0, label > 0))
        perfs['ET'].update(dice_binary(pred == 3, label == 3))
        perfs['TC'].update(dice_binary(pred >= 2, label >= 2))
    for c in perfs.keys():
        logger.info(f'class {c} dice mean: {perfs[c].avg}')
    logger.info('------------ ----------- ------------')
    perf = np.mean([perfs[c].avg for c in perfs.keys()])
    return perf
