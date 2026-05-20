import torch
import os, pickle
import numpy as np
import torch.nn as nn
import statistics as stat
from config.config import config
from model.seg.MUnet import MUnet
from medpy.metric.binary import *
from model.utils.function import inference
import torch.backends.cudnn as cudnn
from model.utils.utils import create_logger, setup_seed
from utils.dataloader import crop_or_pad_2d

device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

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
def inference(model, logger, config, dataset, metrics):
    model.eval().to(device)
    perfs = {}

    for metric in metrics:
        perfs[metric.__name__] = {'WT': [], 'ET': [], 'TC': []}
    nonline = nn.Softmax(dim=1)

    with open(os.path.join(config.SPLIT_eva.ROOT, 'split_data.pkl'), 'rb') as f:
        splits = pickle.load(f)

    valids = splits[dataset]
    eval_clients = set(getattr(config, 'EVAL_CLIENTS', []))
    if eval_clients:
        valids = [item for item in valids if isinstance(item, dict) and item['client'] in eval_clients]
    for item in valids:
        if isinstance(item, dict):
            data_path = os.path.join(config.DATASET_eva.ROOT, item['client'], item['case'] + '.npy')
        else:
            data_path = os.path.join(config.DATASET_eva.ROOT, item + '.npy')
        data = np.load(data_path)
        slices = []
        for slice_idx in range(data.shape[1]):
            slice_data = data[:, slice_idx]
            center = np.array(slice_data.shape[1:]) // 2
            slices.append(crop_or_pad_2d(slice_data, config.TRAIN.PATCH_SIZE, center=center))
        data = np.stack(slices, axis=1)

        image = torch.from_numpy(data[:-1]).permute(1, 0, 2, 3).to(device)
        label = data[-1]
        out_list = [model(torch.tensor(np.expand_dims(image[i].cpu(), 0)).to(device)) for i in range(image.shape[0])]
        out_list = torch.cat(out_list, dim=0)

        out = nonline(out_list)
        pred = torch.argmax(out, dim=1).cpu().numpy()

        mask = image.permute(1, 0, 2, 3)[0].cpu().numpy() != 0

        for metric in metrics:
            predi = pred if metric.__name__ == 'hd95' else pred[mask]
            labeli = label if metric.__name__ == 'hd95' else label[mask]
            if metric.__name__ == 'dc':
                perfs[metric.__name__]['WT'].append(dice_binary(predi > 0, labeli > 0))
                perfs[metric.__name__]['ET'].append(dice_binary(predi == 3, labeli == 3))
                perfs[metric.__name__]['TC'].append(dice_binary(predi >= 2, labeli >= 2))
            else:
                if np.any(labeli > 0) and np.any(predi > 0):
                    perfs[metric.__name__]['WT'].append(metric(predi > 0, labeli > 0))
                if np.any(labeli == 3) and np.any(predi == 3):
                    perfs[metric.__name__]['ET'].append(metric(predi == 3, labeli == 3))
                if np.any(labeli >= 2) and np.any(predi >= 2):
                    perfs[metric.__name__]['TC'].append(metric(predi >= 2, labeli >= 2))
    for metric in perfs.keys():
        et = perfs[metric]['ET']
        tc = perfs[metric]['TC']
        wt = perfs[metric]['WT']
        logger.info(f'------------ {metric} ------------')
        print(f'ET mean / std: {stat.mean(et)} / {stat.stdev(et)}')
        logger.info(f'ET mean / std: {stat.mean(et)} / {stat.stdev(et)}')
        logger.info(f'TC mean / std: {stat.mean(tc)} / {stat.stdev(tc)}')
        logger.info(f'WT mean / std: {stat.mean(wt)} / {stat.stdev(wt)}')

def main():
    setup_seed(config.SEED)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = MUnet(inc=4, outc=4, midc=16, stages=config.MODEL.STAGES)
    model = nn.DataParallel(model, config.TRAIN.DEVICES)
    model.load_state_dict(torch.load('./experiments/model_best.pth', map_location=torch.device('cpu')))

    logger = create_logger('log', 'test.log')
    inference(model, logger, config, dataset='test', metrics=[dc, jc, hd95, sensitivity, precision, specificity])

if __name__ == '__main__':
    main()
