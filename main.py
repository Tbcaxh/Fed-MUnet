import argparse, json
import random
import numpy as np
from server import *
from client import *
import torch
from config.config import config
import torch.backends.cudnn as cudnn
from model.utils.loss import DiceCELoss, MultiOutLoss
from model.utils.scheduler import PolyScheduler
from model.utils.utils import create_logger, setup_seed
from model.seg.MUnet import MUnet
from model.seg.UNet import UNet
from utils.dataloader import get_trainloader
from utils.augmenter import get_train_generator
import sys
sys.setrecursionlimit(10**7)
import os

def load_splits(config):
    split_path = os.path.join(config.SPLIT.ROOT, 'split_data.pkl')
    import pickle
    with open(split_path, 'rb') as f:
        return pickle.load(f)

def resolve_client_name(conf, config, client_names):
    unseen_client = conf.get("unseen_client", None)
    if unseen_client is None:
        return None
    if isinstance(unseen_client, int):
        return client_names[unseen_client]
    if unseen_client not in client_names:
        raise ValueError("unseen_client '{}' is not in split clients: {}".format(unseen_client, client_names))
    return unseen_client


device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf', default="./config/conf.json")
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf = json.load(f)

    setup_seed(config.SEED)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    stages = config.MODEL.STAGES

    midc = config.MODEL.MIDCHANNEL
    #segnet = MUnet(inc=4, outc=4, midc=midc, stages=stages)
    segnet = UNet(inc=4, outc=4, midc=midc, stages=stages)
    segnet = nn.DataParallel(segnet, config.TRAIN.DEVICES).to(device)

    scales = [1 / 2 ** i for i in range(stages)][::-1]
    criterion = MultiOutLoss(DiceCELoss(), weights=scales)
    splits = load_splits(config)
    all_client_names = list(splits['clients'].keys())
    unseen_client = resolve_client_name(conf, config, all_client_names)
    source_clients = [name for name in all_client_names if name != unseen_client]
    if unseen_client is not None:
        unseen_split = splits['clients'][unseen_client]
        if len(unseen_split.get('train', [])) > 0 or len(unseen_split.get('val', [])) > 0:
            raise RuntimeError(
                "split_data.pkl still contains train/val cases for unseen client '{}'. "
                "Regenerate it with: python3 make_data.py --unseen_client {}".format(
                    unseen_client, unseen_client
                )
            )
        if len(unseen_split.get('test', [])) == 0:
            raise RuntimeError("unseen client '{}' has no test cases in split_data.pkl".format(unseen_client))
        config.SOURCE_CLIENTS = source_clients
        config.UNSEEN_CLIENTS = [unseen_client]
        # Use source validation for checkpoint selection. Do not select models on the unseen site.
        config.EVAL_CLIENTS = source_clients
    trainloader = get_trainloader(
        config,
        conf["no_models"],
        exclude_clients=[unseen_client] if unseen_client is not None else None
    )
    train_generator = get_train_generator(trainloader, scales, num_workers=config.NUM_WORKERS)

    os.makedirs('./seg/log', exist_ok=True)
    os.makedirs('./tmp', exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    logger = create_logger('./seg/log', 'train.log')

    with open(args.conf, 'r') as f:
        conf = json.load(f)

    server = Server(conf, segnet, logger, device)
    clients = []

    for c in range(len(train_generator)):
        train_size = getattr(trainloader[c], 'max_size', 1)
        clients.append(Client(conf, train_generator[c], criterion, scales, config, logger, c, device, train_size))

    for e in range(conf["global_epochs"]):
        candidates = random.sample(clients, min(conf["k"], len(clients)))

        weight_accumulator = {}
        client_params = []
        lens = len(candidates)
        total_train_size = sum(max(1, c.train_size) for c in candidates)
        aggregation_weights = [max(1, c.train_size) / total_train_size for c in candidates]

        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)
        diff_list = []

        for c in candidates:
            optim_seg = torch.optim.SGD(c.local_model.parameters(), lr=config.TRAIN.LR * np.exp(- config.TRAIN.LAMDA * e),
                                  weight_decay=config.TRAIN.WEIGHT_DECAY,
                                  momentum=0.95, nesterov=True)
            sched_seg = PolyScheduler(optim_seg, t_total=conf["local_epochs"])
            params, diff = c.local_train(server.global_model, optim_seg, sched_seg, device)
            client_params.append(params)
            diff_list.append(diff)
        for key in client_params[0].keys():
            first_value = client_params[0][key]
            if torch.is_floating_point(first_value):
                weight_accumulator[key] = sum(
                    aggregation_weights[i] * (client_params[i][key].to(device) + diff_list[i][key].to(device))
                    for i in range(lens)
                )
            else:
                weight_accumulator[key] = first_value.to(device)

        server.model_aggregate(weight_accumulator)
        server.model_eval()
        print("Global Epoch {0} done.".format(e))
