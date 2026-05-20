import torch
from config.config import config
from model.utils.utils import save_checkpoint
from model.utils.function import inference

device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')


class Server(object):

    def __init__(self, conf, model, logger, device):

        self.conf = conf
        self.global_model = model
        self.epoch = 0
        self.best_model = False
        self.best_perf = 0.0
        self.logger = logger
        self.global_weights = {}
        self.device = device

    def model_aggregate(self, weight_accumulator):

        for name, _ in self.global_model.named_parameters():
            sigma = self.conf['sigma']
            if torch.cuda.is_available():
                noise = torch.cuda.FloatTensor(weight_accumulator[name].shape).normal_(0, sigma).to(self.device)
            else:
                noise = torch.FloatTensor(weight_accumulator[name].shape).normal_(0, sigma).to(self.device)
            weight_accumulator[name] = weight_accumulator[name].to(self.device) + noise
        self.global_model.load_state_dict(weight_accumulator)

    def model_eval(self):

        eval_split = self.conf.get('eval_split', 'val')
        perf = inference(self.global_model, self.logger, config, eval_split, self.device)
        if perf > self.best_perf:
            self.best_perf = perf
            self.best_model = True
        else:
            self.best_model = False

        save_checkpoint({
            'epoch': self.epoch + 1,
            'state_dict': self.global_model.state_dict(),
            'perf': perf
        }, self.best_model, config.OUTPUT_DIR, filename='checkpoint.pth')

        if self.conf.get('log_unseen_during_training', False) and getattr(config, 'UNSEEN_CLIENTS', None):
            source_eval_clients = getattr(config, 'EVAL_CLIENTS', None)
            config.EVAL_CLIENTS = config.UNSEEN_CLIENTS
            target_split = self.conf.get('target_eval_split', 'test')
            target_perf = inference(self.global_model, self.logger, config, target_split, self.device)
            self.logger.info('unseen %s perf: %s', target_split, target_perf)
            config.EVAL_CLIENTS = source_eval_clients

        self.epoch = self.epoch + 1
