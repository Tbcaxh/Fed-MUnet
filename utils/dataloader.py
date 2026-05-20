import os
import pickle
import numpy as np
from collections import OrderedDict
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase

MODALITY_CHANNELS = 4

def crop_or_pad_2d(data, patch_size, center=None):
    patch_size = np.array(patch_size).astype(int)
    _, h, w = data.shape
    if center is None:
        center = np.array([h // 2, w // 2]).astype(int)
    else:
        center = np.array(center).astype(int)

    out = np.zeros((data.shape[0], patch_size[0], patch_size[1]), dtype=data.dtype)
    src_start = center - patch_size // 2
    src_end = src_start + patch_size

    src_h1 = max(src_start[0], 0)
    src_h2 = min(src_end[0], h)
    src_w1 = max(src_start[1], 0)
    src_w2 = min(src_end[1], w)

    dst_h1 = src_h1 - src_start[0]
    dst_h2 = dst_h1 + (src_h2 - src_h1)
    dst_w1 = src_w1 - src_start[1]
    dst_w2 = dst_w1 + (src_w2 - src_w1)

    out[:, dst_h1:dst_h2, dst_w1:dst_w2] = data[:, src_h1:src_h2, src_w1:src_w2]
    return out

def low_freq_mutate_np(amp_src, amp_trg, L=0.01, lam=None):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    _, h, w = a_src.shape
    radius = int(np.floor(np.amin((h, w)) * L))
    c_h = int(np.floor(h / 2.0))
    c_w = int(np.floor(w / 2.0))

    h1 = c_h - radius
    h2 = c_h + radius + 1
    w1 = c_w - radius
    w2 = c_w + radius + 1

    if lam is None:
        lam = np.random.randint(1, 11) / 10.0

    a_src[:, h1:h2, w1:w2] = a_src[:, h1:h2, w1:w2] * (1.0 - lam) + a_trg[:, h1:h2, w1:w2] * lam
    return np.fft.ifftshift(a_src, axes=(-2, -1))

def modality_wise_frequency_interpolation(src_img, trg_img, L=0.01):
    """Apply CFS independently to FLAIR/T1/T1ce/T2 channels."""
    fft_src = np.fft.fft2(src_img, axes=(-2, -1))
    fft_trg = np.fft.fft2(trg_img, axes=(-2, -1))
    amp_src, pha_src = np.abs(fft_src), np.angle(fft_src)
    amp_trg = np.abs(fft_trg)
    amp_src = low_freq_mutate_np(amp_src, amp_trg, L=L)
    fft_mutated = amp_src * np.exp(1j * pha_src)
    image = np.real(np.fft.ifft2(fft_mutated, axes=(-2, -1)))
    return np.clip(image, 0.0, 1.0).astype(np.float32)

class DataLoader2D(SlimDataLoaderBase):
    def __init__(self, data, batch_size, patch_size, max_size, cfs_peers=None, cfs_l=0.01):
        super().__init__(data, batch_size, None)
        self.oversample_foreground_percent = 1/3
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.max_size = max_size
        self.cfs_peers = cfs_peers or []
        self.cfs_l = cfs_l

    def _load_source_patch(self, name, force_fg):
        data = np.load(self._data[name]['path'], allow_pickle=True)
        if force_fg and len(self._data[name]['locs']) > 0:
            locs = self._data[name]['locs']
            cls = np.random.choice(list(locs.keys()))
            indices = locs[cls][:, 0]
            sel_idx = np.random.choice(np.unique(indices))
            data = data[:, sel_idx]
            loc = locs[cls][indices == sel_idx]
            center = loc[np.random.choice(len(loc))][1:]
        else:
            sel_idx = np.random.choice(data.shape[1])
            data = data[:, sel_idx]
            center = np.array(data.shape[1:]) // 2
        return crop_or_pad_2d(data, self.patch_size, center=center)

    def _load_target_image_patch(self):
        if len(self.cfs_peers) == 0:
            return None
        peer_data = self.cfs_peers[np.random.choice(len(self.cfs_peers))]
        target_name = np.random.choice(list(peer_data.keys()))
        data = np.load(peer_data[target_name]['path'], allow_pickle=True)
        sel_idx = np.random.choice(data.shape[1])
        data = data[:, sel_idx]
        center = np.array(data.shape[1:]) // 2
        return crop_or_pad_2d(data[:-1], self.patch_size, center=center)

    def generate_train_batch(self):
        # random select data
        bs = 0
        if self.batch_size > self.max_size:
            bs = self.max_size
        else:
            bs = self.batch_size
        sels = np.random.choice(list(self._data.keys()), bs, True)
        # read data, form slice
        images, cfs_images, labels = [], [], []
        for i, name in enumerate(sels):
            if i < round(bs * (1 - self.oversample_foreground_percent)):
                force_fg = False
            else:
                force_fg = True
            data = self._load_source_patch(name, force_fg)
            image = data[:-1].astype(np.float32)
            target = self._load_target_image_patch()
            if target is None:
                cfs_image = image.copy()
            else:
                cfs_image = modality_wise_frequency_interpolation(image, target, L=self.cfs_l)
            images.append(image)
            cfs_images.append(cfs_image)
            labels.append(data[-1:])
        image = np.stack(images)
        cfs_image = np.stack(cfs_images)
        label = np.stack(labels)
        return {'data': image, 'cfs_data': cfs_image, 'label': label}

def _build_case_dict(config, client_name, case_names):
    dataset = OrderedDict()
    for name in case_names:
        dataset[name] = OrderedDict()
        dataset[name]['client'] = client_name
        dataset[name]['path'] = os.path.join(config.DATASET.ROOT, client_name, name + '.npy')
        with open(os.path.join(config.DATASET.ROOT, client_name, name + ".pkl"), 'rb') as f:
            dataset[name]['locs'] = pickle.load(f)
    return dataset

def get_trainloader(config, num_clients, exclude_clients=None):
    with open(os.path.join(config.SPLIT.ROOT, 'split_data.pkl'), 'rb') as f:
        splits = pickle.load(f)

    if 'clients' not in splits:
        raise RuntimeError('split_data.pkl must contain client-aware splits. Run make_data.py first.')

    exclude_clients = set(exclude_clients or [])
    client_names = [name for name in splits['clients'].keys() if name not in exclude_clients]
    client_names = client_names[:num_clients]
    client_datasets = []
    for client_name in client_names:
        train_cases = splits['clients'][client_name]['train']
        client_datasets.append(_build_case_dict(config, client_name, train_cases))

    cfs_l = getattr(config.TRAIN, 'CFS_L', 0.01)
    dataloader_list = []
    for idx, dataset in enumerate(client_datasets):
        peers = [client_datasets[j] for j in range(len(client_datasets)) if j != idx]
        dataloader_list.append(
            DataLoader2D(
                dataset,
                config.TRAIN.BATCH_SIZE,
                config.TRAIN.PATCH_SIZE,
                max(1, len(dataset)),
                cfs_peers=peers,
                cfs_l=cfs_l
            )
        )
    return dataloader_list
