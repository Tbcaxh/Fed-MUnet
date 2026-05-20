import random
import os, pickle, argparse
import numpy as np
import SimpleITK as sitk
from collections import OrderedDict
from scipy.ndimage import binary_fill_holes

DATA_PATH = './data/raw'
DATASET_ROOT = './dataset/data'
PROCESSED_PATH = os.path.join(DATASET_ROOT, 'proceed')
MODALITIES = ['flair', 't1', 't1ce', 't2', 'seg']

def get_bbox(inp):
    coords = np.where(inp != 0)
    minz = np.min(coords[0])
    maxz = np.max(coords[0]) + 1
    minx = np.min(coords[1])
    maxx = np.max(coords[1]) + 1
    miny = np.min(coords[2])
    maxy = np.max(coords[2]) + 1
    return slice(minz, maxz), slice(minx, maxx), slice(miny, maxy)

def convert_seg(seg):
    """ convert brats labels from {0, 1, 2, 4} to {0, 1, 2, 3} """
    new_seg = np.zeros_like(seg)
    new_seg[seg == 4] = 3
    new_seg[seg == 2] = 1
    new_seg[seg == 1] = 2
    return new_seg

def find_modality_file(case_path, case_name, modality):
    candidates = [
        os.path.join(case_path, f'{case_name}_{modality}.nii.gz'),
        os.path.join(case_path, f'{case_name}_{modality}.nii'),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
        if os.path.isdir(candidate):
            nii_files = sorted(
                os.path.join(candidate, filename)
                for filename in os.listdir(candidate)
                if filename.endswith(('.nii', '.nii.gz'))
            )
            if nii_files:
                return nii_files[0]
    return None

def get_case_files(data_path, name):
    case_path = os.path.join(data_path, name)
    files = {}
    missing = []
    for modality in MODALITIES:
        filepath = find_modality_file(case_path, name, modality)
        if filepath is None:
            missing.append(modality)
        else:
            files[modality] = filepath
    return files, missing

def discover_clients(data_path):
    clients = OrderedDict()
    for client_name in sorted(os.listdir(data_path)):
        client_path = os.path.join(data_path, client_name)
        if not os.path.isdir(client_path):
            continue
        case_names = sorted(
            name for name in os.listdir(client_path)
            if os.path.isdir(os.path.join(client_path, name))
        )
        if case_names:
            clients[client_name] = case_names
    return clients

def convert(data_path, out_path):

    os.makedirs(out_path, exist_ok=True)
    clients = discover_clients(data_path)
    if clients:
        client_cases = clients
    else:
        names = sorted(name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name)))
        client_cases = OrderedDict([('client0', names)])

    converted = 0
    skipped = []
    for client_name, names in client_cases.items():
        client_in_path = os.path.join(data_path, client_name) if clients else data_path
        client_out_path = os.path.join(out_path, client_name)
        os.makedirs(client_out_path, exist_ok=True)

        for name in names:
            files, missing = get_case_files(client_in_path, name)
            if missing:
                skipped.append((client_name, name, missing))
                continue

            flair = sitk.GetArrayFromImage(sitk.ReadImage(files['flair']))
            t1 = sitk.GetArrayFromImage(sitk.ReadImage(files['t1']))
            t1ce = sitk.GetArrayFromImage(sitk.ReadImage(files['t1ce']))
            t2 = sitk.GetArrayFromImage(sitk.ReadImage(files['t2']))
            seg = sitk.GetArrayFromImage(sitk.ReadImage(files['seg']))
            img = np.stack([flair, t1, t1ce, t2]).astype(np.float32)
            seg = convert_seg(seg)
            # crop foreground regions
            mask = np.zeros_like(seg).astype(bool)
            for i in range(len(img)):
                mask = mask | (img[i] != 0)
            mask = binary_fill_holes(mask)
            bbox = get_bbox(mask)
            img = img[:, bbox[0], bbox[1], bbox[2]]
            seg = seg[bbox[0], bbox[1], bbox[2]]
            mask = mask[bbox[0], bbox[1], bbox[2]]
            # normalization
            for i in range(len(img)):
                denom = img[i][mask].max() - img[i][mask].min()
                if denom > 0:
                    img[i][mask] = (img[i][mask] - img[i][mask].min()) / denom
                img[i][mask == 0] = 0
            # compensate label imbalance
            approx_nsamp = 10000
            samp_locs = OrderedDict()
            for cls in [1, 2, 3]:
                locs = np.argwhere(seg == cls)
                if len(locs) == 0:
                    continue
                nsamp = min(approx_nsamp, len(locs))
                nsamp = max(nsamp, int(np.ceil(0.1 * len(locs))))
                samp = locs[random.sample(range(len(locs)), nsamp)]
                samp_locs[cls] = samp
            data = np.concatenate([img, seg[None]])
            np.save(os.path.join(client_out_path, f'{name}.npy'), data)
            with open(os.path.join(client_out_path, f'{name}.pkl'), 'wb') as f:
                pickle.dump(samp_locs, f)
            converted += 1

    print(f"Converted {converted} cases to {out_path}")
    if skipped:
        print(f"Skipped {len(skipped)} incomplete cases:")
        for client_name, name, missing in skipped:
            print(f"  {client_name}/{name}: missing {', '.join(missing)}")

def data_split(directory, output_dir=DATASET_ROOT, unseen_client=None):

    os.makedirs(output_dir, exist_ok=True)

    client_dirs = sorted(
        name for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    )

    data_split = {'clients': OrderedDict(), 'train': [], 'val': [], 'test': []}
    for client_name in client_dirs:
        client_dir = os.path.join(directory, client_name)
        pkl_files = [os.path.splitext(f)[0] for f in os.listdir(client_dir) if f.endswith('.pkl')]
        random.shuffle(pkl_files)

        if unseen_client is not None and client_name == unseen_client:
            train_files = []
            val_files = []
            test_files = pkl_files
        else:
            total_files = len(pkl_files)
            train_size = int(total_files * 0.8)

            train_files = pkl_files[:train_size]
            val_files = pkl_files[train_size:]
            test_files = []

        data_split['clients'][client_name] = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            data_split[split_name].extend({'client': client_name, 'case': case_name} for case_name in files)

    output_file1 = os.path.join(output_dir, 'split_data.pkl')
    output_file2 = os.path.join(output_dir, 'split_data.npy')

    np.save(output_file2, data_split)
    with open(output_file1, 'wb') as f:
        pickle.dump(data_split, f)

    print(f"Data split saved to {output_file1}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare FeTS client-aware data for Fed-MUnet.')
    parser.add_argument('--unseen_client', type=str, default=None,
                        help='Client held out as unseen target. Its cases are all assigned to test.')
    args = parser.parse_args()

    convert(DATA_PATH, PROCESSED_PATH)
    data_split(PROCESSED_PATH, unseen_client=args.unseen_client)
