from pathlib import Path

import torch
from tqdm import tqdm

from eda import GraphTileDataset

if __name__ == '__main__':
    ds_train = GraphTileDataset(Path('data/npz_all/npz/tile'), 'train')
    node_fts = []
    conf_fts = []
    for itm in tqdm(ds_train):
        node_fts.extend(itm['node_feat'])
        conf_fts.extend(itm['config_feat'])
    dct = {
        'node_mean': torch.stack(node_fts).mean(dim=0),
        'node_std': torch.stack(node_fts).std(dim=0),
        'config_mean': torch.stack(conf_fts).mean(dim=0),
        'config_std': torch.stack(conf_fts).std(dim=0)
    }
    dct['node_std'][dct['node_std'] == 0] = 1
    dct['config_std'][dct['config_std'] == 0] = 1

    torch.save(dct, 'stats_tile.pt')