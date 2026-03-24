import json
from hy_datautils import collate_fn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class HYCalibrationSet(Dataset):
    def __init__(self, metapath):
        with open(metapath, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.dataset_name = self.metatdata['dataset_name']
        self.config_name = self.metadata['config_name']
        self.split = self.metadata['split']
        self.caliset_name = self.metadata['subset_name']
        self.indices = self.metadata['hf_indices']
        
        self.dataset = load_dataset(self.dataset_name, name=self.config_name, split=self.split)
        self.caliset = self.dataset.select(self.indices)
        
    def __len__(self):
        return len(self.caliset)

    def __getitem__(self, idx):
        items = self.caliset[idx]
        return {
            'audio' : items['audio']['array'],
            'context' : items['text'],
            'language' : items['cv_lang'] if 'cv_lang' in items.keys() else 'en'
        }

def simple_collate(batch):
    return batch[0]


def make_dataloader(mpath):
    dataset = HYCalibrationSet(mpath)
    dataloader = DataLoader(dataset, 
                            batch_size=1,
                            shuffle=False,
                            collate_fn=simple_collate)
    
    return dataloader

def main():
    
    ls_b_mpath = ""
    ls_l_mpath = ""
    cv_s_mpath = ""
    cv_m_mpath = ""
    
    ls_b_loader = make_dataloader(ls_b_mpath)
    ls_l_loader = make_dataloader(ls_l_mpath)
    cv_s_loader = make_dataloader(cv_s_mpath)
    cv_m_loader = make_dataloader(cv_m_mpath)
    
    