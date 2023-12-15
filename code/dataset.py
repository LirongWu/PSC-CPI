import torch
import numpy as np


class data_loader(torch.utils.data.Dataset):
    def __init__(self, name_split='train', data_path='../data/'):
        if name_split == 'train':
            self.prot_data, self.drug_data_ver, self.drug_data_adj, _, self.prot_contacts_true, self.prot_inter, self.prot_inter_exist, self.label = load_train_data(data_path)
        elif name_split == 'val':
            self.prot_data, self.drug_data_ver, self.drug_data_adj, _, self.prot_contacts_true, self.prot_inter, self.prot_inter_exist, self.label = load_val_data(data_path)
        elif name_split == 'test':
            self.prot_data, self.drug_data_ver, self.drug_data_adj, _, self.prot_contacts_true, self.prot_inter, self.prot_inter_exist, self.label = load_test_data(data_path)
        elif name_split == 'unseen_prot':
            self.prot_data, self.drug_data_ver, self.drug_data_adj, _, self.prot_contacts_true, self.prot_inter, self.prot_inter_exist, self.label = load_uniqProtein_data(data_path)
        elif name_split == 'unseen_comp':
            self.prot_data, self.drug_data_ver, self.drug_data_adj, _, self.prot_contacts_true, self.prot_inter, self.prot_inter_exist, self.label = load_uniqCompound_data(data_path)
        elif name_split == 'unseen_both':
            self.prot_data, self.drug_data_ver, self.drug_data_adj, _, self.prot_contacts_true, self.prot_inter, self.prot_inter_exist, self.label = load_uniqDouble_data(data_path)

        self.prot_data, self.drug_data_ver, self.drug_data_adj, self.prot_contacts_true, self.prot_inter, self.prot_inter_exist, self.label = torch.tensor(self.prot_data), torch.tensor(self.drug_data_ver).float().float(), torch.tensor(self.drug_data_adj).float(), torch.tensor(self.prot_contacts_true).float(), torch.tensor(self.prot_inter).float(), torch.tensor(self.prot_inter_exist).float().squeeze().float(), torch.tensor(self.label).float()
    
    def __len__(self):
        return self.prot_data.size()[0]
    
    def __getitem__(self, index):
        return self.prot_data[index], self.drug_data_ver[index], self.drug_data_adj[index], self.prot_contacts_true[index], self.prot_inter[index], self.prot_inter_exist[index], self.label[index]


def load_train_data(data_processed_dir):
    protein_train = np.load(data_processed_dir+'protein_train.npy')
    compound_train_ver = np.load(data_processed_dir+'compound_train_ver.npy')
    compound_train_adj = np.load(data_processed_dir+'compound_train_adj.npy')
    prot_train_contacts = np.load(data_processed_dir+'prot_train_contacts.npy')
    prot_train_contacts_true = np.load(data_processed_dir+'prot_train_contacts_true.npy')
    prot_train_inter = np.load(data_processed_dir+'prot_train_inter.npy')
    prot_train_inter_exist = np.load(data_processed_dir+'prot_train_inter_exist.npy')
    IC50_train = np.load(data_processed_dir+'IC50_train.npy')
    return protein_train, compound_train_ver, compound_train_adj, prot_train_contacts, prot_train_contacts_true, prot_train_inter, prot_train_inter_exist, IC50_train


def load_val_data(data_processed_dir):
    protein_dev = np.load(data_processed_dir+'protein_dev.npy')
    compound_dev_ver = np.load(data_processed_dir+'compound_dev_ver.npy')
    compound_dev_adj = np.load(data_processed_dir+'compound_dev_adj.npy')
    prot_dev_contacts = np.load(data_processed_dir+'prot_dev_contacts.npy')
    prot_dev_contacts_true = np.load(data_processed_dir+'prot_dev_contacts_true.npy')
    prot_dev_inter = np.load(data_processed_dir+'prot_dev_inter.npy')
    prot_dev_inter_exist = np.load(data_processed_dir+'prot_dev_inter_exist.npy')
    IC50_dev = np.load(data_processed_dir+'IC50_dev.npy')
    return protein_dev, compound_dev_ver, compound_dev_adj, prot_dev_contacts, prot_dev_contacts_true, prot_dev_inter, prot_dev_inter_exist, IC50_dev


def load_test_data(data_processed_dir):
    protein_test = np.load(data_processed_dir+'protein_test.npy')
    compound_test_ver = np.load(data_processed_dir+'compound_test_ver.npy')
    compound_test_adj = np.load(data_processed_dir+'compound_test_adj.npy')
    prot_test_contacts = np.load(data_processed_dir+'prot_test_contacts.npy')
    prot_test_contacts_true = np.load(data_processed_dir+'prot_test_contacts_true.npy')
    prot_test_inter = np.load(data_processed_dir+'prot_test_inter.npy')
    prot_test_inter_exist = np.load(data_processed_dir+'prot_test_inter_exist.npy')
    IC50_test = np.load(data_processed_dir+'IC50_test.npy')
    return protein_test, compound_test_ver, compound_test_adj, prot_test_contacts, prot_test_contacts_true, prot_test_inter, prot_test_inter_exist, IC50_test


def load_uniqProtein_data(data_processed_dir):
    protein_uniq_protein = np.load(data_processed_dir+'protein_uniq_protein.npy')
    protein_uniq_compound_ver = np.load(data_processed_dir+'protein_uniq_compound_ver.npy')
    protein_uniq_compound_adj = np.load(data_processed_dir+'protein_uniq_compound_adj.npy')
    protein_uniq_prot_contacts = np.load(data_processed_dir+'protein_uniq_prot_contacts.npy')
    protein_uniq_prot_contacts_true = np.load(data_processed_dir+'protein_uniq_prot_contacts_true.npy')
    protein_uniq_prot_inter = np.load(data_processed_dir+'protein_uniq_prot_inter.npy')
    protein_uniq_prot_inter_exist = np.load(data_processed_dir+'protein_uniq_prot_inter_exist.npy')
    protein_uniq_label = np.load(data_processed_dir+'protein_uniq_label.npy')
    return protein_uniq_protein, protein_uniq_compound_ver, protein_uniq_compound_adj, protein_uniq_prot_contacts, protein_uniq_prot_contacts_true, protein_uniq_prot_inter, protein_uniq_prot_inter_exist, protein_uniq_label


def load_uniqCompound_data(data_processed_dir):
    compound_uniq_protein = np.load(data_processed_dir+'compound_uniq_protein.npy')
    compound_uniq_compound_ver = np.load(data_processed_dir+'compound_uniq_compound_ver.npy')
    compound_uniq_compound_adj = np.load(data_processed_dir+'compound_uniq_compound_adj.npy')
    compound_uniq_prot_contacts = np.load(data_processed_dir+'compound_uniq_prot_contacts.npy')
    compound_uniq_prot_contacts_true = np.load(data_processed_dir+'compound_uniq_prot_contacts_true.npy')
    compound_uniq_prot_inter = np.load(data_processed_dir+'compound_uniq_prot_inter.npy')
    compound_uniq_prot_inter_exist = np.load(data_processed_dir+'compound_uniq_prot_inter_exist.npy')
    compound_uniq_label = np.load(data_processed_dir+'compound_uniq_label.npy')
    return compound_uniq_protein, compound_uniq_compound_ver, compound_uniq_compound_adj, compound_uniq_prot_contacts, compound_uniq_prot_contacts_true, compound_uniq_prot_inter, compound_uniq_prot_inter_exist, compound_uniq_label


def load_uniqDouble_data(data_processed_dir):
    double_uniq_protein = np.load(data_processed_dir+'double_uniq_protein.npy')
    double_uniq_compound_ver = np.load(data_processed_dir+'double_uniq_compound_ver.npy')
    double_uniq_compound_adj = np.load(data_processed_dir+'double_uniq_compound_adj.npy')
    double_uniq_prot_contacts = np.load(data_processed_dir+'double_uniq_prot_contacts.npy')
    double_uniq_prot_contacts_true = np.load(data_processed_dir+'double_uniq_prot_contacts_true.npy')
    double_uniq_prot_inter = np.load(data_processed_dir+'double_uniq_prot_inter.npy')
    double_uniq_prot_inter_exist = np.load(data_processed_dir+'double_uniq_prot_inter_exist.npy')
    double_uniq_label = np.load(data_processed_dir+'double_uniq_label.npy')
    return double_uniq_protein, double_uniq_compound_ver, double_uniq_compound_adj, double_uniq_prot_contacts, double_uniq_prot_contacts_true, double_uniq_prot_inter, double_uniq_prot_inter_exist, double_uniq_label




