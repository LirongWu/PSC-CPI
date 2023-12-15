import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ProteinEmbed_Model(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.aminoAcid_embedding = nn.Embedding(29, param['hidden_dim'])
        if param['seq_encoder'] == 'HRNN':
            self.gru0 = nn.GRU(param['hidden_dim'], param['hidden_dim'], batch_first=True)
            self.gru1 = nn.GRU(param['hidden_dim'], param['hidden_dim'], batch_first=True)
        elif param['seq_encoder'] == 'LSTM':
            self.lstm0 = nn.LSTM(param['hidden_dim'], param['hidden_dim'], batch_first=True)
            self.lstm1 = nn.LSTM(param['hidden_dim'], param['hidden_dim'], batch_first=True)
        elif param['seq_encoder'] == 'Bi_LSTM':
            self.bi_lstm0 = nn.LSTM(param['hidden_dim'], int(param['hidden_dim']/2), batch_first=True, bidirectional=True)
            self.bi_lstm1 = nn.LSTM(param['hidden_dim'], int(param['hidden_dim']/2), batch_first=True, bidirectional=True)
        elif param['seq_encoder'] == 'Transformer':
            encoder_layer = nn.TransformerEncoderLayer(param['hidden_dim'], 8, param['hidden_dim'], 0.5)
            self.transformer = nn.TransformerEncoder(encoder_layer, 6)

        if param['str_encoder'] == 'GAT':
            self.gat = net_prot_gat(param)
        elif param['str_encoder'] == 'GCN':
            self.gat = net_prot_gcn(param)
        elif param['str_encoder'] == 'SAGE':
            self.gat = net_prot_sage(param)

        
        self.crossInteraction = crossInteraction(param)
        self.gcn_comp = net_comp_gcn(param)

        self.joint_attn_prot, self.joint_attn_comp = nn.Linear(param['hidden_dim'], param['hidden_dim']), nn.Linear(param['hidden_dim'], param['hidden_dim'])
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.regressor0 = nn.Sequential(nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1),
                                       nn.LeakyReLU(0.1),
                                       nn.MaxPool1d(kernel_size=4, stride=4))
        self.regressor1 = nn.Sequential(nn.Linear(int(64*param['hidden_dim']/8), 600),
                                        nn.LeakyReLU(0.1),
                                        nn.Dropout(0.5),
                                        nn.Linear(600, 300),
                                        nn.LeakyReLU(0.1),
                                        nn.Dropout(0.5),
                                        nn.Linear(300, 1))

        self.projection_head_seq = nn.Sequential(nn.Linear(param['hidden_dim'], 256), nn.ReLU(), nn.Linear(256, 256))
        self.projection_head_graph = nn.Sequential(nn.Linear(param['hidden_dim'], 256), nn.ReLU(), nn.Linear(256, 256))

        self.task_mode = param['task_mode']
        self.seq_encoder = param['seq_encoder']
        
        self.hidden_dim = param['hidden_dim']
        self.temp = param['temp']
        self.aug_left = param['aug_left']
        self.aug_right = param['aug_right']
        self.intra_loss = param['intra_loss']

    def forward(self, prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label, fused_matrix, pre_train_mode='fine-tune'):
        # protein embedding
        aminoAcid_embedding = self.aminoAcid_embedding(prot_data)

        b, i, j, d = aminoAcid_embedding.size()
        prot_seq_embedding = aminoAcid_embedding.reshape(b*i, j, d)
        if self.seq_encoder == 'HRNN':
            prot_seq_embedding, _ = self.gru0(prot_seq_embedding)
            prot_seq_embedding = prot_seq_embedding.reshape(b*j, i, d)
            prot_seq_embedding, _ = self.gru1(prot_seq_embedding)
        elif self.seq_encoder == 'LSTM':
            prot_seq_embedding, _ = self.lstm0(prot_seq_embedding)
            prot_seq_embedding = prot_seq_embedding.reshape(b*j, i, d)
            prot_seq_embedding, _ = self.lstm1(prot_seq_embedding)
        elif self.seq_encoder == 'Bi_LSTM':
            prot_seq_embedding, _ = self.bi_lstm0(prot_seq_embedding)
            prot_seq_embedding = prot_seq_embedding.reshape(b*j, i, d)
            prot_seq_embedding, _ = self.bi_lstm1(prot_seq_embedding)
        elif self.seq_encoder == 'Transformer':
            prot_seq_embedding = self.transformer(prot_seq_embedding.transpose(0, 1)).transpose(0, 1)
        prot_seq_embedding = prot_seq_embedding.reshape(b, i*j, d)

        prot_graph_embedding = aminoAcid_embedding.reshape(b, i*j, d)
        prot_graph_embedding = self.gat(prot_graph_embedding, prot_contacts)

        prot_embedding = self.crossInteraction(prot_seq_embedding, prot_graph_embedding)

        # compound embedding
        comp_embedding = self.gcn_comp(drug_data_ver, drug_data_adj)

        # compound-protein interaction
        inter_comp_prot = self.sigmoid(torch.einsum('bij,bkj->bik', self.joint_attn_prot(self.relu(prot_embedding)), self.joint_attn_comp(self.relu(comp_embedding))))
        inter_comp_prot_sum = torch.einsum('bij->b', inter_comp_prot)
        inter_comp_prot = torch.einsum('bij,b->bij', inter_comp_prot, 1/inter_comp_prot_sum)

        # compound-protein joint embedding
        cp_embedding = self.tanh(torch.einsum('bij,bkj->bikj', prot_embedding, comp_embedding))
        cp_embedding = torch.einsum('bijk,bij->bk', cp_embedding, inter_comp_prot)

        # compound-protein affinity
        affn_comp_prot = cp_embedding[:, None, :]
        affn_comp_prot = self.regressor0(affn_comp_prot)
        affn_comp_prot = affn_comp_prot.view(b, int(64*self.hidden_dim/8))
        affn_comp_prot = self.regressor1(affn_comp_prot)

        if pre_train_mode == 'pre-train':
            loss = self.loss_contras(prot_data, prot_contacts, prot_seq_embedding, prot_graph_embedding)
        else:
            if self.task_mode == 0:
                loss = self.loss_affn(affn_comp_prot, label) 
            elif self.task_mode == 1:
                loss = self.loss_reg(inter_comp_prot, fused_matrix.to(inter_comp_prot.device), prot_contacts) + self.loss_inter(inter_comp_prot, prot_inter, prot_inter_exist)

        return loss

    def forward_inter_affn(self, prot_data, drug_data_ver, drug_data_adj, prot_contacts):
        # protein embedding
        aminoAcid_embedding = self.aminoAcid_embedding(prot_data)

        b, i, j, d = aminoAcid_embedding.size()
        prot_seq_embedding = aminoAcid_embedding.reshape(b*i, j, d)
        if self.seq_encoder == 'HRNN':
            prot_seq_embedding, _ = self.gru0(prot_seq_embedding)
            prot_seq_embedding = prot_seq_embedding.reshape(b*j, i, d)
            prot_seq_embedding, _ = self.gru1(prot_seq_embedding)
        elif self.seq_encoder == 'LSTM':
            prot_seq_embedding, _ = self.lstm0(prot_seq_embedding)
            prot_seq_embedding = prot_seq_embedding.reshape(b*j, i, d)
            prot_seq_embedding, _ = self.lstm1(prot_seq_embedding)
        elif self.seq_encoder == 'Bi_LSTM':
            prot_seq_embedding, _ = self.bi_lstm0(prot_seq_embedding)
            prot_seq_embedding = prot_seq_embedding.reshape(b*j, i, d)
            prot_seq_embedding, _ = self.bi_lstm1(prot_seq_embedding)
        elif self.seq_encoder == 'Transformer':
            prot_seq_embedding = self.transformer(prot_seq_embedding.transpose(0, 1)).transpose(0, 1)
        prot_seq_embedding = prot_seq_embedding.reshape(b, i*j, d)

        prot_graph_embedding = aminoAcid_embedding.reshape(b, i*j, d)
        prot_graph_embedding = self.gat(prot_graph_embedding, prot_contacts)

        prot_embedding = self.crossInteraction(prot_seq_embedding, prot_graph_embedding)

        # compound embedding
        comp_embedding = self.gcn_comp(drug_data_ver, drug_data_adj)

        # compound-protein interaction
        inter_comp_prot = self.sigmoid(torch.einsum('bij,bkj->bik', self.joint_attn_prot(self.relu(prot_embedding)), self.joint_attn_comp(self.relu(comp_embedding))))
        inter_comp_prot_sum = torch.einsum('bij->b', inter_comp_prot)
        inter_comp_prot = torch.einsum('bij,b->bij', inter_comp_prot, 1/inter_comp_prot_sum)

        # compound-protein joint embedding
        cp_embedding = self.tanh(torch.einsum('bij,bkj->bikj', prot_embedding, comp_embedding))
        cp_embedding = torch.einsum('bijk,bij->bk', cp_embedding, inter_comp_prot)

        # compound-protein affinity
        affn_comp_prot = cp_embedding[:, None, :]
        affn_comp_prot = self.regressor0(affn_comp_prot)
        affn_comp_prot = affn_comp_prot.view(b, int(64*self.hidden_dim/8))
        affn_comp_prot = self.regressor1(affn_comp_prot)

        return inter_comp_prot, affn_comp_prot

    def forward_prot_data(self, prot_data, prot_contacts):

        aminoAcid_embedding = self.aminoAcid_embedding(prot_data)

        b, i, j, d = aminoAcid_embedding.size()
        prot_seq_embedding = aminoAcid_embedding.reshape(b*i, j, d)
        if self.seq_encoder == 'HRNN':
            prot_seq_embedding, _ = self.gru0(prot_seq_embedding)
            prot_seq_embedding = prot_seq_embedding.reshape(b*j, i, d)
            prot_seq_embedding, _ = self.gru1(prot_seq_embedding)
        elif self.seq_encoder == 'LSTM':
            prot_seq_embedding, _ = self.lstm0(prot_seq_embedding)
            prot_seq_embedding = prot_seq_embedding.reshape(b*j, i, d)
            prot_seq_embedding, _ = self.lstm1(prot_seq_embedding)
        elif self.seq_encoder == 'Bi_LSTM':
            prot_seq_embedding, _ = self.bi_lstm0(prot_seq_embedding)
            prot_seq_embedding = prot_seq_embedding.reshape(b*j, i, d)
            prot_seq_embedding, _ = self.bi_lstm1(prot_seq_embedding)
        elif self.seq_encoder == 'Transformer':
            prot_seq_embedding = self.transformer(prot_seq_embedding.transpose(0, 1)).transpose(0, 1)
        prot_seq_embedding = prot_seq_embedding.reshape(b, i*j, d)

        prot_graph_embedding = aminoAcid_embedding.reshape(b, i*j, d)
        prot_graph_embedding = self.gat(prot_graph_embedding, prot_contacts)

        return prot_seq_embedding, prot_graph_embedding

    def prot_data_aug(self, prot_data, prot_contacts):

        b, i, j = prot_data.size()
        prot_data = prot_data.reshape(b, i*j)
        mat_mask = torch.zeros((b, i*j), dtype=torch.float32).to(device)
        p_sub = np.random.uniform(self.aug_left, self.aug_right, (b, ))
        pro_mat_mask = torch.zeros((b, i*j), dtype=torch.float32).to(device)
        pro_mat_mask[prot_data != 0] = 1

        idx_subcentre = np.array([np.random.choice(int(1000*(1-p_sub[n])), 1)[0] for n in range(b)])
        idx_sub = np.array([[n, m] for n in range(b)
                                   for m in range(int(idx_subcentre[n]), int(idx_subcentre[n]+1000*p_sub[n]))])

        _prot_data = torch.zeros(prot_data.size(), dtype=torch.int64).to(device)
        _prot_data[idx_sub[:,0], idx_sub[:,1]] = prot_data[idx_sub[:,0], idx_sub[:,1]]
        _prot_data = _prot_data.reshape(b, i, j)

        mat_mask[idx_sub[:,0], idx_sub[:,1]] = 1

        for n in range(b):
            idx_nonsub = torch.nonzero(1-mat_mask[n,:])[:,0]
            prot_contacts[n][idx_nonsub] = 0
            prot_contacts[n][:, idx_nonsub] = 0
            prot_contacts[n][list(range(1000)), list(range(1000))] = 1

        return _prot_data, prot_contacts, mat_mask, pro_mat_mask

    def loss_reg(self, inter, fused_matrix, prot_contacts):
        reg_l1 = torch.abs(inter).sum(dim=(1,2)).mean()

        reg_fused = torch.abs(torch.einsum('bij,ti->bjt', inter, fused_matrix)).sum(dim=(1,2)).mean()

        group = torch.einsum('bij,bki->bjk', inter**2, prot_contacts).sum(dim=1)
        group[group==0] = group[group==0] + 1e10
        reg_group = ( torch.sqrt(group) * torch.sqrt(prot_contacts.sum(dim=2)) ).sum(dim=1).mean()

        reg_loss = (reg_l1 + reg_fused + reg_group) * 0.001

        return reg_loss

    def loss_inter(self, inter, prot_inter, prot_inter_exist):
        label = torch.einsum('b,bij->bij', prot_inter_exist, prot_inter)
        loss = torch.sqrt(((inter - label) ** 2).sum(dim=(1,2))).mean()
        return loss * 1000

    def loss_affn(self, affn, label):
        loss = ((affn - label) ** 2).mean()
        return loss

    def loss_contras(self, prot_data, prot_contacts, prot_full_seq_embedding, prot_full_graph_embedding):

        prot_data_clone = prot_data.detach().clone()
        prot_contacts_clone = prot_contacts.detach().clone()
        prot_data_aug, prot_contacts_aug, mat_mask, pro_mat_mask = self.prot_data_aug(prot_data_clone, prot_contacts_clone)
        prot_seq_embedding, prot_graph_embedding = self.forward_prot_data(prot_data_aug, prot_contacts_aug)

        output1 = self.projection_head_seq(torch.einsum('bij,bi,b->bj', prot_seq_embedding, (mat_mask * pro_mat_mask), 1/((mat_mask * pro_mat_mask).sum(dim=1)+1e-12)))
        output2 = self.projection_head_graph(torch.einsum('bij,bi,b->bj', prot_graph_embedding, (mat_mask * pro_mat_mask), 1/((mat_mask * pro_mat_mask).sum(dim=1)+1e-12)))
        output3 = self.projection_head_seq(torch.einsum('bij,bi,b->bj', prot_full_seq_embedding, pro_mat_mask, 1/pro_mat_mask.sum(dim=1)))
        output4 = self.projection_head_graph(torch.einsum('bij,bi,b->bj', prot_full_graph_embedding, pro_mat_mask, 1/pro_mat_mask.sum(dim=1)))

        loss = (self.com_cl_loss(output1, output2) + self.com_cl_loss(output2, output1)) / 2 \
             + ((self.com_cl_loss(output1, output3) + self.com_cl_loss(output3, output1)) / 2 \
             + (self.com_cl_loss(output2, output4) + self.com_cl_loss(output4, output2)) / 2) * self.intra_loss

        return loss

    def com_cl_loss(self, output1, output2):
        norm1, norm2 = output1.norm(dim=1), output2.norm(dim=1)

        mat_norm = torch.einsum('i,j->ij', norm1, norm2)
        mat_sim = torch.exp(torch.einsum('ik,jk,ij->ij', output1, output2, 1/mat_norm) / self.temp)

        b, _ = output1.size()
        loss = - torch.log(mat_sim[list(range(b)), list(range(b))] / (mat_sim.sum(dim=1) - mat_sim[list(range(b)), list(range(b))])).mean()

        return loss


class net_prot_gat(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.linear0 = nn.ModuleList([nn.Linear(param['hidden_dim'], param['hidden_dim']) for _ in range(7)])
        self.linear1 = nn.ModuleList([nn.Linear(param['hidden_dim'], param['hidden_dim']) for _ in range(7)])
        self.w_attn = nn.ModuleList([nn.Linear(param['hidden_dim'], param['hidden_dim']) for _ in range(7)])
        self.linear_final = nn.Linear(param['hidden_dim'], param['hidden_dim'])

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, adj):
        adj[:, list(range(1000)), list(range(1000))] = 1
        
        for l in range(7):
            x0 = x

            adj_attn = self.sigmoid(torch.einsum('bij,bkj->bik', self.w_attn[l](x), x))
            adj_attn = adj_attn + 1e-5 * torch.eye(1000).to(x.device)
            adj_attn = torch.einsum('bij,bij->bij', adj_attn, adj)
            adj_attn_sum = torch.einsum('bij->bi', adj_attn)
            adj_attn = torch.einsum('bij,bi->bij', adj_attn, 1/adj_attn_sum)

            x = torch.einsum('bij,bjd->bid', adj_attn, x)
            x = self.relu(self.linear0[l](x))
            x = self.relu(self.linear1[l](x))

            x += x0

        x = self.linear_final(x)
        return x


class net_prot_gcn(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.linear = nn.ModuleList([nn.Linear(param['hidden_dim'], param['hidden_dim']) for _ in range(7)])
        self.linear_final = nn.Linear(param['hidden_dim'], param['hidden_dim'])

        self.relu = nn.ReLU()

    def forward(self, x, adj):
        adj[:, list(range(1000)), list(range(1000))] = 1
        degrees = adj.sum(dim=2)
        degree_mat_inv_sqrt = torch.diag_embed(torch.pow(degrees, -0.5))
        adj_norm = torch.einsum('bij,bjk->bik', torch.einsum('bij,bjk->bik', degree_mat_inv_sqrt, adj), degree_mat_inv_sqrt)

        for l in range(7):
            x0 = x

            x = torch.einsum('bij,bjd->bid', adj_norm, x)
            x = self.relu(self.linear[l](x))

            x += x0

        x = self.linear_final(x)
        return x


class net_prot_sage(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.linear = nn.ModuleList([nn.Linear(param['hidden_dim'] * 2, param['hidden_dim']) for _ in range(7)])
        self.linear_final = nn.Linear(param['hidden_dim'], param['hidden_dim'])

        self.relu = nn.ReLU()

    def forward(self, x, adj):
        adj[:, list(range(1000)), list(range(1000))] = 1
        degrees = adj.sum(dim=2)
        degree_mat_inv_sqrt = torch.diag_embed(torch.pow(degrees, -0.5))
        adj_norm = torch.einsum('bij,bjk->bik', torch.einsum('bij,bjk->bik', degree_mat_inv_sqrt, adj), degree_mat_inv_sqrt)

        for l in range(7):
            x0 = x

            neigh_x = torch.einsum('bij,bjd->bid', adj_norm, x)
            x = torch.cat([neigh_x, x], dim=-1)
            x = self.relu(self.linear[l](x))

            x += x0

        x = self.linear_final(x)
        return x


class crossInteraction(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.linear = nn.Linear(param['hidden_dim'] * 2, param['hidden_dim'])
        self.modality = param['modality']

        if self.modality == 'seq_str_cross':
            self.crossInteraction0 = nn.Sequential(nn.Linear(param['hidden_dim'], param['hidden_dim']), nn.ReLU(), nn.Linear(param['hidden_dim'], param['hidden_dim']))
            self.crossInteraction1 = nn.Sequential(nn.Linear(param['hidden_dim'], param['hidden_dim']), nn.ReLU(), nn.Linear(param['hidden_dim'], param['hidden_dim']))
            self.tanh = nn.Tanh()

    def forward(self, x_seq, x_graph):
        if self.modality == 'seq_str_linear':
            x = torch.cat((x_seq, x_graph), dim=2)
            x = self.linear(x)
        elif self.modality == 'sequence':
            x = x_seq
        elif self.modality == 'structure':
            x = x_graph
        elif self.modality == 'seq_str_cross':
            CI0 = self.tanh(torch.einsum('bij,bij->bi', self.crossInteraction0(x_graph), x_seq)) + 1
            CI1 = self.tanh(torch.einsum('bij,bij->bi', self.crossInteraction1(x_seq), x_graph)) + 1
            x_seq = torch.einsum('bij,bi->bij', x_seq, CI0)
            x_graph = torch.einsum('bij,bi->bij', x_graph, CI1)

            x = torch.cat((x_seq, x_graph), dim=2)
            x = self.linear(x)
        
        return x


class net_comp_gcn(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.linear = nn.ModuleList([nn.Linear(43, param['hidden_dim']), nn.Linear(param['hidden_dim'], param['hidden_dim']), nn.Linear(param['hidden_dim'], param['hidden_dim'])])
        self.linear_final = nn.Linear(param['hidden_dim'], param['hidden_dim'])
        self.relu = nn.ReLU()
        
    def forward(self, x, adj):
        for l in range(3):
            x = self.linear[l](x)
            x = torch.einsum('bij,bjd->bid', adj, x)
            x = self.relu(x)
        x = self.linear_final(x)

        return x