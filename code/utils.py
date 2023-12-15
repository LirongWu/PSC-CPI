import time
import torch
import scipy
import random
import logging
import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_root_logger(file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler("../logs/log_{}_N_{:.3f}.txt".format(time.strftime("%Y-%m-%d %H-%M-%S"), np.random.uniform()))
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cal_affinity_torch(model, loader, batch_size):
    y_pred, labels = np.zeros(len(loader.dataset)), np.zeros(len(loader.dataset))

    batch = 0
    model.eval()
    for prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label in loader:
        prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label = prot_data.to(device), drug_data_ver.to(device), drug_data_adj.to(device), prot_contacts.to(device), prot_inter.to(device), prot_inter_exist.to(device), label.to(device)
        with torch.no_grad():
            _, affn = model.forward_inter_affn(prot_data, drug_data_ver, drug_data_adj, prot_contacts)

        if batch != len(loader.dataset) // batch_size:
            labels[batch*batch_size:(batch+1)*batch_size] = label.squeeze().cpu().numpy()
            y_pred[batch*batch_size:(batch+1)*batch_size] = affn.squeeze().detach().cpu().numpy()
        else:
            labels[batch*batch_size:] = label.squeeze().cpu().numpy()
            y_pred[batch*batch_size:] = affn.squeeze().detach().cpu().numpy()
        batch += 1

    mse = 0
    for n in range(labels.shape[0]):
        mse += (y_pred[n] - labels[n]) ** 2
    mse /= labels.shape[0]
    rmse = np.sqrt(mse)

    pearson, _ = scipy.stats.pearsonr(y_pred.squeeze(), labels.squeeze())
    # tau, _ = scipy.stats.kendalltau(y_pred.squeeze(), labels.squeeze())
    # rho, _ = scipy.stats.spearmanr(y_pred.squeeze(), labels.squeeze())

    return rmse, pearson


def cal_interaction_torch(model, loader, prot_length, comp_length, batch_size):
    outputs, labels, ind = np.zeros((len(loader.dataset), 1000, 56)), np.zeros((len(loader.dataset), 1000, 56)), np.zeros(len(loader.dataset))

    batch = 0
    model.eval()
    for prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label in loader:
        prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label = prot_data.to(device), drug_data_ver.to(device), drug_data_adj.to(device), prot_contacts.to(device), prot_inter.to(device), prot_inter_exist.to(device), label.to(device)
        with torch.no_grad():
            inter, _ = model.forward_inter_affn(prot_data, drug_data_ver, drug_data_adj, prot_contacts)

        if batch != len(loader.dataset) // batch_size:
            labels[batch*batch_size:(batch+1)*batch_size] = prot_inter.cpu().numpy()
            outputs[batch*batch_size:(batch+1)*batch_size] = inter.detach().cpu().numpy()
            ind[batch*batch_size:(batch+1)*batch_size] = prot_inter_exist.cpu().numpy()
        else:
            labels[batch*batch_size:] = prot_inter.cpu().numpy()
            outputs[batch*batch_size:] = inter.detach().cpu().numpy()
            ind[batch*batch_size:] = prot_inter_exist.cpu().numpy()
        batch += 1

    AP = []
    AUC = []
    # AP_margin = []
    # AUC_margin = []

    for i in range(labels.shape[0]):
        if ind[i] != 0:

            length_prot = int(prot_length[i])
            length_comp = int(comp_length[i])

            true_label_cut = np.asarray(labels[i])[:length_prot, :length_comp]
            true_label = np.reshape(true_label_cut, (length_prot*length_comp))
            full_matrix = np.asarray(outputs[i])[:length_prot, :length_comp]
            pred_label = np.reshape(full_matrix, (length_prot*length_comp))

            average_precision_whole = average_precision_score(true_label, pred_label)
            AP.append(average_precision_whole)

            fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label)
            roc_auc_whole = auc(fpr_whole, tpr_whole)
            AUC.append(roc_auc_whole)

            # true_label = np.amax(true_label_cut, axis=1)
            # pred_label = np.amax(full_matrix, axis=1)

            # average_precision_whole = average_precision_score(true_label, pred_label)
            # AP_margin.append(average_precision_whole)

            # fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label)
            # roc_auc_whole = auc(fpr_whole, tpr_whole)
            # AUC_margin.append(roc_auc_whole)

    return np.mean(AP), np.mean(AUC)



