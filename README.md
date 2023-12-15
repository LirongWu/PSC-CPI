#  Multi-Scale Protein Sequence-Structure Contrasting for Compound-Protein Interaction Prediction (PSC-CPI)

This is a PyTorch implementation of Protein Sequence-sructure Contrasting for CPI Prediction (PSC-CPI), and the code includes the following modules:

* Dataset loader (train/val/test)

* Four evaluation settings: Seen-Both, Unseen-Compound, Unseen-Protein, and Unseen-Both

* Four evaluation metrics: CPI pattern prediction (AUPRC and AUROC) and CPI  Strength Prediction (RMSE and PPCs)

* Pre-training, fine-tuning, and inference paradigm

  

## Main Requirements

* numpy==1.21.6
* scipy==1.7.3
* torch==1.6.0
* sklearn == 1.0.2



## Dataset

The datasets used in this paper are available in:

https://drive.google.com/file/d/1_iZ8B1JZkCKmKlQNewOCr3kbnWfAIc-r/view?usp=sharing



## Description

* train.py  
  * Pre-training, fine-tuning, and inference
* models.py  
  * ProteinEmbed_Model() - Learning protein sequence and structure representations
    * prot_data_aug() -- Data augmentation on proteins
    * loss_inter() -- loss for CPI pattern prediction
    * loss_affn() -- loss for CPI strength prediction
    * loss_contras() -- loss for (pre-training) multi-scale contrastive learning

* dataloader.py  

  * data_loader() -- Load train, val, and test data (with four evaluation data spilts)
* utils.py  
  * set_seed() -- Set radom seeds for reproducible results
  * cal_affinity_torch() -- Use Pytorch to calculate CPI affinity (RMSE and PPCs)
  * cal_interaction_torch() -- Use Pytorch to calculate CPI pattern (AUPRC and AUROC)




## Running the code

1. Install the required dependency packages

3. To pre-train and fine-tune the model, please run with proper hyperparameters:

  ```
python train.py --task_mode 0 --modality seq_str_linear --pre-train 1 --seq_encoder HRNN --str_encoder GAT
  ```

where (1) *task_mode* is one of the two CPI tasks: 0 (Strength Prediction) and 1 (Pattern Prediction); (2) *modality* is one of the three inference settings: 'seq_str_linear' (both two modalities are provided), 'sequence' (only sequence is provided), and 'structure' (only structure is provided); (3) *pre-train* denotes whether the pre-training is conducted: 0 (w/o pre-training) and 1 (w/ pre-training); (4) *seq_encoder* is one of the four protein sequence encoders: HRNN, LSTM, bi-LSTM, and Transformer; and (5) *str_encoder* is one of the three protein structure encoders: GCN, GAT, and SAGE.

## Citation

If you find this project useful for your research, please use the following BibTeX entry.

```
@inproceedings{wu2024protein,
  title={PSC-CPI: Multi-Scale Protein Sequence-Structure Contrasting for Efficient and Generalizable Compound-Protein Interaction Prediction},
  author={Wu, Lirong and Huang, Yufei and Tan, Cheng and Gao, Zhangyang and Lin, Haitao and Hu, Bozhen and Liu, Zicheng and Li, Stan Z},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```