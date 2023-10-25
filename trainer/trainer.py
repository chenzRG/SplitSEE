import os
import sys

sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.loss import swavloss



def Trainer(temporal_model, frequency_model, temporal_contr_model, frequency_contr_model, classification_model, model_optimizer, frequency_model_optimizer, temp_cont_optimizer, freq_cont_optimizer, classification_optimizer, train_dl, valid_dl, test_dl, device, logger, config, experiment_log_dir, mode):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    final_loss = swavloss(device, config.TFC.hidden_dim*2).to(device)
    final_loss_optimizer = torch.optim.Adam(final_loss.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=3e-4)

    train_loss_list = []
    valid_loss_list = []
    train_acc_list = []
    valid_acc_list = []
    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        logger.debug(f'\nEpoch : {epoch}')
        train_loss, train_acc = model_train(temporal_model, frequency_model, temporal_contr_model, frequency_contr_model, classification_model, final_loss, model_optimizer, frequency_model_optimizer, temp_cont_optimizer, freq_cont_optimizer, classification_optimizer, final_loss_optimizer ,criterion, train_dl, config, device, logger, mode)
        valid_loss, valid_acc, _, _ = model_evaluate(temporal_model, frequency_model, temporal_contr_model, frequency_contr_model, classification_model, valid_dl, config, device, mode)
        if mode != 'self_supervised':  # use scheduler in all other modes.
            scheduler.step(valid_loss)

        logger.debug(f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': temporal_model.state_dict(), 'frequency_model_state_dict': frequency_model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict(), 'frequency_contr_model_state_dict': frequency_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if mode != "self_supervised":  # no need to run the evaluation for self-supervised mode.
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _ = model_evaluate(temporal_model, frequency_model, temporal_contr_model, frequency_contr_model, classification_model, test_dl, config, device, mode)
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}')

        # Learning curve.
        plt.plot(range(1,config.num_epoch + 1), train_loss_list, linewidth=3, label="train_loss")
        plt.plot(range(1,config.num_epoch + 1), valid_loss_list, linewidth=3, label="valid_loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.xticks(range(5,config.num_epoch + 1,5))
        plt.legend()
        plt.savefig(os.path.join(experiment_log_dir, "Loss.png"))
        plt.close('all')
        plt.plot(range(1,config.num_epoch + 1), train_acc_list, linewidth=3, label="train_acc")
        plt.plot(range(1,config.num_epoch + 1), valid_acc_list, linewidth=3, label="valid_acc")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.xticks(range(5,config.num_epoch + 1,5))
        plt.legend()
        plt.savefig(os.path.join(experiment_log_dir, "Acc.png"))
        plt.close('all')
    else:
        plt.plot(range(1,config.num_epoch + 1), train_loss_list, linewidth=3, label="train_loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.xticks(range(20,config.num_epoch + 1,20))
        plt.legend()
        plt.savefig(os.path.join(experiment_log_dir, "Loss.png"))
        plt.close('all')

    logger.debug("\n################## Training is Done! #########################")


def model_train(temporal_model, frequency_model, temporal_contr_model, frequency_contr_model, classification_model, final_loss, model_optimizer, frequency_model_optimizer, temp_cont_optimizer, freq_cont_optimizer, classification_optimizer, final_loss_optimizer, criterion, train_loader, config, device, logger, mode):
    total_loss = []
    temp_loss1 = []
    temp_loss2 = []
    freq_loss1 = []
    freq_loss2 = []
    cross_loss_list = []
    total_acc = []
    temporal_model.train()
    frequency_model.train()
    temporal_contr_model.train()
    frequency_contr_model.train()
    classification_model.train()
    final_loss.train()
    #torch.autograd.set_detect_anomaly(True)



    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # optimizer
        model_optimizer.zero_grad()
        frequency_model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()
        freq_cont_optimizer.zero_grad()
        final_loss_optimizer.zero_grad()
        classification_optimizer.zero_grad()

        if mode == "self_supervised":
            features1 = temporal_model(aug1)
            features2 = temporal_model(aug2)
            features3= frequency_model(data)
 
            # normalize projection feature vectors
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)
            features3 = F.normalize(features3, dim=1)

            temp_cont_loss1, temp_cont_lstm_feat1, _ = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_lstm_feat2, _ = temporal_contr_model(features2, features1)

            freq_cont_loss1, freq_cont_lstm_feat1, _ = frequency_contr_model(features3)
            freq_cont_loss2, freq_cont_lstm_feat2, _ = frequency_contr_model(torch.flip(features3, dims=[2]))

            # normalize projection feature vectors
            zis = temp_cont_lstm_feat1 
            zjs = temp_cont_lstm_feat2 
            zhs = freq_cont_lstm_feat1 
            zls = freq_cont_lstm_feat2 

            lambda1 = 1
            lambda2 = 1
            cross_loss = final_loss(torch.cat([zis, zjs], dim=1), torch.cat([zhs, zls], dim=1) )
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 +  (freq_cont_loss1 + freq_cont_loss2) * lambda2 + cross_loss
            temp_loss1.append(temp_cont_loss1)
            temp_loss2.append(temp_cont_loss2)
            freq_loss1.append(freq_cont_loss1)
            freq_loss2.append(freq_cont_loss2)
            cross_loss_list.append(cross_loss)
            
        else: 
            temporal_features = temporal_model(data)
            frequency_features = frequency_model(data)
            temporal_features = F.normalize(temporal_features, dim=1)
            frequency_features = F.normalize(frequency_features, dim=1)
            _, _, ct = temporal_contr_model(temporal_features, temporal_features)
            _, _, cf = frequency_contr_model(frequency_features)
            predictions = classification_model(torch.cat([ct, cf], dim=1))

            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        frequency_model_optimizer.step()
        temp_cont_optimizer.step()
        freq_cont_optimizer.step()
        classification_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()
    

    if mode == "self_supervised":
        total_acc = 0
        temp_cont_loss1 = torch.tensor(temp_loss1).mean()
        temp_cont_loss2 = torch.tensor(temp_loss2).mean()
        freq_cont_loss1 = torch.tensor(freq_loss1).mean()
        freq_cont_loss2 = torch.tensor(freq_loss2).mean()
        cross_loss = torch.tensor(cross_loss_list).mean()
        logger.debug(f'T_Loss1        : {temp_cont_loss1:2.4f}\t | \tT_loss2        : {temp_cont_loss2:2.4f}\t | \tF_Loss1        : {freq_cont_loss1:2.4f}\t | \tF_loss2        : {freq_cont_loss2:2.4f}\t | \tcross_loss     : {cross_loss:2.4f}')
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc


def model_evaluate(temporal_model, frequency_model, temporal_contr_model, frequency_contr_model, classification_model, test_dl, configs, device, mode):
    temporal_model.eval()
    frequency_model.eval()
    temporal_contr_model.eval()
    frequency_contr_model.eval()
    classification_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if mode != "self_supervised":
                temporal_features = temporal_model(data)
                frequency_features = frequency_model(data)
                temporal_features = F.normalize(temporal_features, dim=1)
                frequency_features = F.normalize(frequency_features, dim=1)
                _, _, ct = temporal_contr_model(temporal_features, temporal_features)
                _, _, cf = frequency_contr_model(frequency_features)
                predictions = classification_model(torch.cat([ct, cf], dim=1))
                
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
    else:
        total_loss = 0
    if mode == "self_supervised":
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc
    return total_loss, total_acc, outs, trgs
