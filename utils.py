# modules
import os
import torch
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

# function to train the model (performs both forward and backward passes)
def train_step(args, plm, cls, loss_function, optimizer, dataloader):
    plm.train()
    cls.train()
    loss_total = 0
    for batch_index, data in enumerate(dataloader):
        input_ids = data['input_ids'].to(args.DEVICE)
        attention_mask = data['attention_mask'].to(args.DEVICE)
        labels = data['label'].to(args.DEVICE)
        plm_output = plm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        logit = cls(plm_output)
        loss = loss_function(logit, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss
        if (batch_index + 1) % args.PRINT_BATCH == 0:
            print(f'Loss after {batch_index + 1} batches: {loss:.5f}')
    loss_average = loss_total / len(dataloader)
    return loss_total, loss_average

# function to tes the model (performs just forward pass and doesn't store gradient)
def test_step(args, plm, cls, dataloader):
    plm.eval()
    cls.eval()
    labels_true, labels_pred = [], []
    with torch.inference_mode():
        for batch_index, data in enumerate(dataloader):
            input_ids = data['input_ids'].to(args.DEVICE)
            attention_mask = data['attention_mask'].to(args.DEVICE)
            labels = data['label']
            plm_output = plm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            logit = cls(plm_output)
            labels_true.extend(labels.tolist())
            labels_pred.extend(logit.argmax(dim=1).tolist())
    return labels_true, labels_pred

# function to get the metrics after evaluation (return sklearn classification report, accuracy and macro f1)
def get_metrics(labels_true, labels_pred, labels_to_ids):
    cls_report = metrics.classification_report(labels_true, labels_pred, target_names=labels_to_ids,
                                               zero_division=0.0, digits=5)
    cls_report_dict = metrics.classification_report(labels_true, labels_pred, target_names=labels_to_ids,
                                                    zero_division=0.0, output_dict=True)
    accuracy, f1_macro = cls_report_dict['accuracy'], cls_report_dict['macro avg']['f1-score']
    return cls_report, accuracy, f1_macro

# function that plot confusion matrix (recommend not to use when execute in terminal, vice versa in notebook)
def plot_confmat(args, labels_true, labels_pred, ids_to_labels, labels_to_ids):
    disp = metrics.ConfusionMatrixDisplay.from_predictions(labels_true, labels_pred,
                                                           labels=list(ids_to_labels.keys()),
                                                           display_labels=labels_to_ids,
                                                           xticks_rotation='vertical')
    fig = disp.ax_.get_figure()
    fig.set_figwidth(args.FIG_SIZE)
    fig.set_figheight(args.FIG_SIZE)
    plt.show()

# function to save the model after achieving the highest accuracy at each epoch
def save_model(epoch, accuracy, best_epoch, best_accuracy, plm, cls, optimizer, path):
    if accuracy > best_accuracy:
        best_accuracy, best_epoch = accuracy, epoch
        model_name = f"{round(best_accuracy, 4)}.pt"
        save_path = path / model_name
        print(f'** Best model found in this epoch ({epoch}), saving model to: {save_path} **')
        state = {"plm": plm.state_dict(),
                "cls": cls.state_dict(),
                "optimizer": optimizer.state_dict()}
        torch.save(state, save_path)
    return best_epoch, best_accuracy

# function to record all the user's input arguments (hyperparams in some cases) in a .txt file
def log_hyperparams(args, path):
    with open(path / args.INFO_FILE, "w") as f:
        f.write(f'Hyperparameters:\n')
        f.write('------------------------\n')
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')
        f.write('------------------------\n')

# function to record the progress (sklearn classification report) 
# after each epoch in a .txt file (should be in the same file as log_hyperparams)
def log_progress(args, epoch, cls_report, loss_total, loss_average, path):
    with open(path / args.INFO_FILE, "a") as f:
        f.write(f'\nepoch {epoch}:\n')
        f.write(f'Total loss: {loss_total:.5f} | Average loss: {loss_average:.5f}\n')
        f.write(f'Classification report:\n{cls_report}')

# function to export prediction as a .csv file
def export_prediction(df, labels_pred, ids_to_labels, path, name='prediction.csv'):
    prediction_dict = {'prediction': map(ids_to_labels.get, labels_pred)}
    pd.concat([df['id'], pd.DataFrame(prediction_dict)], axis=1).to_csv(path / name, index=False)

# function to test the best model in the saved ones
def test_best_model(args, labels_to_ids, dataloader, model_path, plm, cls):
    models_list = sorted(model for model in os.listdir(model_path) if model.split('.')[-1] == 'pt')
    best_model_path = model_path / models_list[-1]
    state = torch.load(best_model_path, weights_only=False)
    plm.load_state_dict(state['plm'])
    cls.load_state_dict(state['cls'])
    labels_true, labels_pred = test_step(args, plm, cls, dataloader)
    cls_report, _, _ = get_metrics(labels_true, labels_pred, labels_to_ids)
    print('[+] METRICS:')
    print(f'Classification report:\n{cls_report}')
    return labels_true, labels_pred