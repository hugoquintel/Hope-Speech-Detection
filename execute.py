# external modules
import os
import sys
import tqdm
import torch
import random
import pathlib
import matplotlib
import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

# created modules
from args import get_arguments
from model import ClassificationLayers
from data_setup import get_df, get_labels, preprocess_data, HopeDataset
from utils import train_step, test_step, get_metrics, plot_confmat, export_prediction, \
                  save_model, log_hyperparams, log_progress, test_best_model

# execute process
def run():
    # get user's input arguments
    args = get_arguments()

    font = {'size': args.FONT_SIZE}
    matplotlib.rc('font', **font)

    # set random seed (based on pytorch lightning set everything 
    # https://pytorch-lightning.readthedocs.io/en/1.7.7/api/pytorch_lightning.utilities.seed.html#pytorch_lightning.utilities.seed.seed_everything)
    random.seed(args.RANDOM_SEED)
    np.random.seed(args.RANDOM_SEED)
    torch.manual_seed(args.RANDOM_SEED)
    torch.cuda.manual_seed_all(args.RANDOM_SEED)

    # create some utility folders
    if args.SAVE_MODEL:
        model_path = pathlib.Path(args.SAVE_PATH)
        model_path.mkdir(parents=True, exist_ok=True)
    if args.EXPORT_PREDICTION:
        pred_path = pathlib.Path(args.PREDICTION_PATH)
        pred_path.mkdir(parents=True, exist_ok=True)
    info_path = pathlib.Path(args.INFO_PATH)
    info_path.mkdir(parents=True, exist_ok=True)

    # Handling data (import -> preprocess -> dataloader)
    print('Handling data...')
    data_path = pathlib.Path(args.DATA_PATH)
    train_df = get_df(data_path, args.DATA_FOLDER, 'train')
    val_df = get_df(data_path, args.DATA_FOLDER, 'val')

    labels_to_ids, ids_to_labels = get_labels(args, train_df)
    tokenizer = AutoTokenizer.from_pretrained(args.PLM)
    train_df_processed = preprocess_data(args, train_df, tokenizer, labels_to_ids)
    val_df_processed = preprocess_data(args, val_df, tokenizer, labels_to_ids)
    train_data = HopeDataset(train_df_processed)
    val_data = HopeDataset(val_df_processed)

    no_workers = os.cpu_count()
    train_dataloader = DataLoader(train_data, batch_size=args.TRAIN_BATCH,
                                  shuffle=True, num_workers=no_workers)
    val_dataloader = DataLoader(val_data, batch_size=args.TEST_BATCH,
                                shuffle=False, num_workers=no_workers)

    # import the necessary models
    plm = AutoModel.from_pretrained(args.PLM, return_dict=True).to(args.DEVICE)
    cls = ClassificationLayers(plm.config, args.CLS_DROP_OUT, labels_to_ids).to(args.DEVICE)

    # list of Pytorch optimizer, part of the torch.optim module (pytorch version 2.0.1 at the time)
    # https://docs.pytorch.org/docs/stable/optim.html#algorithms
    optimizer_map = {'ASGD': optim.ASGD, 'Adadelta': optim.Adadelta, 'Adagrad': optim.Adagrad, 'Adam': optim.Adam, 
                     'AdamW': optim.AdamW, 'Adamax': optim.Adamax, 'LBFGS': optim.LBFGS, 'NAdam': optim.NAdam, 'RAdam': optim.RAdam,
                     'RMSprop': optim.RMSprop, 'Rprop': optim.Rprop,'SGD': optim.SGD, 'SparseAdam': optim.SparseAdam}
    
    # set up training params, optimizer and loss function
    train_params = [{'params': plm.parameters(), 'lr': args.PLM_LR},
                    {'params': cls.parameters(), 'lr': args.CLS_LR}]
    optimizer = optimizer_map[args.OPTIMIZER](train_params)
    loss_function = nn.CrossEntropyLoss()

    # show some information about the data
    print(f'Number of samples in train set: {len(train_data)}')
    print(f'Number of samples in val set: {len(val_data)}')
    print(f'Number of train batches: {len(train_dataloader)}')
    print(f'Number of val batches: {len(val_dataloader)}')
    print(f'Labels to ids: {labels_to_ids}')
    print(f'Ids to labels: {ids_to_labels}\n')

    # set up mutiple GPUS to use (if available) agnostically
    if args.DEVICE == 'cuda':
        if torch.cuda.device_count() > 1:
            print('--Using multiple GPUs to train--\n')
            plm = torch.nn.DataParallel(plm)
            cls = torch.nn.DataParallel(cls)
        else: print('--Using single GPU to train--\n')
    else: print('--No GPU detected, using CPU to train--\n')

    # show user's input arguments
    print(f'Arguments:')
    print('------------------------')
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('------------------------')

    # record user's input arguments
    log_hyperparams(args, info_path)
    best_accuracy, best_epoch = 0, 0
    
    # start executing
    for epoch in tqdm.trange(args.EPOCHS, file=sys.stdout):
        print(f'\n\nEpoch {epoch}:')
        # train
        print('-----------')
        loss_total, loss_average = train_step(args, plm, cls, loss_function,
                                              optimizer, train_dataloader)
        print(f'Total loss: {loss_total:.5f} | Average loss: {loss_average:.5f}')
        print('-----------')
        # test
        labels_val_true, labels_val_pred = test_step(args, plm, cls, val_dataloader)
        cls_report, accuracy, f1_macro = get_metrics(labels_val_true, labels_val_pred, labels_to_ids)
        print('[+] METRICS:')
        print(f'Classification report:\n{cls_report}')
        # record the evaluation metrics
        log_progress(args, epoch, cls_report, loss_total, loss_average, info_path)
        # plot confusion matrix
        if args.PLOT_CONFMAT:
            plot_confmat(args, labels_val_true, labels_val_pred, ids_to_labels, labels_to_ids)
        # plot confusion matrix
        if args.SAVE_MODEL:
            best_epoch, best_accuracy = save_model(epoch, accuracy, best_epoch, best_accuracy, plm, cls, optimizer, model_path)
            saved_models = sorted(model for model in os.listdir(model_path) if model.split('.')[-1] == 'pt')
            if len(saved_models) > args.MODELS_LIMIT:
                os.remove(model_path / saved_models[0])
        # export prediction
        if args.EXPORT_PREDICTION:
            if args.PREDICTION_PER_EPOCH:
                export_prediction(val_df, labels_val_pred, ids_to_labels, pred_path, f'epoch_{epoch}.csv')
            else:
                export_prediction(val_df, labels_val_pred, ids_to_labels, pred_path)
    # test best model in the saved ones
    if args.TEST_BEST_MODEL and args.SAVE_MODEL:
        print('\n\n\n******TESTING THE BEST MODEL******')
        best_pred_path = pathlib.Path('best_prediction')
        best_pred_path.mkdir(parents=True, exist_ok=True)
        labels_val_true, labels_val_pred = test_best_model(args, labels_to_ids, val_dataloader, model_path, plm, cls)
        if args.PLOT_CONFMAT:
            plot_confmat(args, labels_val_true, labels_val_pred, ids_to_labels, labels_to_ids)
        if args.EXPORT_PREDICTION:
            export_prediction(val_df, labels_val_pred, ids_to_labels, best_pred_path, 'best_prediction.csv')
        print('**************FINISH**************')