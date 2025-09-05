import torch
import argparse
from distutils.util import strtobool

# all the arguments the user can input
def get_arguments():
    parser = argparse.ArgumentParser('Hyperparameters for training and some additional arguments')
    parser.add_argument('--EPOCHS', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--TASK', type=int, choices=(1, 2), default=1,
                        help='The task to train the model on (1 or 2)')
    parser.add_argument('--CLS_TYPE', type=str, choices=('binary', 'multiclass'), default='binary',
                        help='Classification type of task 2 (binary or multiclass)')
    parser.add_argument('--PLM', type=str, default='google-bert/bert-base-uncased',
                        help='HuggingFace pre-trained language model')
    parser.add_argument('--PLM_MAX_TOKEN', type=int, default=200,
                        help='Max number of tokens when tokenized by the pre-trained tokenizer')
    parser.add_argument('--CLS_DROP_OUT', type=float, default=0.0,
                        help='Drop out probability for the classification layers')
    parser.add_argument('--PLM_LR', type=float, default=1e-5,
                        help='Learning rate for pre-trained language model')
    parser.add_argument('--CLS_LR', type=float, default=1e-4,
                        help='Learning rate for classification layer')
    parser.add_argument('--OPTIMIZER', type=str, default='AdamW',
                        help='Pytorch optimizer (check torch.optim for the full list)')
    parser.add_argument('--TRAIN_BATCH', type=int, default=64,
                        help='Number of instances in a batch during training')
    parser.add_argument('--TEST_BATCH', type=int, default=32,
                        help='Number of instances in a batch during testing')
    parser.add_argument('--PRINT_BATCH', type=int, default=10,
                        help='Print loss after a number of batches')
    parser.add_argument('--RANDOM_SEED', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--DATA_PATH', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--DATA_FOLDER', type=str,
                        help='Name of the dataset folder')
    parser.add_argument('--PLOT_CONFMAT', type=strtobool, default=False,
                        help='Show confusion matrix at each epoch, recommend to set to False when executing in terminal')
    parser.add_argument('--FIG_SIZE', type=int, default=4,
                        help='The height and width of confusion matrix (based on matplotlib)')
    parser.add_argument('--SAVE_MODEL', type=strtobool, default=True,
                        help='Save the model at each epoch while training')
    parser.add_argument('--SAVE_PATH', type=str, default='saved_models',
                        help='Create a folder (if not exist) to store all the saved models')
    parser.add_argument('--EXPORT_PREDICTION', type=strtobool, default=False,
                        help='Export the prediction result (.csv)')
    parser.add_argument('--PREDICTION_PATH', type=str, default='prediction',
                        help='Create a folder (if not exist) to store the prediction file')
    parser.add_argument('--PREDICTION_PER_EPOCH', type=strtobool, default=False,
                        help='Export the prediction result at each epoch, if no then just at the latest epoch')
    parser.add_argument('--TEST_BEST_MODEL', type=strtobool, default=True,
                        help='Evaluate the performance of the best model on the valset (SAVE_MODEL must be True)')
    parser.add_argument('--BEST_PATH', type=str, default='best_prediction',
                        help='Create a folder (if not exist) to store the best prediction file')
    parser.add_argument('--INFO_PATH', type=str, default='info',
                        help='Path to the info file')
    parser.add_argument('--INFO_FILE', type=str, default='info.txt',
                        help='Name of the information file')
    parser.add_argument('--MODELS_LIMIT', type=int, default=3,
                        help='The max number of saved models')
    parser.add_argument('--FONT_SIZE', type=int, default=8,
                        help='The font size for matplotlib confusion matrix')
    args = parser.parse_args()
    # set up GPU for training (if available) agnostically
    args.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args
