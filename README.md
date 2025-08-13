## Hope Speech detection

### Introduction
This is the implementation of the paper *Choosing the Right Language Model for the Right Task* as part of the [Iberlef HOPE 2024 shared tasks](https://codalab.lisn.upsaclay.fr/competitions/17714). The full paper can be accessed through this [link](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj1lbfX54KPAxXgaPUHHV9iPAQQFnoECBkQAQ&url=https%3A%2F%2Fceur-ws.org%2FVol-3756%2FHOPE2024_paper3.pdf&usg=AOvVaw10GVu5qjGVJpoZHNH7Vgv7&opi=89978449).

There are 2 tasks, whose goal is to detect hope speech given an input text:
- **Task 1 - Hope for Equality, Diversity, and Inclusion**: This task aims to detect hope speech, focusing on promoting the inclusion of vulnerable groups and ultimately achieving Equality, Diversity, and Inclusion. The dataset for this task is only Spanish.
- **Task 2 - Hope as Expectations**: This task focuses on expectations, desirable and undesirable facts. There are two subtasks in this task: binary and multiclass hope speech detection. The dataset provided for this task is in both English and Spanish.

### Requirements
The code was written and tested at the time using these packages/libraries:
- python==3.10.4
- tqdm==4.62.3
- numpy==1.25.2
- pandas==1.5.2
- matplotlib==3.7.1
- scikit-learn==1.2.0
- transformers==4.47.1

### Data
The data files are in .csv format. Below is the format of the dataset for each task:
- Task 1

| id | text | category |
| ----------- | ----------- | ----------- |
| ... | ... | nhs |
| ... | ... | hs |

- Task 2

| text | binary | multiclass | id |
| ----------- | ----------- | ----------- | ----------- |
| ... | Not Hope | Not Hope | ... |
| ... | Hope | Generalized Hope | ... |
| ... | Hope | Realistic Hope | ... |
| ... | Hope | Unrealistic Hope | ... |

The id and text columns are input IDs and texts, respectively. For task 1, the target column is "category", containing 2 labels: nhs (not hope) or hs (hope), and for task 2, the target columns are "binary", which is either "Not Hope" or "Hope", and "multiclass", including Not Hope, Generalized Hope, Realistic Hope, or Unrealistic Hope. Sample data can be viewed in the [data folder](https://github.com/hugoquintel/git_test/tree/master/data).

The structure of the data folder should be the following:

```
Hope speech detection
│   ...
└───data
    └───Task 1
    |   train.csv
    |   val.csv
    └───Task 2
    |   train.csv
    |   val.csv
```

### Custom arguments
All of the user input arguments:
```
--EPOCHS', type=int, default=10, help='Number of training epochs'
--TASK', type=int, choices=(1, 2), default=1, help='The task to train the model on (1 or 2)'
--CLS_TYPE', type=str, choices=('binary', 'multiclass'), default='binary', help='Classification type of task 2 (binary or multiclass)'
--PLM', type=str, default='google-bert/bert-base-uncased', help='HuggingFace pre-trained language model'
--PLM_MAX_TOKEN', type=int, default=200, help='Max number of tokens when tokenized by the pre-trained tokenizer'
--CLS_DROP_OUT', type=float, default=0.0, help='Drop out probability for the classification layers'
--PLM_LR', type=float, default=1e-5, help='Learning rate for pre-trained language model'
--CLS_LR', type=float, default=1e-4, help='Learning rate for classification layer'
--OPTIMIZER', type=str, default='AdamW', help='Pytorch optimizer (check torch.optim for the full list)'
--TRAIN_BATCH', type=int, default=64, help='Number of instances in a batch during training'
--TEST_BATCH', type=int, default=32, help='Number of instances in a batch during testing'
--PRINT_BATCH', type=int, default=10, help='Print loss after a number of batches'
--RANDOM_SEED', type=int, default=2024, help='Random seed'
--DATA_PATH', type=str, default='data', help='Path to data directory'
--DATA_FOLDER', type=str, help='Name of the dataset folder'
--PLOT_CONFMAT', type=bool, default=False, help='Show confusion matrix at each epoch, recommend to set to False when executing in terminal'
--FIG_SIZE', type=int, default=4, help='The height and width of confusion matrix (based on matplotlib)'
--SAVE_MODEL', type=bool, default=True, help='Save the model at each epoch while training'
--SAVE_PATH', type=str, default='saved_models', help='Create a folder (if not exist) to store all the saved models'
--EXPORT_PREDICTION', type=bool, default=False, help='Export the prediction result (.csv)'
--PREDICTION_PATH', type=str, default='prediction', help='Create a folder (if not exist) to store the prediction file'
--PREDICTION_PER_EPOCH', type=bool, default=False, help='Export the prediction result at each epoch, if no then just at the latest epoch'
--TEST_BEST_MODEL', type=bool, default=False, help='Evaluate the performance of the best model on the valset (SAVE_MODEL must be True)'
--BEST_PATH', type=str, default='best_prediction', help='Create a folder (if not exist) to store the best prediction file'
--INFO_PATH', type=str, default='info', help='Path to the info file'
--INFO_FILE', type=str, default='info.txt', help='Name of the information file'
--MODELS_LIMIT', type=int, default=3, help='The max number of saved models'
--FONT_SIZE', type=int, default=8, help='The font size for matplotlib confusion matrix'
```

### Execution
For a quick execution on the terminal:
```
python path/to/main.py --DATA_FOLDER task_1_or_2
```

[*] Some notes:
- The default task is 1
- The default language model is bert-base-uncased and might not work well on Spanish datasets, so a change of pre-trained model is advised to fit the language
- --PLOT_CONFMAT should be turned off when executing using the local terminal because matplotlib will show the confusion matrix in separate pop-up windows after every epoch

An execution with all the input arguments:
```
python "path/to/main.py" --EPOCHS 10 --TASK 1 --CLS_TYPE binary --PLM "dccuchile/bert-base-spanish-wwm-cased" --PLM_MAX_TOKEN 200 --CLS_DROP_OUT 0.0 --PLM_LR 1e-5 --CLS_LR 1e-4 --OPTIMIZER AdamW --TRAIN_BATCH 64 --TEST_BATCH 32 --PRINT_BATCH 10 --RANDOM_SEED 2024 --DATA_PATH "data" --DATA_FOLDER "task_1_or_2" --PLOT_CONFMAT False --FIG_SIZE 3 --SAVE_MODEL True --SAVE_PATH saved_models --EXPORT_PREDICTION True --PREDICTION_PATH prediction --PREDICTION_PER_EPOCH False --TEST_BEST_MODEL True --BEST_PATH best_prediction --INFO_PATH info --INFO_FILE info.txt --MODELS_LIMIT 3 --FONT_SIZE 8
```

When using notebook, it's better to use the ```%run``` magic command in one of the cells and set ```--PLOT_CONFMAT True``` for better visualization with the confusion matrix, for example:
```
%run path/to/main.py --DATA_FOLDER task_1_or_2 --PLOT_CONFMAT True
```

### Citation
```
Chau Pham Quoc Hung at hope2024@ iberlef: Choosing the Right Language Model for the Right Task (2024).
```