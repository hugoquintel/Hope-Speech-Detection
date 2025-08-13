import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# function to read data file (.csv)
def get_df(path, folder, dataset):
    data_path = path / folder / f'{dataset}.csv'
    return pd.read_csv(data_path)

# function to get the labels in the data (different for task 1 and task 2)
def get_labels(args, df):
    if args.TASK == 1:
        labels = df['category'].unique()
    else:
        labels = df[args.CLS_TYPE].unique()
    labels_to_ids = {label:id for id, label in enumerate(labels)}
    ids_to_labels = {id:label for label, id in labels_to_ids.items()}
    return labels_to_ids, ids_to_labels

# function to preprocess data (convert text to input ids, create attention mask, encode labels)
# and return a new processed dataframe
def preprocess_data(args, data_df, tokenizer, labels_to_ids):
    data_id = data_df['id'].tolist()
    tokenizer_output = tokenizer(data_df['text'].to_list(), padding='max_length',
                                 truncation=True, max_length=args.PLM_MAX_TOKEN, return_tensors='pt')
    input_ids = tokenizer_output.input_ids.tolist()
    attention_mask = tokenizer_output.attention_mask.tolist()
    label = data_df['category'].map(labels_to_ids).tolist() if args.TASK == 1 else \
            data_df[args.CLS_TYPE].map(labels_to_ids).tolist()
    data_dict = {'data_id': data_id, 'input_ids': input_ids,
                 'attention_mask': attention_mask, 'label': label}
    return pd.DataFrame(data_dict)

# hope data class (pytorch custom dataset)
class HopeDataset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        data_id = self.df['data_id'].iloc[index]
        input_ids = torch.tensor(self.df['input_ids'].iloc[index])
        attention_mask = torch.tensor(self.df['attention_mask'].iloc[index])
        label = self.df['label'].iloc[index]
        data = {'data_id': data_id, 'input_ids': input_ids,
                'attention_mask': attention_mask, 'label': label}
        return data