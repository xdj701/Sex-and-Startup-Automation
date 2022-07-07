import time
import torch
import pandas as pd, numpy as np
pd.options.display.float_format = '{:,.2f}'.format

from transformers import Trainer, TrainingArguments

COL_QUESTION = "Question"
COL_CONTEXT = "Context"
COL_ANSWER = "Answer"
COL_SERIES_NAME = "Series Name"

DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_NUM_EPOCHS = 25

def import_data_from_excel(file_path, sheet_name):
    data = pd.read_excel(file_path, sheet_name, dtype={COL_QUESTION: str, COL_CONTEXT: str, COL_ANSWER: str, COL_SERIES_NAME: str})

    # drop rows without answers
    data = data[data[COL_ANSWER].notna()]

    # fill the value for merged cells
    data[COL_QUESTION] = data[COL_QUESTION].fillna(method='ffill')
    data[COL_CONTEXT] = data[COL_CONTEXT].fillna(method='ffill')
    
    # Questions in some sheets are not complete sentences
    if COL_SERIES_NAME in data.columns:
        data[COL_SERIES_NAME] = data[COL_SERIES_NAME].fillna(method='ffill')
        data[COL_QUESTION] = data.apply(lambda x: x[COL_QUESTION].format(x[COL_SERIES_NAME]), axis=1)
    return data


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    
def add_token_positions(tokenizer, encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        answer_start = answers[i]['answer_start']
        answer_end = answers[i]['answer_end']
        
        # if answer_start equals to -1, there is no answer available from the context
        if answer_start == -1:
            start_positions.append(tokenizer.model_max_length)
            end_positions.append(tokenizer.model_max_length)
        else:
            start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
            end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

            # if start position is None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length

            # if end position is None, the 'char_to_token' function may point to a space 
            # near the correct token. Try forward looking first
            if end_positions[-1] is None:
                end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] + 1)
                # try backward looking
                if end_positions[-1] is None:
                    end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - 1) + 1
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
    return encodings


def transform_classification_data(tokenizer, train_df, val_df, with_context=False):
    
    if with_context:
        train_encodings = tokenizer(train_df[COL_QUESTION].tolist(), train_df[COL_CONTEXT].tolist(), padding='max_length')
        val_encodings = tokenizer(val_df[COL_QUESTION].tolist(), val_df[COL_CONTEXT].tolist(), padding='max_length')
    else:
        train_encodings = tokenizer(train_df[COL_CONTEXT].tolist(), padding='max_length')
        val_encodings = tokenizer(val_df[COL_CONTEXT].tolist(), padding='max_length')

    train_dataset = TorchDataset(train_encodings, train_df[COL_ANSWER].tolist())
    val_dataset = TorchDataset(val_encodings, val_df[COL_ANSWER].tolist())

    return train_dataset, val_dataset


def train_model(model, train_dataset, val_dataset, model_type, learning_rate = DEFAULT_LEARNING_RATE, epochs = DEFAULT_NUM_EPOCHS):

    timestamp = int(time.time())

    training_args = TrainingArguments(
        output_dir='./models/{}/{}'.format(model_type, timestamp),
        num_train_epochs=epochs,              # total number of training epochs
        per_device_train_batch_size=5,  # batch size per device during training
        per_device_eval_batch_size=5,   # batch size for evaluation
        weight_decay=0.01,               # strength of weight decay
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        load_best_model_at_end = True,
        learning_rate=learning_rate,
        logging_dir='./logs/{}/{}'.format(model_type, timestamp),
        logging_steps=5,
    )

    trainer = Trainer(
        model=model,                 # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )

    trainer.train()
    
    return model
    
