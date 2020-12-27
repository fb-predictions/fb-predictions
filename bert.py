import transformers
from transformers import BertModel, BertTokenizer, \
    DistilBertTokenizer, DistilBertModel,\
    AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import datetime
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelWithLMHead
from os import listdir
from os.path import isfile, join

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




import argparse

parser = argparse.ArgumentParser(description='Process input')

parser.add_argument('-p', dest='path', type=str, default="processed_data", required=False)
parser.add_argument('-s', dest='saving_path', type=str, default="regressors_out", required=False)
parser.add_argument('-m', dest='model', type=int, default=2, required=False)
parser.add_argument('-lr', dest='lr', type=float, default=2e-5, required=False)
parser.add_argument('-l', dest='max_len', type=int, default=256, required=False)
parser.add_argument('-e', dest='epochs', type=int, default=10, required=False)
parser.add_argument('-f', dest='freeze', type=bool, default=False, required=False)
parser.add_argument('-b', dest='batch_size', type=int, default=16, required=False)
parser.add_argument('-t', dest='train', type=str, default="yes", required=False)
parser.add_argument('-w', dest='pretrianed_weights', type=str, default="no", required=False)

args = parser.parse_args()

path = args.path
saving_path = f"{args.saving_path}/BERT_weight_lstmmodelgpu.pt"

models_names = ['bert-base-cased', 'distilbert-base-cased']
models = [BertModel, DistilBertModel]
tokenizers = [BertTokenizer, DistilBertTokenizer]

SELECTED_MODEL = models[args.model]
SELECTED_TOKENIZER = tokenizers[args.model]
PRE_TRAINED_MODEL_NAME = models_names[args.model]
tokenizer = SELECTED_TOKENIZER.from_pretrained(PRE_TRAINED_MODEL_NAME)
LR = args.lr
MAX_LEN = args.max_len
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size


relevant_features = ['words_count_scaled', 'char_count_scaled', 'is_response', 'long_post', 'day_time_sin', 'day_time_cos', 'neg', 'pos', 'neu', 'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'day_time_sin', 'day_time_cos', 'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6', 'day_7']

q_middle, q_high = None, None
mean, std = None, None

class PostsDataset(Dataset):
    def __init__(self, posts, eng_rate, tokenizer, max_len, relevant_features, followers):
        self.posts = posts
        self.eng_rate = eng_rate
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.relevant_features = relevant_features
        self.followers = followers

    def __len__(self):
        return len(self.posts)
    def __getitem__(self, item):
        post = str(self.posts[item])
        eng_rate = self.eng_rate[item]
        relevant_features = self.relevant_features.iloc[item]
        followers = self.followers[item]
        encoding = self.tokenizer.encode_plus(
            post,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        return {
            'posts_text': post,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_masks': encoding['attention_mask'].flatten(),
            'eng_rate': torch.tensor(eng_rate, dtype=torch.float),
            'non_text_features': torch.tensor(relevant_features.values.astype(np.float32), dtype=torch.float),
            'followers':  torch.tensor(followers, dtype=torch.float)
        }


def create_data_loader(df, tokenizer, max_len, batch_size, shuffle=True):
    ds = PostsDataset(
        posts=df.post_text.to_numpy(), eng_rate=df.eng_rate.to_numpy(), tokenizer=tokenizer,
        max_len=max_len, relevant_features=df[relevant_features], followers=df.fitted_followers.to_numpy())
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4
    )

class Regressor(nn.Module):
    def __init__(self, num_features):
        super(Regressor, self).__init__()
        # non-text part
        self.fc1 = nn.Linear(num_features, 100)
        self.fc2 = nn.Linear(100, 100)

        # text part
        self.bert = SELECTED_MODEL.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.1)

        # ensemble part
        self.fc3 = nn.Linear(self.bert.config.hidden_size + 100, 434)
        self.fc4 = nn.Linear(434, 100)
        self.fc5 = nn.Linear(100, 100)
        self.fc6 = nn.Linear(100, 1)


    def forward(self, input_ids, attention_mask, non_text_features):
        # text part
        if args.model != 1:
            _, pooled_output = self.bert(
              input_ids=input_ids,
              attention_mask=attention_mask
            )
        else:
            pooled_output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )[0][:, 0, :]
        bert_out = self.drop3(pooled_output)

        # non-text part
        non_text_out = F.relu(self.drop1(self.fc1(non_text_features)))
        non_text_out = F.relu(self.drop2(self.fc2(non_text_out)))

        # ensemble part
        ensemble_features = torch.cat((bert_out, non_text_out), dim=1)
        out = F.relu(self.fc3(ensemble_features))
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))
        return self.fc6(out)


class lossCalculator:
    def __init__(self):
        self.main_loss = []
        self.ratios = {"low": 0, "middle": 0, "high": 0, "total": 0}
        self.diffs = {"low": 0, "middle": 0, "high": 0, "total": 0}
        self.accuracy = {"low": 0, "middle": 0, "high": 0, "total": 0}
        self.counter = {"low": 0, "middle": 0, "high": 0, "total": 0}

    def calcLosses(self, main_loss_item, eng_rate, outputs):
        self.main_loss.append(main_loss_item)
        self.ratios["total"] += val_ratio(outputs, eng_rate)
        self.diffs["total"] += val_difference(outputs, eng_rate)

        eng_groups = {
            "low": eng_rate <= q_middle,
            "middle": (eng_rate > q_middle) & (eng_rate <= q_high),
            "high": eng_rate > q_high
        }

        preds_groups = {
            "low": outputs <= q_middle,
            "middle": (outputs > q_middle) & (outputs <= q_high),
            "high": outputs > q_high
        }

        # zeros = torch.zeros_like(eng_rate)
        # ones = torch.ones_like(eng_rate)

        for name, sep in eng_groups.items():
            if eng_rate[sep].shape[0] != 0:
                self.ratios[name] += val_ratio(outputs[sep], eng_rate[sep])
                self.diffs[name] += val_difference(outputs[sep], eng_rate[sep])
                #acc = torch.where(preds_groups[name], ones, zeros) * torch.where(sep, ones, zeros)
                #self.accuracy[name] += torch.sum(acc).item()
                self.accuracy[name] += torch.sum(preds_groups[name] & sep).item()
                self.counter[name] += eng_rate[sep].shape[0]

        self.counter["total"] = sum([self.counter[v] for v in ["low", "middle", "high"]])
        self.accuracy["total"] = sum([self.accuracy[v] for v in ["low", "middle", "high"]])

    def results(self, h):
        """
        all_correct = correct_low + correct_middle + correct_middle
        all_count = count_low + count_middle + count_high
        result = {"loss_ratio": losses_ratio / all_count, "loss_diff": losses_difference / all_count, "loss_train": loss_for_train / all_count, "acc": all_correct / all_count,
                "loss_ratio_low": losses_ratio_low / count_low, "loss_diff_low": losses_difference_low / count_low, "acc_low": correct_low / count_low,
                "loss_ratio_middle": losses_ratio_middle / count_middle, "loss_diff_middle": losses_difference_middle / count_middle, "acc_middle": correct_middle / count_middle,
                "loss_ratio_high": losses_ratio_high / count_high, "loss_diff_high": losses_difference_high / count_high, "acc_high": correct_high / count_high}
        """

        main_loss = np.mean(self.main_loss)
        print(f'{h} MAIN loss {main_loss}')

        categories = {"ratio": self.ratios, "diff": self.diffs, "acc": self.accuracy}
        for name, cat in categories.items():
            print(f'{h} loss {name} {cat["total"] / self.counter["total"]}')
            glosses = ", ".join([f"{g} - {str(cat[g] / self.counter[g]) if self.counter[g] > 0 else '∅'}" for g in ["low", "middle", "high"]])
            print(f'{h} loss {name} {glosses}')

        return main_loss

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model = model.train()
    epoch_loss = lossCalculator()

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_masks"].to(device)
        eng_rate = d["eng_rate"].to(device)
        non_text_features = d["non_text_features"].to(device)
        # followers = d["followers"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask, non_text_features=non_text_features).float()
        outputs = torch.squeeze(outputs, dim=1)

        loss = loss_fn(scale_sigmoid(eng_rate), scale_sigmoid(outputs))

        epoch_loss.calcLosses(loss.item(), eng_rate, outputs)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    epoch_loss.results("Train")

def eval_epoch(model, data_loader, loss_fn, device):
    model = model.eval()
    epoch_loss = lossCalculator()

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_masks"].to(device)
            eng_rate = d["eng_rate"].to(device)
            non_text_features = d["non_text_features"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask, non_text_features=non_text_features).float()
            outputs = torch.squeeze(outputs, dim=1)

            loss = loss_fn(scale_sigmoid(eng_rate), scale_sigmoid(outputs))
            epoch_loss.calcLosses(loss.item(), eng_rate, outputs)

    return epoch_loss.results("Val")


def train(model, train_data_loader, val_data_loader, loss_fn, optimizer, device, scheduler, df_train, df_val):
    best_loss = np.inf
    saving_path = f"{args.name}_lstmmodelgpu.pt"


    for epoch in range(EPOCHS):

        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler)

        print()

        result = eval_epoch(model, val_data_loader, loss_fn, device)
        print()

        # history['train_loss'].append(train_loss)
        # history['train_loss_low'].append(train_loss_low)
        # history['train_loss_middle'].append(train_loss_middle)
        # history['train_loss_high'].append(train_loss_high)
        #
        #
        # history['val_loss'].append(val_loss)
        # history['val_loss_low'].append(val_loss_low)
        # history['val_loss_middle'].append(val_loss_middle)
        # history['val_loss_high'].append(val_loss_high)


        if result < best_loss:
            torch.save(model, saving_path)
            best_loss = result

    # plt.plot(history['train_loss'], label='train loss')
    # plt.plot(history['val_loss'], label='validation loss')
    #
    #
    # plt.title('Training history')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend()
    # plt.ylim([0, 0.02])
    # plt.savefig(f"{args.name}_train_validation.pdf")
    # plt.clf()
    # plt.cla()
    #
    # plt.plot(history['val_loss'], label='validation loss')
    # plt.plot(history['val_loss_low'], label='validation loss low')
    # plt.plot(history['val_loss_middle'], label='validation loss middle')
    # plt.plot(history['val_loss_high'], label='validation loss high')
    #
    # plt.title('Validation losses history')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend()
    # plt.ylim([0, 0.02])
    # plt.savefig(f"{args.name}_validation.pdf")
    # plt.clf()
    # plt.cla()
    #
    # plt.plot(history['train_loss'], label='train loss')
    # plt.plot(history['train_loss_low'], label='train loss low')
    # plt.plot(history['train_loss_middle'], label='train loss middle')
    # plt.plot(history['train_loss_high'], label='train loss high')
    #
    # plt.title('Train losses history')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend()
    # plt.ylim([0, 0.01])
    # plt.savefig(f"{args.name}_train.pdf")
    # plt.clf()
    # plt.cla()

def compute_mean_std(df_train):
    mean = np.mean(df_train["eng_rate"])

    std = np.std(df_train["eng_rate"])
    return mean, std

def compute_high_middle(df_train):
    q_middle = df_train["eng_rate"].median()
    q_high = df_train[df_train["eng_rate"] >= q_middle]["eng_rate"].median()
    return q_middle, q_high

def scale_sigmoid(targets):
    targets = (targets - mean.item()) / std.item()
    targets = torch.sigmoid(targets)
    return targets


# def loss_train(target, pred):
#     pred = torch.max(pred, torch.zeros_like(pred))
#     pred = pred + 1
#     target = target + 1
#     return 0.5 * ((target / pred) + (pred / target)) * torch.pow((target - pred), 2)
#

# def loss_train(target, pred):
#     loss = torch.log(target + 1) - torch.log(pred + 1)
#     return torch.pow(loss, 2)

# def loss_train(target, pred):
#     loss = torch.log(target + 1) - torch.log(pred + 1)
#     loss = torch.pow(loss, 2)
#     loss = torch.where(((target >= q_middle) | (target > q_high)), loss * 2, loss)
#
#     return loss


def loss_train(target, pred):
    # loss = torch.log(target + 1) - torch.log(pred + 1)
    loss = target - pred

    loss = torch.pow(loss, 2)
    # loss = torch.where(((target <= q_middle) & (pred > q_middle)), loss * 3, loss)
    # loss = torch.where(((target > q_high) & (pred <= q_high)), loss * 1.5, loss)
    return loss

# def loss_train(target, pred):
#     msle = torch.log(target + 1) - torch.log(pred + 1)
#     msle = torch.pow(msle, 2)
#
#     mse = target - pred
#     mse = torch.pow(mse, 2)
#
#     loss = torch.where(target > q_high,  0.9 * msle + 0.1 * mse, 0.5 * msle + 0.5 * mse)
#     loss = torch.where(target <= q_middle, 0.9 * msle + 0.1 * mse, loss)
#     return loss

# def loss_train(target, pred):
#     msle = torch.log(target + 1) - torch.log(pred + 1)
#     msle = torch.pow(msle, 2)
#
#     mse = target - pred
#     mse = 2 * torch.pow(mse, 2)
#
#     loss = torch.where(target > q_high,  2 * msle, 0.5 * msle + 0.5 * mse)
#     loss = torch.where(target <= q_middle, 3 * mse, loss)
#     return loss

# def loss_train(target, pred):
#     msle = torch.log(target + 1) - torch.log(pred + 1)
#     msle = torch.pow(msle, 2)
#
#     mse = target - pred
#     mse = torch.pow(mse, 2)
#
#     loss = torch.where(target > q_high,  1.8 * msle, msle)
#     loss = torch.where(target <= q_middle, mse, loss)
#     return loss

def val_ratio(target, pred):
    out = torch.min(target / pred, pred / target)
    return torch.sum(out).item()

def val_difference(target, pred):
    out = torch.abs(target - pred)
    return torch.sum(out).item()

class LossFunc(nn.Module):
    def __init__(self, loss_func):
        super(LossFunc, self).__init__()
        self.loss_func = loss_func

    def forward(self, target, pred, followers=None):
        if followers is None:
            out = self.loss_func(target, pred)
        else:
            out = self.loss_func(target, pred, followers)
        return torch.mean(out)


def balance_data(df):
    rate_squared = np.power(df["eng_rate"], 2)
    p = rate_squared / np.sum(rate_squared)
    df = df.reset_index()
    posts = np.random.choice(df.index.to_numpy(), size=int(len(df) * (2 / 3)), p=p.to_numpy(), replace=False)
    df = df.iloc[posts]
    return df

def get_predictions(model, df_val):

  model = model.eval()
  all_outputs = torch.tensor([])
  data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE, False)
  with torch.no_grad():
      for d in data_loader:
          input_ids = d["input_ids"].to(device)
          attention_mask = d["attention_masks"].to(device)
          non_text_features = d["non_text_features"].to(device)
          eng_rate = d["eng_rate"].to(device)
          outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask, non_text_features=non_text_features).float()
          outputs = torch.squeeze(outputs, dim=1)
          all_outputs = torch.cat((all_outputs, outputs.to("cpu")), dim=0)
  df_val['predictions'] = all_outputs.numpy()
  print("hi")
  df_val.to_csv(f"{args.saving_path}/BERT_predictions.csv")

def main():
    global mean, std,  q_middle, q_high
    df_train = pd.read_csv(join(path, "all_train.csv"))
    df_train = df_train[df_train["year"] >= 2018]

    q_middle, q_high = compute_high_middle(df_train)

    # df_train = balance_data(df_train)

    df_val = pd.read_csv(join(path, "all_validation.csv"))
    df_val = df_val[df_val["year"] >= 2018]

    # df_test = pd.read_csv(join(path, "all_test.csv"))
    # for col in relevant_features:
    #     df_train[col].apply(pd.to_numeric, downcast='float', errors='coerce')
    #     df_val[col].apply(pd.to_numeric, downcast='float', errors='coerce')

    # relevant_features = ['words_count_scaled']
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    # test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    mean, std = compute_mean_std(df_train)
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)

    if args.pretrined_weights == "no":
        model = Regressor(len(relevant_features))

    else:
        model = torch.load(args.pretrined_weights)

    model = model.to(device)

    # lr values = 2e-5, 3e-5, 4e-5, 5e-5, 2e-6, 3e-6
    optimizer = AdamW([
            {"params": model.bert.parameters(), "lr": 2e-5},
            {"params": model.drop1.parameters()},
            {"params": model.drop2.parameters()},
            {"params": model.drop3.parameters()},
            {"params": model.fc1.parameters(), "lr": 1e-4},
            {"params": model.fc2.parameters(), "lr": 1e-4},
            {"params": model.fc3.parameters(), "lr": 1e-4},
            {"params": model.fc4.parameters(), "lr": 1e-4},
            {"params": model.fc5.parameters(), "lr": 1e-4},
            {"params": model.fc6.parameters(), "lr": 1e-4},

    ], lr=LR, correct_bias=False)

    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    # scheduler = ReduceLROnPlateau(optimizer, 'min')
    if args.freeze == True:
        for param in model.bert.parameters():
            param.requires_grad = False

    loss_fn = nn.MSELoss().to(device)
    if args.train == "yes":
        train(model, train_data_loader, val_data_loader, loss_fn, optimizer, device, scheduler, df_train, df_val)
    # model = torch.load(saving_path)
    get_predictions(model, df_val)
if __name__ == "__main__":
    main()