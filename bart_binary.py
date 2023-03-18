import torch
import torch.nn as nn

from datasets import load_dataset
import numpy as np
#####################
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from bart_dataset import IMDBDataset
#####################
from transformers import AdamW
from transformers import get_scheduler
from transformers import BartTokenizer, BertTokenizer
#####################
from bart_model import TestModel 
from tqdm import tqdm
#####################
from torch.optim import AdamW

if torch.cuda.is_available():
    device = "cuda:2"
else:
    device = "cpu"   

device = torch.device(device)

from datasets import load_dataset

print('load imdb')
dataset = load_dataset("imdb")

src_list = dict()
trg_list = dict()

train_dat = dataset['train']
test_dat = dataset['test']


print('tokenizer')

# tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
# bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# reduce the max_length for tokenizer
tokenizer.model_max_length = 200

# train_encoding = tokenizer(
#     train_dat['text'],
#     return_tensors='pt',
#     padding=True,
#     truncation=True
# )
train_encoding = tokenizer(
    train_dat['text'][:int(0.8*len(train_dat['text']))],
    return_tensors='pt',
    padding=True,
    truncation=True
)

val_encoding = tokenizer(
    train_dat['text'][int(0.8*len(train_dat['text'])):],
    return_tensors='pt',
    padding=True,
    truncation=True
)

test_encoding = tokenizer(
    test_dat['text'],
    return_tensors='pt',
    padding=True,
    truncation=True
)

# train_len = int(len(train_dat['text']))
train_len = int(0.8*len(train_dat['text']))
val_len = int(0.2*len(train_dat['text']))
test_len = int(len(test_dat['text']))


print('dataloader')    
train_set = IMDBDataset(train_encoding, train_dat['label'][:train_len])
# train_set = IMDBDataset(train_encoding, train_dat['label'])
val_set = IMDBDataset(val_encoding, train_dat['label'][train_len:])
test_set = IMDBDataset(test_encoding, test_dat['label'])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

from tqdm import tqdm

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
model = TestModel()

# use_cuda = torch.cuda.is_available()
# if use_cuda:

#     model = model.cuda()
#     criterion = criterion.cuda()

print("train loop")
# def train(epoch, model, trainloader, optimizer, device):
def train(epoch, model, trainloader, valloader, optimizer, device):
    model.to(device)
    best_val = 0 # for saving the best model

    for e in range(1, epoch+1):

        total_acc_train = 0
        total_loss_train = 0

        # model training
        model.train()
        for i, train_input in enumerate(tqdm(trainloader)):
            
            train_label = train_input['labels'].to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            
            output, _ = model(input_id, mask)
            
            batch_loss = criterion(output.contiguous(), train_label.long())
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            # model validation
            model.eval()
        with torch.no_grad():

            for i, val_input in enumerate(valloader):

                val_label = val_input['labels'].to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output, _ = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
                
            if best_val < (total_acc_val / val_len):
                best_val = (total_acc_val / val_len)
                
                # save model
                print('save the best model')
                PATH = './model.pt'
                torch.save(model.state_dict(), PATH)
            else:
                print('not the best model')  

        print('='*64)
        print(
            f'Epochs: { e } | Train Loss: {total_loss_train / train_len: .3f} \
            | Train Accuracy: {total_acc_train / train_len: .3f} \
            | Val Loss: {total_loss_val / val_len: .3f} \
            | Val Accuracy: {total_acc_val / val_len: .3f}')
        print('='*64)

print("evaluate loop")        
def evaluate(model, testloader, device):
    model.to(device)
    model.eval()
    
    total_acc_test = 0

    with torch.no_grad():
        
        for i, test_input in enumerate(tqdm(testloader)):

            test_label = test_input['labels'].to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output, _ = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print('='*64)    
    print(f'Test Accuracy: {total_acc_test / test_len: .3f}')
    print('='*64)    

print("train")
optimizer = AdamW(model.parameters(), lr=5e-5)
train(5, model, train_loader, val_loader, optimizer, device)

print("evaluate")
# load model
PATH = './model.pt'
model.load_state_dict(torch.load(PATH))
evaluate(model, test_loader, device)
