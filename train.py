from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer, get_scheduler
from gddc.model import Model
from gddc.masking import get_masked_inputs

torch.manual_seed(73)
np.random.seed(73)
generator = torch.Generator().manual_seed(73)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

bert: BertModel = BertModel.from_pretrained(
    'bert-base-german-cased').to(device)
tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
    'bert-base-german-cased')

mask = tokenizer.mask_token


def mask_sentence(sentence: str, mask: str, max_length: int = 128):
    sentences, positons = get_masked_inputs(
        sentence, ['das', 'dass'], mask, lower=True)
    if len(sentences) == 0:
        raise Exception('No das or dass found in sentence')

    # select a random sentence
    idx = np.random.randint(0, len(sentences))
    random_sent = sentences[idx]
    token_len = positons[idx][1] - positons[idx][0] + 1
    das_len = 3
    label = float(token_len - das_len)  # we get 1 for dass and 0 for das

    tokenized = tokenizer(random_sent, return_tensors='pt',
                          max_length=max_length, padding='max_length', truncation=True)

    return {'input_ids': tokenized['input_ids'], 'attention_mask': tokenized['attention_mask'], 'labels': torch.tensor([label])}


model = Model(bert).to(device)

dataset = load_dataset('alexanderbluhm/wiki_sentences_de_2k')
dataset['train'] = dataset['train'].shuffle(seed=73).select(range(30000))

dataset['train'] = dataset['train'].map(lambda data: mask_sentence(
    data['sentences'], mask), remove_columns=['sentences'])

dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# create train and validation split
dataset = dataset['train'].train_test_split(test_size=0.2)
train_data = dataset['train']
val_data = dataset['test']

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, generator=generator)
valn_dataloader = DataLoader(val_data, batch_size=32)

epochs = 2
num_training_steps = epochs * len(train_dataloader)

# training
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
scheduler = get_scheduler(name='linear', optimizer=optimizer,
                          num_warmup_steps=100, num_training_steps=num_training_steps)

progress_bar = tqdm(range(num_training_steps))

losses = []

model.train()
for epoch in range(epochs):
    for batch in train_dataloader:
        input_ids = batch['input_ids'].squeeze().to(device)
        attention_mask = batch['attention_mask'].squeeze().to(device)
        labels = batch['labels'].to(device)

        out = model(input_ids, attention_mask)
        loss = criterion(out, labels)

        loss.backward()

        # clip gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        optimizer.zero_grad()

        # Optional learning rate scheduler
        scheduler.step()

        progress_bar.update(1)

        # track stats
        losses.append(loss.item())


# smooth losses and plot them
smoothed_losses = []
interval = (int) (len(losses) / 75)

for i in range(0, len(losses), interval):
    chunk = losses[i:i + interval]
    smoothed_losses.append(np.mean(chunk))

plt.plot(smoothed_losses)
plt.show()

# evaluate
model.eval()
accuracy = 0
with torch.inference_mode():
    for batch in valn_dataloader:
        input_ids = batch['input_ids'].squeeze().to(device)
        attention_mask = batch['attention_mask'].squeeze().to(device)
        labels = batch['labels'].to(device)

        out = model(input_ids, attention_mask)
        predicted = torch.round(out)
        B = labels.shape[0]
        acc = (predicted == labels).sum().float() / B

        accuracy += acc.item()

print('Accuracy: ', accuracy / len(valn_dataloader))

with torch.inference_mode():
    # test some sentences
    test_sentences = ["Man merkt schon, dass die Programme deutlich schneller sind.",
                    "Das Haus ist sehr schön", "Dass jenes nicht geht, sollte jedem klar sein", "Das ist ein Test!", "Was soll das heissen?", "Irgendwie denke ich, dass das nicht so gut ist.", "Denkst du dass man im Auto rauchen darf?"]
    for sentence in test_sentences:
        inputs = mask_sentence(sentence, mask)
        out = model(inputs['input_ids'].to(device),
                    inputs['attention_mask'].to(device))
        print(sentence)
        print('Predicted: ', out.item())
        print()

# save the model
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
torch.save(model.state_dict(), f'bert_{timestamp}.pt')
