import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

from dataset import BilingualDataset

from model import build_transformer

from config import get_weights_file_path,get_config

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.tensorboard import SummaryWriter

def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config,ds,lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]','[PAD]','[SOS]','[EOS]'],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    ds_raw = load_dataset('opus_books',f"{config['lang_src']}-{config['lang_tar']}",split='train')


    tokenizer_src = get_or_build_tokenizer(config,ds_raw,config['lang_src'])
    tokenizer_tar = get_or_build_tokenizer(config,ds_raw,config['lang_tar'])

    train_ds_size = int(0.9 * len(ds_raw))

    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw,val_ds_raw = random_split(ds_raw,[train_ds_size,val_ds_size])

    train_ds = BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_tar,config['lang_src'],config['lang_tar'],config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw,tokenizer_src,tokenizer_tar,config['lang_src'],config['lang_tar'],config["seq_len"])

    max_len_src = 0
    max_len_tar = 0

    for item in ds_raw:

        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tar_ids = tokenizer_tar.encode(item['translation'][config['lang_tar']]).ids

        max_len_src = max(len(src_ids),max_len_src)
        max_len_tar = max(len(tar_ids),max_len_tar)

    print(f"the max source sentance length is {max_len_src}")
    print(f"the max target sentance length is {max_len_tar}")

    train_data_loader = DataLoader(train_ds,batch_size=config["batch_size"],shuffle=True)
    val_data_loader = DataLoader(val_ds,batch_size=1,shuffle=True)


    return train_data_loader,val_data_loader,tokenizer_src,tokenizer_tar


def get_model(config,vocab_src_len,vocab_tar_len):

    model = build_transformer(vocab_src_len,vocab_tar_len,config["seq_len"],config["seq_len"],config["d_model"])
    return model


def train_model(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"device is run on {device}")
    Path(config['model_folder']).mkdir(parents=True,exist_ok=True)

    train_data_loader,val_data_loader,tokenizer_src,tokenizer_tar = get_ds(config)

    model = get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tar.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'],eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config["preload"]


    if config['preload']:

        model_filename = get_weights_file_path(config,config['preload'])
        print(f"Preloading model {model_filename}")

        state = torch.load(model_filename)

        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])

        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'),label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch,config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()

        batch_iterator = tqdm(train_data_loader,desc=f"Processing Epoch {epoch:02d}")

        for batch in batch_iterator:

            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)


            encoder_output = model.encode(encoder_input,encoder_mask)
            decoder_output = model.encode(encoder_output,encoder_mask,decoder_input,decoder_mask)

            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)

            loss = loss_fn(proj_output.view(-1,tokenizer_tar.get_vocab_size()),label.view(-1))

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})


            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()                       
            optimizer.step()                

            optimizer.zero_grad(set_to_none=True) 

            global_step += 1

        model_filename = get_weights_file_path(config,f'{epoch:02d}')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config =  get_config()

    train_model(config)