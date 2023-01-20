import os, sys, gc
import json
import random
import numpy as np
import argparse
import logging
from tqdm import tqdm

from nltk.translate.bleu_score import corpus_bleu

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

from datasets import load_dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    AutoConfig, get_linear_schedule_with_warmup
)
from promptsource.templates import DatasetTemplates

# global params
MODEL_CACHE_DIR = "./models"
DATASET_CACHE_DIR = "./datasets"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


class MultipleChoiceDataset(Dataset):

    def __init__(self, dataset, prompt, choices: list):
        super(MultipleChoiceDataset, self).__init__()
        self.dataset = dataset
        self.prompt = prompt
        self.choices = choices
        self.inputs = []
        self.gts = []
        
        for sample in tqdm(self.dataset, desc="Prompting dataset"):
            inputs_pretokenized, groundtruth_choice = tuple(self.prompt.apply(sample))
            self.inputs.append(inputs_pretokenized.strip() + " Answer: <extra_id_0>")
            self.gts.append(groundtruth_choice)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return {
            "context": self.inputs[index],
            "gt": "<extra_id_0> " + self.gts[index] + " <extra_id_1>",
            "label": self.gts[index],
        }


class ConditionalGenerationDataset(Dataset):

    def __init__(self, dataset, prompt):
        self.dataset = dataset
        self.prompt = prompt
        self.inputs = []
        self.gts = []

        for sample in tqdm(self.dataset, desc="Prompting dataset"):
            inputs_pretokenized, groundtruth_choice = tuple(self.prompt.apply(sample))
            self.inputs.append(inputs_pretokenized.strip())
            self.gts.append(groundtruth_choice)
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return {
            "context": self.inputs[index],
            "target": self.gts[index],
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune T5")
    parser.add_argument("--seed", type=int, default=23333333)
    # checkpoints and log saving directory
    parser.add_argument("--basedir", type=str, default="./log")
    parser.add_argument("--expname", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    # dataset
    parser.add_argument("--cls", action="store_true")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--choices", nargs='+', type=str)
    # model
    parser.add_argument("--backbone", type=str, default="google/flan-t5-xl")
    # fine-tuning options
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--train_bsz", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_bsz", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--maxlen", type=int, default=256)
    parser.add_argument("--maxgenlen", type=int, default=32)
    parser.add_argument("--warmup_steps", type=int, default=3000)
    # testing
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--topk", type=int, default=5)
    # parse args
    args = parser.parse_args()
    return args


def init_logger(logdir):
    logger = logging.getLogger("default")
    cmd_handler = logging.StreamHandler(sys.stdout)
    cmd_handler.setLevel(logging.DEBUG)
    cmd_handler.setFormatter(logging.Formatter(r"[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)s] %(message)s"))
    log_handler = logging.FileHandler(os.path.join(logdir, "train.log"), mode="w+", encoding="utf-8")
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(logging.Formatter(r"[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)s] %(message)s"))
    logger.addHandler(cmd_handler)
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)
    return logger


def finetune_cls_task(dataset_name: str, prompt_name: str, choices: list=["Yes", "No"]):
    # cls prompt
    dataset_names = dataset_name.split("/")
    if len(dataset_names) == 1:
        dataset = load_dataset(dataset_name, cache_dir=DATASET_CACHE_DIR)
    else:
        dataset = load_dataset(dataset_names[0], dataset_names[1], cache_dir=DATASET_CACHE_DIR)
    prompts = DatasetTemplates(dataset_name)
    cls_train_dataset = MultipleChoiceDataset(
        dataset=dataset["train"],
        prompt=prompts[prompt_name],
        choices=choices,
    )
    cls_valid_dataset = MultipleChoiceDataset(
        dataset=dataset["validation"],
        prompt=prompts[prompt_name],
        choices=choices,
    )
    cls_test_dataset = MultipleChoiceDataset(
        dataset=dataset["test"],
        prompt=prompts[prompt_name],
        choices=choices,
    )
    cls_train_dataloader = DataLoader(cls_train_dataset, batch_size=args.train_bsz, shuffle=True, num_workers=args.num_workers, drop_last=True)
    cls_valid_dataloader = DataLoader(cls_valid_dataset, batch_size=args.eval_bsz, shuffle=False, num_workers=args.num_workers)
    cls_test_dataloader  = DataLoader(cls_test_dataset , batch_size=args.eval_bsz, shuffle=False, num_workers=args.num_workers)

    # optimizer and scheduler
    train_batch_size = args.train_bsz * args.gradient_accumulation_steps
    num_training_steps = args.epoch * (len(cls_train_dataset) // train_batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps, betas=(args.beta1, args.beta2))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)

    # fine-tune process
    if not args.test_only:
        for e in range(1, args.epoch+1):
            logger.info(f"Epoch {e}")
            # train
            tqdm_vars = {
                "lr": np.nan,
                "loss": np.nan,
                "norm": np.nan,
            }
            tbar = tqdm(enumerate(cls_train_dataloader, start=1), desc="train", total=len(cls_train_dataloader), postfix=tqdm_vars)
            train_loss_value = 0.0
            model.train()
            for i, sample in tbar:
                encodings = tokenizer(sample["context"], padding="longest", max_length=args.maxlen, return_tensors="pt").to(device)
                input_ids = encodings.input_ids
                attention_mask = encodings.attention_mask
                labels = tokenizer(sample["gt"], padding="longest", max_length=args.maxgenlen, return_tensors="pt").input_ids.to(device)
                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
                loss = loss / args.gradient_accumulation_steps
                train_loss_value += loss.item()
                loss.backward()
                if i % args.gradient_accumulation_steps == 0:
                    norm = clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm, norm_type=2).item()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    tqdm_vars["lr"] = optimizer.state_dict()["param_groups"][0]["lr"]
                    tqdm_vars["loss"] = train_loss_value
                    tqdm_vars["norm"] = norm
                    tbar.set_postfix(tqdm_vars)
                    train_loss_value = 0.0
            # valid
            valid_loss = 0.0
            model.eval()
            with torch.no_grad():
                for i, sample in tqdm(enumerate(cls_valid_dataloader, start=1), desc="valid", total=len(cls_valid_dataloader)):
                    encodings = tokenizer(sample["context"], padding="longest", max_length=args.maxlen, return_tensors="pt").to(device)
                    input_ids = encodings.input_ids
                    attention_mask = encodings.attention_mask
                    labels = tokenizer(sample["gt"], padding="longest", max_length=args.maxgenlen, return_tensors="pt").input_ids.to(device)
                    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
                    valid_loss += loss.item()
            valid_loss = valid_loss / len(cls_valid_dataloader)
            logger.info(f"[VALID] epoch {e}: loss={valid_loss}")
    # test
    test_gts = []
    test_preds = []
    model.eval()
    with torch.no_grad():
        for i, sample in tqdm(enumerate(cls_test_dataloader, start=1), desc="test", total=len(cls_test_dataloader)):
            encodings = tokenizer(sample["context"], padding="longest", max_length=args.maxlen, return_tensors="pt").to(device)
            outputs = model.generate(**encodings, max_length=args.maxgenlen, early_stopping=True, top_k=1)
            for idx, output in enumerate(outputs):
                text = tokenizer.decode(output.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
                test_gts.append(sample["label"][idx])
                test_preds.append(text)
    acc_count = 0
    for gt, pred in zip(test_gts, test_preds):
        if gt in pred:
            acc_count += 1
    test_acc = acc_count / len(test_gts)
    logger.info(f"[TEST] acc={test_acc}")
    gc.collect()
    torch.cuda.empty_cache()


def finetune_gen_task(dataset_name: str, prompt_name: str):
    dataset_names = dataset_name.split("/")
    if len(dataset_names) == 1:
        dataset = load_dataset(dataset_name, cache_dir=DATASET_CACHE_DIR)
    else:
        dataset = load_dataset(dataset_names[0], dataset_names[1], cache_dir=DATASET_CACHE_DIR)
    prompts = DatasetTemplates(dataset_name)
    gen_train_dataset = ConditionalGenerationDataset(
        dataset=dataset["train"],
        prompt=prompts[prompt_name],
    )
    gen_valid_dataset = ConditionalGenerationDataset(
        dataset=dataset["validation"],
        prompt=prompts[prompt_name],
    )
    gen_test_dataset = ConditionalGenerationDataset(
        dataset=dataset["test"],
        prompt=prompts[prompt_name],
    )
    gen_train_dataloader = DataLoader(gen_train_dataset, batch_size=args.train_bsz, shuffle=True, num_workers=args.num_workers, drop_last=True)
    gen_valid_dataloader = DataLoader(gen_valid_dataset, batch_size=args.eval_bsz, shuffle=False, num_workers=args.num_workers)
    gen_test_dataloader  = DataLoader(gen_test_dataset , batch_size=args.eval_bsz, shuffle=False, num_workers=args.num_workers)

    # optimizer and scheduler
    train_batch_size = args.train_bsz * args.gradient_accumulation_steps
    num_training_steps = args.epoch * (len(gen_train_dataset) // train_batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
    
    # generation task
    if not args.test_only:
        for e in range(1, args.epoch+1):
            logger.info(f"Epoch {e}")
            # train
            tqdm_vars = {
                "lr": np.nan,
                "loss": np.nan,
                "ppl": np.nan,
                "norm": np.nan,
            }
            tbar = tqdm(enumerate(gen_train_dataloader, start=1), desc="train", total=len(gen_train_dataloader), postfix=tqdm_vars)
            train_loss_value = 0.0
            model.train()
            for i, sample in tbar:
                encoding = tokenizer(sample["context"], padding="longest", max_length=args.maxlen, truncation=True, return_tensors="pt").to(device)
                input_ids = encoding.input_ids
                attention_mask = encoding.attention_mask
                labels = tokenizer(sample["target"], padding="longest", max_length=args.maxgenlen, truncation=True, return_tensors="pt").input_ids.to(device)
                labels[labels == tokenizer.pad_token_id] = -100
                # the forward function automatically creates the correct decoder_input_ids
                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
                loss = loss / args.gradient_accumulation_steps
                train_loss_value += loss.item()
                loss.backward()
                if i % args.gradient_accumulation_steps == 0:
                    norm = clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm, norm_type=2).item()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    tqdm_vars["lr"] = optimizer.state_dict()["param_groups"][0]["lr"]
                    tqdm_vars["loss"] = train_loss_value
                    tqdm_vars["ppl"] = np.exp(train_loss_value)
                    tqdm_vars["norm"] = norm
                    tbar.set_postfix(tqdm_vars)
                    train_loss_value = 0.0
            # valid
            valid_loss = 0.0
            model.eval()
            with torch.no_grad():
                for i, sample in tqdm(enumerate(gen_valid_dataloader, start=1), desc="valid", total=len(gen_valid_dataloader)):
                    encoding = tokenizer(sample["context"], padding="longest", max_length=args.maxlen, truncation=True, return_tensors="pt").to(device)
                    input_ids = encoding.input_ids
                    attention_mask = encoding.attention_mask
                    labels = tokenizer(sample["target"], padding="longest", max_length=args.maxgenlen, truncation=True, return_tensors="pt").input_ids.to(device)
                    labels[labels == tokenizer.pad_token_id] = -100
                    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
                    valid_loss += loss.item()
            valid_loss = valid_loss / len(gen_valid_dataloader)
            logger.info(f"[VALID] epoch {e}: loss={valid_loss}, ppl={np.exp(valid_loss)}")
    # test
    test_preds = []
    test_targets = []
    model.eval()
    with torch.no_grad():
        for i, sample in tqdm(enumerate(gen_test_dataloader, start=1), desc="test", total=len(gen_test_dataloader)):
            inputs = tokenizer(sample["context"], padding="longest", max_length=args.maxlen, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_length=args.maxgenlen, early_stopping=True, top_k=args.topk)
            for idx, output in enumerate(outputs):
                text = tokenizer.decode(output.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
                text = text.strip().rstrip(".")
                test_targets.append([sample["target"][idx].split()])
                test_preds.append(text.split())
    bleu1 = corpus_bleu(test_targets, test_preds, weights=(1.0, 0.0, 0.0, 0.0))
    bleu2 = corpus_bleu(test_targets, test_preds, weights=(0.5, 0.5, 0.0, 0.0))
    bleu3 = corpus_bleu(test_targets, test_preds, weights=(1./3, 1./3, 1./3, 0.0))
    bleu4 = corpus_bleu(test_targets, test_preds, weights=(0.25, 0.25, 0.25, 0.25))
    logger.info(f"[TEST] BLEU-1/2/3/4={bleu1:.4f}/{bleu2:.4f}/{bleu3:.4f}/{bleu4:.4f}")
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # make logging directory
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    os.makedirs(args.basedir, exist_ok=True)
    logging_dir = os.path.join(args.basedir, args.expname)
    if os.path.exists(logging_dir) and not args.overwrite:
        print(f"[WARN] logging directory {logging_dir} already exists. If you want to overwrite previous logs, use param `--overwrite` please.")
        exit(-1)
    os.makedirs(logging_dir, exist_ok=True)
    # init logging module
    logger = init_logger(logging_dir)
    # save configs
    with open(os.path.join(logging_dir, "config.json"), "w+", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    config = AutoConfig.from_pretrained(args.backbone, cache_dir=MODEL_CACHE_DIR)
    logger.info(config)
    tokenizer = T5Tokenizer.from_pretrained(args.backbone, cache_dir=MODEL_CACHE_DIR)
    model = T5ForConditionalGeneration.from_pretrained(args.backbone, cache_dir=MODEL_CACHE_DIR).to(device)
    logger.info(f"Model {args.backbone} loaded.")
    # save vocab
    vocab_size = len(tokenizer.get_vocab())
    with open(os.path.join(MODEL_CACHE_DIR, "models--" + args.backbone.replace("/", "--"), "vocab.json"), "w+", encoding="utf-8") as f:
        json.dump(tokenizer.get_vocab(), f, indent=2)
    
    if args.cls:
        logger.info(f"Classification Task")
        finetune_cls_task(args.dataset, args.prompt, args.choices)
    else:
        logger.info(f"Generation Task")
        finetune_gen_task(args.dataset, args.prompt)
    
