import os, sys, gc
import json
import random
import numpy as np
import argparse
import logging
from tqdm import tqdm
from typing import List

from sklearn.metrics import accuracy_score
from scipy.linalg import block_diag
from nltk.translate.bleu_score import corpus_bleu

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, get_linear_schedule_with_warmup
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


# Forward results by single sample is slow. The following codes organize a batch of inputs to speed up training.
def build_multiple_choice_sample(context, choices):
    context_id = tokenizer(context)['input_ids']

    division = len(context_id)
    mask_position = context_id.index(tokenizer.mask_token_id)

    token = np.array(context_id, dtype=np.int64)
    attention_mask = [np.ones((division, division), dtype=np.int64)]
    position_id = np.arange(division, dtype=np.int64)
    block_position_id = np.zeros(division, dtype=np.int64)

    choice_target_id = []
    choice_id = []

    for choice_str in choices:
        choice = np.array(tokenizer(choice_str)['input_ids'][1:-1], dtype=np.int64)

        choice_id.append(choice)
        choice_target_id.append(np.arange(len(token), len(token) + len(choice), dtype=np.int64))
        attention_mask.append(np.tril(np.ones((len(choice), len(choice)), dtype=np.int64)))

        token = np.concatenate((token, [tokenizer.sop_token_id], choice[:-1]))
        position_id = np.concatenate((position_id, [mask_position] * len(choice)))
        block_position_id = np.concatenate((block_position_id, np.arange(1, 1 + len(choice), dtype=np.int64)))

    attention_mask = block_diag(*attention_mask)
    attention_mask[division:, :division] = 1

    return {
        "token": token,
        "position_id": np.stack((position_id, block_position_id)),
        "attention_mask": attention_mask,
        "choices": choice_id,
        "choice_target_ids": choice_target_id
    }


def pad_batch(tokens, position_ids, attention_mask, max_seq_length):
    pad_length = max_seq_length - len(tokens)
    attention_mask = np.pad(
        attention_mask,
        pad_width=((0, pad_length),),
        mode="constant",
        constant_values=0,
    )
    tokens = np.concatenate((tokens, np.zeros(pad_length, dtype=np.int64)))
    position_ids = np.concatenate((position_ids, position_ids[..., -1:].repeat(pad_length, -1)), axis=-1)
    return tokens, position_ids, attention_mask
    
    
def collate_fn(samples):
    TILE = 16
    length_to_pad = (max(map(lambda spl: len(spl["token"]), samples)) + TILE - 1) // TILE * TILE

    token_batch, position_id_batch, attention_mask_batch = [], [], []
    choices_batch, choice_target_ids_batch = [], []

    for sample in samples:
        token, position_id, attention_mask = pad_batch(
            sample["token"], sample["position_id"], sample["attention_mask"], length_to_pad
        )
        token_batch.append(token)
        position_id_batch.append(position_id)
        attention_mask_batch.append(attention_mask)
        choices_batch.append(sample["choices"])
        choice_target_ids_batch.append(sample["choice_target_ids"])

    return {
        "tokens": torch.tensor(np.array(token_batch), dtype=torch.int64),
        "position_ids": torch.tensor(np.array(position_id_batch), dtype=torch.int64),
        "attention_mask": torch.tensor(np.array(attention_mask_batch), dtype=torch.int64),
        "choices": choices_batch,
        "choice_target_ids": choice_target_ids_batch,
    }
    
def cond_log_prob(context: List[str], choices: List[List[str]]) -> List[List[float]]:
    """
    Compute conditonal probability for one or more continuation/infilling options.
    :return The log probablity of each option.
    """
    if not isinstance(context, list):
        context = [context]
        choices = [choices]
    choices = [[(' ' + choice) for choice in choice_pair] for choice_pair in choices]  # Feature of SentencePiece tokenizer

    samples = [build_multiple_choice_sample(ctx, ch) for ctx, ch in zip(context, choices)]
    
    batch = collate_fn(samples)
    
    logits = model.forward(input_ids=batch['tokens'].cuda(),
                           attention_mask=batch['attention_mask'].cuda().unsqueeze(1),
                           position_ids=batch['position_ids'].cuda())['logits']
    
    log_probs = []
    
    for output, choices, choice_target_ids in zip(F.log_softmax(logits, dim=-1), batch['choices'], batch['choice_target_ids']):
        log_probs_single = []
        for choice, choice_target_id in zip(choices, choice_target_ids):
            tmp = output[choice_target_id, choice]
            log_probs_single.append(tmp.sum())
        log_probs.append(torch.stack(log_probs_single))
    
    return torch.stack(log_probs)


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
            self.inputs.append(inputs_pretokenized.strip() + " Answer: [MASK]")
            self.gts.append(groundtruth_choice)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return {
            "context": self.inputs[index],
            "gt": self.choices.index(self.gts[index]),
        }


class ConditionalGenerationDataset(Dataset):
    
    def __init__(self, dataset, prompt):
        super(ConditionalGenerationDataset, self).__init__()
        self.dataset = dataset
        self.prompt = prompt
        self.inputs = []
        self.gts = []

        for sample in tqdm(self.dataset, desc="Prompting dataset"):
            inputs_pretokenized, groundtruth_choice = tuple(self.prompt.apply(sample))
            self.inputs.append(inputs_pretokenized.strip() + " Answer: [MASK]")
            self.gts.append(groundtruth_choice)
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return {
            "context": self.inputs[index],
            "target": self.gts[index],
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune GLM")
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
    parser.add_argument("--backbone", type=str, default="BAAI/glm-roberta-large")
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
    parser.add_argument("--eval_bsz", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--maxlen", type=int, default=256)
    parser.add_argument("--maxgenlen", type=int, default=32)
    parser.add_argument("--warmup_steps", type=int, default=3000)
    # testing
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--num_beams", type=int, default=1)
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
                logits = cond_log_prob(sample["context"], [choices for _ in range(args.train_bsz)])
                gts = sample["gt"].cuda()
                loss = F.nll_loss(logits, gts)
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
            valid_gts = []
            valid_preds = []
            model.eval()
            with torch.no_grad():
                for i, sample in tqdm(enumerate(cls_valid_dataloader, start=1), desc="valid", total=len(cls_valid_dataloader)):
                    logits = cond_log_prob(sample["context"], [choices for _ in range(args.eval_bsz)])
                    gts = sample["gt"].cuda()
                    loss = F.nll_loss(logits, gts)
                    valid_loss += loss.item()
                    valid_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
                    valid_gts.extend(np.array(sample["gt"]).tolist())
            valid_loss = valid_loss / len(cls_valid_dataloader)
            valid_acc = accuracy_score(valid_preds, valid_gts)
            logger.info(f"[VALID] epoch {e}: loss={valid_loss}, acc={valid_acc}")
    # test
    test_loss = 0.0
    test_gts = []
    test_preds = []
    model.eval()
    with torch.no_grad():
        for i, sample in tqdm(enumerate(cls_test_dataloader, start=1), desc="test", total=len(cls_test_dataloader)):
            logits = cond_log_prob(sample["context"], [choices for _ in range(args.eval_bsz)])
            gts = sample["gt"].cuda()
            loss = F.nll_loss(logits, gts)
            test_loss += loss.item()
            test_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
            test_gts.extend(np.array(sample["gt"]).tolist())
    test_loss = test_loss / len(cls_test_dataloader)
    test_acc = accuracy_score(test_preds, test_gts)
    logger.info(f"[TEST] loss={test_loss}, acc={test_acc}")
    gc.collect()
    torch.cuda.empty_cache()


def finetune_gen_task(dataset_name: str, prompt_name: str):
    tokenizer.pad_token_id = tokenizer.eop_token_id
    model.config.pad_token_id = tokenizer.eop_token_id
    # cls prompt
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

    # fine-tune process
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
                inputs = tokenizer(sample["context"], padding="longest", max_length=args.maxlen, return_tensors="pt")
                inputs = tokenizer.build_inputs_for_generation(inputs, targets=sample["target"], max_gen_length=args.maxgenlen)
                inputs = inputs.to('cuda')
                inputs["input_ids"] = torch.where(inputs["input_ids"] == -100, torch.full_like(inputs["input_ids"], tokenizer.eop_token_id), inputs["input_ids"])
                outputs = model(**inputs)
                loss = outputs.loss
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
                    inputs = tokenizer(sample["context"], padding="longest", max_length=args.maxlen, return_tensors="pt")
                    inputs = tokenizer.build_inputs_for_generation(inputs, targets=sample["target"], max_gen_length=args.maxgenlen)
                    inputs = inputs.to('cuda')
                    inputs["input_ids"] = torch.where(inputs["input_ids"] == -100, torch.full_like(inputs["input_ids"], tokenizer.eop_token_id), inputs["input_ids"])
                    outputs = model(**inputs)
                    loss = outputs.loss
                    valid_loss += loss.item()
            valid_loss = valid_loss / len(gen_valid_dataloader)
            logger.info(f"[VALID] epoch {e}: loss={valid_loss}, ppl={np.exp(valid_loss)}")
    # test
    test_preds = []
    test_targets = []
    model.eval()
    with torch.no_grad():
        for i, sample in tqdm(enumerate(gen_test_dataloader, start=1), desc="test", total=len(gen_test_dataloader)):
            inputs = tokenizer(sample["context"], padding="longest", max_length=args.maxlen, return_tensors="pt")
            inputs = tokenizer.build_inputs_for_generation(inputs)
            inputs = inputs.to('cuda')
            inputs["input_ids"] = torch.where(inputs["input_ids"] == -100, torch.full_like(inputs["input_ids"], tokenizer.eop_token_id), inputs["input_ids"])
            outputs = model.generate(**inputs, max_length=args.maxgenlen+args.maxlen, eos_token_id=tokenizer.eop_token_id, early_stopping=True, top_k=args.topk, do_sample=True, num_beams=min(args.num_beams, args.topk))
            for idx, output in enumerate(outputs):
                text = tokenizer.decode(output.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
                text = text[text.index("Answer:"):].lstrip("Answer:").rstrip(".").strip()
                test_targets.append([sample["target"][idx].split()])
                test_preds.append(text.split())
    bleu1 = corpus_bleu(test_targets, test_preds, weights=(1.0, 0.0, 0.0, 0.0))
    bleu2 = corpus_bleu(test_targets, test_preds, weights=(0.5, 0.5, 0.0, 0.0))
    bleu3 = corpus_bleu(test_targets, test_preds, weights=(1./3, 1./3, 1./3, 0.0))
    bleu4 = corpus_bleu(test_targets, test_preds, weights=(0.25, 0.25, 0.25, 0.25))
    logger.info(f"[TEST] BLEU-1/2/3/4={bleu1:.4f}/{bleu2:.4f}/{bleu3:.4f}/{bleu4:.4f}")
    gc.collect()
    torch.cuda.empty_cache()


# params: lr, bsz, epoch
# models: glm, roberta(mlm), t5, BART, glm with linear head-based fine-tuning
# 3 other prompts for each task
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

    # Download the model and the tokenizer
    # DEFAULT: glm-2b, which is able for good zero-shot short blanking infilling ([MASK]) and long left-to-right generation ([gMASK])
    # If you want to do fine-tuning on language understanding or generation,
    # try smaller glm-roberta-large (335M, not for zero-shot)
    config = AutoConfig.from_pretrained(args.backbone, trust_remote_code=True, revision='main', cache_dir=MODEL_CACHE_DIR)
    logger.info(config)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True, revision='main', cache_dir=MODEL_CACHE_DIR)
    # save vocab
    vocab_size = len(tokenizer.get_vocab())
    with open(os.path.join(MODEL_CACHE_DIR, "models--" + args.backbone.replace("/", "--"), "vocab.json"), "w+", encoding="utf-8") as f:
        json.dump(tokenizer.get_vocab(), f, indent=2)
    # build model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.backbone, trust_remote_code=True, revision='main', cache_dir=MODEL_CACHE_DIR).to(device)
    logger.info(f"Model {args.backbone} loaded.")

    #! Generation task: wiqa
    #! Classification task: qasc

    if args.cls:
        logger.info(f"Classification Task")
        finetune_cls_task(args.dataset, args.prompt, args.choices)
    else:
        logger.info(f"Generation Task")
        finetune_gen_task(args.dataset, args.prompt)
