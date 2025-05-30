import argparse
import os
from enum import Enum

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from mewtwo.external_code.dnabert.bert_layers import BertModel
from mewtwo.machine_learning.transformer.dnabert_regressor import DNABERTRegressor
from mewtwo.machine_learning.transformer.regressor_dataset import RegressionDataset
from mewtwo.parsers.parse_dnabert_data import parse_dnabert_data
from mewtwo.machine_learning.data_preparation.calculate_sample_weights import get_sample_weights
from mewtwo.machine_learning.transformer.loss_functions import CombinedMSEPearsonLoss, WeightedMSELoss


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=True, help="Tabular input data, with sequence in one column and \
    efficiency in the second")
    parser.add_argument("-v", type=str, required=True,
                        help="Tabular input data, with sequence in one column and efficiency in the second")
    parser.add_argument('-a', type=float, default=0.5, help="Alpha value used for Pearson loss. The higher alpha, \
    the lower the contribution of Pearson correlation to the loss function")

    parser.add_argument("-o", type=str, required=True, help="Output directoru")
    parser.add_argument("-f", type=str, default='linear_head',
                        help="Finetuning mode, must be one of 'linear_head', 'partial', and 'adapter'")
    parser.add_argument("-e", type=int, default=15, help="Nr of epochs")
    parser.add_argument("-lf", type=str, default="mse", help="Loss function. Must be one of: 'mse', 'weighted_mse', \
    'combined_mse_pearson', 'combined_weighted_mse_pearson'")
    parser.add_argument("-lr", type=float, default=1e-3, help="Learning rate. Starting learning rate if scheduler \
    is used.")
    parser.add_argument("-sc", type=str, default=None, help="Learning rate scheduler. If given, must be one of \
    'reduce_on_plateau', 'cos_anneal_warmup'")
    parser.add_argument("-s", type=str, default=None, help="If given, save model to given location")
    parser.add_argument("-m", type=str, default=None, help="If given, train from this model")
    parser.add_argument("-num_expected_epochs", type=int, default=100, help="Number of total epochs estimated to use \
    for training in total. Used to determine nr of warmup steps.")
    parser.add_argument("-lora_r", type=int, default=8, help="Rank for LoRA adapters")
    parser.add_argument("-lora_alpha", type=int, default=16, help="Alpha scaling factor for LoRA adapters")
    parser.add_argument("-lora_dropout", type=float, default=0.1, help="Dropout for LoRA adapters")
    parser.add_argument("-config", type=str, default=None, help="Path to model config file")

    args = parser.parse_args()

    # TODO: Turn into enums
    assert args.sc in [None, 'reduce_on_plateau', 'cos_anneal_warmup']
    assert args.f in ['linear_head', 'partial', 'adapter']
    assert args.lf in ['mse', 'weighted_mse', 'combined_mse_pearson', 'combined_weighted_mse_pearson']

    return args


def evaluate_model(model, dataloader, loss_fn, weighted: bool = False, device="cpu"):
    model.eval()  # Set to evaluation mode
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["weights"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if weighted:
                loss = loss_fn(outputs, labels, weights)
            else:
                loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_preds, all_labels


def train_model(model, dataloader, optimizer, loss_fn, scheduler, scheduler_type, weighted=False, device='cpu', ):
    model.train()  # set to training mode
    model.to(device)

    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        weights = batch["weights"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        if weighted:
            loss = loss_fn(outputs, labels, weights)
        else:
            loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            if scheduler_type != 'reduce_on_plateau':
                scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def initialise(finetuning_mode: FinetuningType, lora_r, lora_alpha, lora_dropout):
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    base_model = BertModel.from_pretrained("zhihan1996/DNABERT-2-117M")

    if finetuning_mode == FinetuningType.LINEAR_HEAD or finetuning_mode == FinetuningType.ADAPTER:
        for param in base_model.parameters():
            param.requires_grad = False

    use_adapters = False

    if finetuning_mode == FinetuningType.ADAPTER:
        use_adapters = True

    model = DNABERTRegressor(base_model, use_adapters=use_adapters, lora_r=lora_r, lora_alpha=lora_alpha,
                             lora_dropout=lora_dropout)

    return model, tokenizer


def prepare_data(input_file: str, tokenizer: AutoTokenizer, shuffle: bool, batch_size: int = 5) \
        -> DataLoader:
    sequences, labels = parse_dnabert_data(input_file)
    sample_weights = get_sample_weights(labels)
    dataset = RegressionDataset(sequences, labels, sample_weights, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def get_loss_function(string_description, alpha=0.5):

    if string_description == 'mse':
        loss_fn = nn.MSELoss()
    elif string_description == 'weighted_mse':
        loss_fn = WeightedMSELoss()
    elif string_description in ['combined_mse_pearson', 'combined_weighted_mse_pearson']:
        loss_fn = CombinedMSEPearsonLoss(alpha=alpha)
    else:
        raise ValueError(f"Unknown loss function type: {string_description}")

    return loss_fn


def main():

    args = parse_arguments()

    if not os.path.exists(args.o):
        os.mkdir(args.o)

    finetuning_mode = FinetuningType.from_string_description(args.f)
    model, tokenizer = initialise(finetuning_mode, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                                  lora_dropout=args.lora_dropout)

    train_dataloader = prepare_data(args.i, tokenizer, True)
    validation_dataloader = prepare_data(args.v, tokenizer, False)

    loss_fn = get_loss_function(args.lf, args.a)

    use_weights = False

    if 'weighted' in args.lf:
        use_weights = True

    summary_file = os.path.join(args.o, "summary.txt")

    scheduler = None
    warmup_steps = None
    training_steps = None

    if args.m is not None:
        summary = open(summary_file, 'a')
        checkpoint = torch.load(args.m)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.update_epoch(checkpoint["epoch"])

        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        # TODO: Store as dataclass instead
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if args.sc is not None:
            if args.sc == 'cos_anneal_warmup':
                warmup_steps = checkpoint["scheduler_num_warmup_steps"]
                training_steps = checkpoint["scheduler_num_training_steps"]
                scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                            num_training_steps=training_steps)
            elif args.sc == 'reduce_on_plateau':
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True,
                                              min_lr=5e-6)
            else:
                raise ValueError(f"Unrecognised scheduler: {args.sc}")

        if checkpoint["scheduler_state_dict"] is not None and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            for param_group in optimizer.param_groups:
                param_group['lr'] = scheduler.get_last_lr()[0]

    else:
        summary = open(summary_file, 'w')
        summary.write("epoch\taverage_train_loss\taverage_eval_loss\n")

        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        if args.sc is not None:
            if args.sc == 'cos_anneal_warmup':

                training_steps = int(len(train_dataloader) * args.num_expected_epochs)
                warmup_steps = int(0.1 * training_steps)

                scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                            num_training_steps=training_steps)
            elif args.sc == 'reduce_on_plateau':
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True,
                                              min_lr=1e-5)
            else:
                raise ValueError(f"Unrecognised scheduler: {args.sc}")

    starting_epoch = model.current_epoch
    current_epoch = starting_epoch

    if args.m is None:
        avg_loss, all_preds, all_labels = evaluate_model(model, validation_dataloader, loss_fn, weighted=use_weights)
        print(f"Epoch {starting_epoch}\t- Eval loss:\t{avg_loss:.4f}")

    for i in range(args.e):
        current_epoch = starting_epoch + i + 1
        print(f"LR at epoch {current_epoch}: {optimizer.param_groups[0]['lr']}")

        out_file = os.path.join(args.o, f"epoch_{current_epoch:03d}.txt")
        avg_train_loss = train_model(model, train_dataloader, optimizer, loss_fn, scheduler, scheduler_type=args.sc,
                                     weighted=use_weights)
        avg_loss, all_preds, all_labels = evaluate_model(model, validation_dataloader, loss_fn,
                                                         weighted=use_weights)
        print(f"Epoch {current_epoch}\t- Train loss:\t{avg_train_loss:.4f}")
        print(f" \t- Eval loss:\t{avg_loss:.4f}")
        summary.write(f"{current_epoch}\t{avg_train_loss:.5f}\t{avg_loss:.5f}\n")

        with open(out_file, 'w') as out:
            out.write("actual\tpredicted\n")
            for j, prediction in enumerate(all_preds):
                label = all_labels[j]
                out.write(f"{label}\t{prediction}\n")

        model.update_epoch(current_epoch)

        if scheduler is not None and args.sc == 'reduce_on_plateau':
            scheduler.step(avg_loss)

    summary.close()

    if args.s is not None:
        torch.save({"model_state_dict": model.state_dict(),
                    "epoch": current_epoch,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scheduler_num_warmup_steps": warmup_steps,
                    "scheduler_num_training_steps": training_steps}, args.s)


if __name__ == "__main__":
    main()
