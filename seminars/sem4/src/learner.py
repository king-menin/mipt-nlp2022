import logging
import os

import torch
from sklearn_crfsuite.metrics import flat_classification_report

from src import tqdm
from .data_with_vocab import LearnData as LearnDataWitVocab
from .data import LearnData
from .models import BiLSTMCRF
from .optim import BertAdam
from .tblog import TensorboardLog
from .utils import save_pkl, if_none

logging.basicConfig(level=logging.INFO)


def train_step(dl, model, optimizer, num_epoch=1):
    model.train()
    epoch_loss = 0
    idx = 0
    pr = tqdm(dl, total=len(dl), leave=False)
    for batch in pr:
        idx += 1
        model.zero_grad()
        loss = model.score(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.data.cpu().tolist()
        epoch_loss += loss
        pr.set_description("train loss: {}".format(epoch_loss / idx))
        torch.cuda.empty_cache()
    logging.info("\nepoch {}, average train epoch loss={:.5}\n".format(
        num_epoch, epoch_loss / idx))


def transformed_result(batch, y_preds, id2label):
    preds_cpu = []
    targets_cpu = []
    for y_true, y_pred, label_mask in zip(batch["labels"], y_preds, batch["labels_mask"]):
        sample_len = label_mask.sum()
        y_true = y_true.cpu().data.tolist()[:sample_len]
        y_true = [id2label[x] for x in y_true]

        y_pred = y_pred.cpu().data.tolist()[:sample_len]
        y_pred = [id2label[x] for x in y_pred]
        preds_cpu.append(y_pred)
        targets_cpu.append(y_true)
    return preds_cpu, targets_cpu


def validate_step(dl, model, id2label):
    model.eval()
    idx = 0
    preds_cpu, targets_cpu = [], []
    for batch in tqdm(dl, total=len(dl), leave=False):
        idx += 1
        preds = model.forward(batch)
        preds_cpu_, targets_cpu_ = transformed_result(batch, preds, id2label)
        preds_cpu.extend(preds_cpu_)
        targets_cpu.extend(targets_cpu_)
    clf_report = flat_classification_report(targets_cpu, preds_cpu, digits=3)
    return clf_report


class NerLearner(object):

    def __init__(
            self,
            train_df_path,
            valid_df_path,
            embedder,
            tensorboard_dir,
            # BiLSTM params
            embedding_size=300, hidden_dim=100, rnn_layers=1, lstm_dropout=0.3,
            # CRFDecoder params
            crf_dropout=0.5,
            shuffle=True, device="cuda", batch_size=16, num_workers=0,
            pad_token="<pad>",
            pad_label=0,
            # Optimizer
            lr=2e-4, warmup=0.1, t_total=None, schedule='warmup_linear',
            b1=0.8, b2=0.999, e=1e-6, weight_decay=0.01, max_grad_norm=5.0,
            epochs=10, save_every=1, update_freq=1, target_metric="accuracy",
            checkpoint_dir="checkpoints",
            validate_every=1,
            use_embeds=False
    ):
        args = locals()
        args.pop("embedder")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        save_pkl(args, os.path.join(checkpoint_dir, "args.pkl"))
        if use_embeds:
            data_cls = LearnDataWitVocab
        else:
            data_cls = LearnData

        data = data_cls(
            train_df_path, valid_df_path,
            embedder, shuffle=shuffle, device=device, batch_size=batch_size, num_workers=num_workers,
            pad_token=pad_token,
            pad_label=pad_label)
        label_size = len(data.train_ds.label2idx)
        if use_embeds:
            use_embeds = len(data.train_ds.text2idx)
        model = BiLSTMCRF.create(
            label_size=label_size,
            # BiLSTM params
            embedding_size=embedding_size,
            hidden_dim=hidden_dim, rnn_layers=rnn_layers, lstm_dropout=lstm_dropout,
            # CRFDecoder params
            crf_dropout=crf_dropout,
            # Global params
            device=device,
            use_embeds=use_embeds
        )

        optimizer_args = {
            "lr": lr,
            "warmup": warmup,
            "t_total": t_total,
            "schedule": schedule,
            "b1": b1,
            "b2": b2,
            "e": e,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm
        }
        len_dl = len(data.train_dl)
        optimizer_args["t_total"] = if_none(
            optimizer_args["t_total"], epochs * len_dl / update_freq)
        optimizer = BertAdam(model=model, **optimizer_args)
        tb_log = TensorboardLog(tensorboard_dir)
        self.data = data
        self.optimizer = optimizer
        self.tb_log = tb_log
        self.model = model
        self.validate_every = validate_every
        self.checkpoint_dir = checkpoint_dir

    def fit(self, epochs=10):
        logging.info("Start training. Total epochs {}.".format(epochs))
        for epoch in range(epochs):
            self.fit_one_cycle(epoch + 1)

    def fit_one_cycle(self, epoch):
        train_step(self.data.train_dl, self.model, self.optimizer, epoch)
        if epoch % self.validate_every == 0:
            rep = validate_step(self.data.valid_dl, self.model, self.data.train_ds.idx2label)
            print(rep)
        self.save_model(os.path.join(self.checkpoint_dir, f"{epoch}.cpt"))

    def save_model(self, path=None):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path=None):
        self.model.load_state_dict(torch.load(path))
