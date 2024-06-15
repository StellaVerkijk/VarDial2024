"""
Code to fine-tune different (L)LMs on event detection as described in Verkijk & Vossen, 2023 on an HPC cluster using Lightning.
Code written by Sophie Arnoult for NER
Adapted to event detection by Stella Verkijk
"""

import torch
import torchmetrics
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, BertTokenizerFast, RobertaTokenizerFast
import pickle
import json
import os
import evaluate
import numpy as np
import zipfile as z
import lightning as L
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.cli import LightningCLI
from datasets import Dataset, DatasetDict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")

# set your parameters for this script
modelpath = "emanjavacas/GysBERT-v2" # should correspond to glb_events.yaml
path_to_tagset = 'globalise/tagsets/glb_events.json' #should correspond to .sh file, glb_events.yaml
tokzr = BertTokenizerFast

"""
tokenizer settings: should be changed where relevant
when using GysBERT: BertTokenizerFast.from_pretrained(pretrained_model, model_max_length = 512) 
when using GysBERT-v2: BertTokenizerFast.from_pretrained(pretrained_model)
when using XLM-R: AutoTokenizer.from_pretrained(pretrained model)
when using RobBERT: RobertaTokenizerFast.from_pretrained(pretrained_model, add_prefix_space=True, model_max_length = 512)
"""


def create_dataset(datadir, tagset, words_or_tokens, predict_data):
    """Create HuggingFace dataset from jsonl data, converting ner labels to their tagset index.

    predict_data: 'validation', 'test', or path to zip file of jsons"""
    datasets = {}
    for datasplit in ['train', 'dev', 'test']: #predefined files
        with open(os.path.join(datadir, datasplit + '.json'), 'r') as f: #with .json ending
            dic = {"ids": [], "tokens": [], "event_tags": []}
            extract_data(tagset, words_or_tokens, f, dic)
        if datasplit == 'dev':
            datasplit = 'validation'
        datasets[datasplit] = Dataset.from_dict(dic)

    if predict_data == 'validation' or predict_data == 'test': #setting
        datasets['predict'] = datasets[predict_data]
    else:  # get prediction data from zip
        zip_file = z.ZipFile(os.path.join(datadir, predict_data))
        members = sorted(zip_file.namelist())
        with zip_file as zf:
            dic = {"ids": [], "tokens": [], "event_tags": []}
            for member in members:
                with zip_file.open(member) as f:
                    extract_data(tagset, words_or_tokens, f, dic)
            datasets['predict'] = Dataset.from_dict(dic)

    return DatasetDict(datasets)


def extract_data(tagset, words_or_tokens, f, dic, id_pfx=''):
    for id, line in enumerate(f.readlines()):
        instance = json.loads(line)
        dic["ids"].append(f"{id_pfx}{id}")
        dic["tokens"].append(instance[words_or_tokens])
        dic["event_tags"].append([tagset[x] for x in instance["events"]])


class NerDataset(Dataset):
    def __init__(self, dataset_dict, partition_key="train"):
        self.partition = dataset_dict[partition_key]

    def __getitem__(self, index):
        return self.partition[index]

    def __len__(self):
        return self.partition.num_rows


def map_begin_to_mid_label(tagset):
    """map indices of B labels to I labels"""
    vals = list(tagset.values())
    for k, v in tagset.items():
        if k.startswith("B-") and k.replace("B-", "I-") in tagset:
            vals[v] = tagset[k.replace("B-", "I-")]
    return vals


class NerDataModule(L.LightningDataModule):
    def __init__(self, data_dir="data", batch_size=32, num_workers=1, pretrained_model=modelpath,
                 words_or_tokens="words", tagset_path=path_to_tagset, predict_data='validation'):
        super().__init__()
        with open(tagset_path) as f:
            self.tagset = json.load(f)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_pkl = "data.pkl"
        self.num_labels = len(self.tagset)
        self.b2i = map_begin_to_mid_label(self.tagset)
        self.words_or_tokens = words_or_tokens
        self.predict_data = predict_data
        self.align_tokens = token_aligner(tokzr.from_pretrained(pretrained_model), self.b2i)

    def prepare_data(self):
        # download
        print(self.data_dir)
        dataset = create_dataset(self.data_dir, self.tagset, self.words_or_tokens, predict_data=self.predict_data)
        # tokenize
        tokenized = {}
        for split in dataset:
            tokenized[split] = dataset[split].map(self.align_tokens, batched=True, batch_size=None)
        # save tokenized dataset
        with open(os.path.join(self.data_dir, self.data_pkl), 'wb') as f:
            pickle.dump(tokenized, f)

    def setup(self, stage: str):
        with open(os.path.join(self.data_dir, self.data_pkl), 'rb') as f:
            tokenized = pickle.load(f)

            # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            tokenized['train'].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            tokenized['train'].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            self.gmevent_train = NerDataset(tokenized)
            self.gmevent_val = NerDataset(tokenized, 'train')

        # Assign test dataset for use in dataloader(s)
        # In the current setting test and predict are the same
        if stage == "test":
            tokenized['test'].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            self.gmevent_test = NerDataset(tokenized, 'test')

        if stage == "predict":
            tokenized['predict'].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            self.gmevent_predict = NerDataset(tokenized, 'predict')

    def train_dataloader(self):  # add num_workers
        return DataLoader(self.gmevent_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.gmevent_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.gmevent_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.gmevent_predict, batch_size=self.batch_size, num_workers=self.num_workers)

def token_aligner(tokenizer, b2i):
    def align_tokens(instances):
        """split tokens with tokenizer and assign label to each.

        the O label is 0, B labels are uneven and I labels are the following even figure. Non initial subtokens receive the same label
        as the initial subtoken except if that is a B label (convert to I label)"""
        label_batches = instances["event_tags"]
        subtokens = tokenizer.batch_encode_plus(
            instances["tokens"], pad_to_max_length=True, truncation=True, is_split_into_words=True
        )
        aligned_labels_batches = []
        for i, labels in enumerate(label_batches):
            aligned_labels = []
            word_id = None
            for wid in subtokens.word_ids(batch_index=i):
                if wid is None:
                    aligned_labels.append(-100)
                elif wid != word_id:
                    aligned_labels.append(labels[wid])
                    word_id = wid
                else:
                    aligned_labels.append(b2i[aligned_labels[-1]])
            aligned_labels_batches.append(aligned_labels)
        subtokens["labels"] = aligned_labels_batches
        return subtokens

    return align_tokens

def define_model(num_labels, freeze_model_params, pretrained_model):
    model = AutoModelForTokenClassification.from_pretrained(pretrained_model, num_labels=num_labels,
                                                            ignore_mismatched_sizes=True)
    if freeze_model_params:
        for param in model.parameters():
            param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model


class NERModel(L.LightningModule):
    def __init__(self, learning_rate=5e-5, tagset_path=path_to_tagset, pretrained_model=modelpath,
                 freeze_model_params=True):
        super().__init__()
        with open(tagset_path) as f:
            self.tagset = json.load(f)
        self.num_labels = len(self.tagset)
        self.learning_rate = learning_rate
        self.tokenizer = tokzr.from_pretrained(pretrained_model)
        self.model = define_model(self.num_labels, freeze_model_params, pretrained_model)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_labels,
                                           ignore_index=self.tagset["O"])
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_labels,
                                            ignore_index=self.tagset["O"])
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                       labels=batch["labels"])
        self.log("train_loss", outputs["loss"])
        return outputs["loss"]  # this is passed to the optimizer for training

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                       labels=batch["labels"])
        self.log("val_loss", outputs["loss"], prog_bar=True)

        logits = outputs["logits"]
        predicted_labels = torch.argmax(logits, 2)
        self.val_f1(predicted_labels, batch["labels"])
        self.log("val_f1", self.val_f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                       labels=batch["labels"])

        logits = outputs["logits"]
        predicted_labels = torch.argmax(logits, 2)
        self.test_f1(predicted_labels, batch["labels"])
        self.log("test F1", self.test_f1, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                       labels=batch["labels"])
        logits = outputs["logits"]
        predicted_labels = torch.argmax(logits, 2)
        score = self.test_f1(predicted_labels, batch["labels"])
        print(score)
        return predicted_labels, batch["labels"], batch["input_ids"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class SeqevalCb(Callback):
    def __init__(self):
        super().__init__()

    def on_predict_end(self, trainer, pl_module):
        preds, tokens, subpreds, subrefs = compute_seqeval(pl_module.tokenizer, pl_module.tagset,
                                                       predictions_path=os.path.join(trainer.default_root_dir,
                                                                                     "predictions.pt"))
        with open(os.path.join(trainer.default_root_dir, "predictions.json"), 'w', encoding='utf8') as f:
            for line_preds, line_subtokens, sp, sr in zip(preds, tokens, subpreds, subrefs):
                jsondict = {"subtokens": line_subtokens, "ner": line_preds, "preds": sp, "ref": sr}
                json.dump(jsondict, f, ensure_ascii=False)
                f.write("\n")


class CustomWriter(BasePredictionWriter):
    def __init__(self, write_interval):
        super().__init__(write_interval)

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        torch.save(prediction, os.path.join(trainer.default_root_dir, dataloader_idx, f"{batch_idx}.pt"))

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        print(f"Saving predictions to {trainer.default_root_dir}")
        torch.save(predictions, os.path.join(trainer.default_root_dir, "predictions.pt"))
        print(os.path.join(trainer.default_root_dir, "predictions.pt"))


def ner_labels(tagset, preds, refs):
    tags = list(tagset.keys())

    def ner_label(i):
        if i == -1:
            return 'X'
        return tags[i]

    ner_vec = np.vectorize(ner_label)
    str_preds = ner_vec(torch.where(refs != -100, preds, -1).numpy())
    str_preds = [list(filter(lambda x: x != 'X', seq)) for seq in list(str_preds)]
    str_refs = ner_vec(torch.where(refs != -100, refs, -1).numpy())
    str_refs = [list(filter(lambda x: x != 'X', seq)) for seq in list(str_refs)]
    return str_preds, str_refs
def compute_seqeval(tokenizer, tagset, predictions=None, predictions_path=None):
    """predictions are sequences of tuples of batched predictions and references

    get stats on subtoken/token ratio
    """
    if predictions is None:
        predictions = torch.load(predictions_path)
    seqeval = evaluate.load('seqeval')
    preds, references, subtokens, subpreds, subrefs = [], [], [], [], []

    def interpolate(token_starts, subtoken_predictions):
        token_preds = []
        for (tstarts, st_preds) in zip(token_starts, subtoken_predictions):
            token_preds.append([label for (tstart, label) in zip(tstarts, st_preds) if tstart])
        return token_preds

    for batch_preds, batch_refs, batch_input_ids in predictions:
        batch_subtokens = [tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True) for input_ids in
                           batch_input_ids]
        # map token starts to True
        batch_token_starts = [list(map(lambda st: st.startswith("#"), subtokens)) for subtokens in batch_subtokens] # ▁ for roberta models, # for bert models, Ġ for robbert-snli
        # get predicted subtoken labels and reference token labels
        str_subtoken_preds, str_refs = ner_labels(tagset, batch_preds, batch_refs)
        # interpolate subtoken predictions and token starts
        token_preds = interpolate(batch_token_starts, str_subtoken_preds)
        token_refs = interpolate(batch_token_starts, str_refs)
        subtokens.extend(batch_subtokens)
        preds.extend(token_preds)
        references.extend(token_refs)
        subpreds.extend(str_subtoken_preds)
        subrefs.extend(str_refs)

    score = seqeval.compute(predictions=preds, references=references)
    print(score)
    subscore = seqeval.compute(predictions=subpreds, references=subrefs)
    print("subtoken score")
    print(subscore)
    return preds, subtokens, subpreds, subrefs
class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.init_args.tagset_path", "model.init_args.tagset_path")
        parser.link_arguments("data.init_args.pretrained_model", "model.init_args.pretrained_model")
def cli_main():
    cli = MyLightningCLI(save_config_kwargs={"overwrite": "true"})


if __name__ == "__main__":
    cli_main()