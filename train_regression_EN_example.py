# Based on Pranay fine-tuning Script using huggingface trainer
# Adapted by Aaron for irony starting from 13/12/2022
# further adapted by aaron for ordinal values 25/01/2024
# !/usr/bin/env python
# coding: utf-8
import transformers
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import datasets
from transformers import Trainer, TrainingArguments
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
import torch.nn.functional as F
# this
#from torch.nn.functional import huber_loss
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import DataCollatorWithPadding
from datasets import load_metric
from sklearn.metrics import classification_report
import torch.nn as nn
import os
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import huggingface_hub

os.environ["WANDB_PROJECT"] = "Ordinal-Classification"
# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"] = "true"
# turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"

def huber_loss(y_pred, y_true, delta=1.0):
    error = y_pred - y_true
    abs_error = torch.abs(error)
    quadratic = torch.min(abs_error, delta)
    linear = (abs_error - quadratic)
    return 0.5 * quadratic ** 2 + delta * linear

def Binarize_Label(pred, truelabel, labelcount = 7):
    ### THIS FUNCTION IS NEEDED TO MAP ANY OF THE FEATURES TO BINARY VALUES
    ## this cannot be done at the start because we want to train for regression
    # and evaluate for classification
    # set labelcount equal to global variable value
    if merge_label_count:
        labelcount = merge_label_count

    #account for the predicted label being larger than the allowed max
    if labelcount == 1:
        pass
    elif pred > labelcount-1:
        # if value is more than 1 out of bounds, set to max value of class range
        pred = labelcount-1
    else:
        pred = round(pred)
    if pred < 0:
        pred = 0
    # not needed for true label as it is always a rounded value in range
    truelabel = int(truelabel)
    if labelcount == 7:
        label_tobinary = {0:0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1}
        pred = label_tobinary[pred]
        truelabel = label_tobinary[truelabel]

    elif labelcount == 5:
        #  merge to 5 point scale
        label_tobinary = {0: 0, 1: 0, 2: 0, 3: 1, 4:1}
        pred = label_tobinary[pred]
        truelabel = label_tobinary[truelabel]
    elif labelcount == 3:
        label_tobinary = {0: 0, 1:0, 2: 1}
        pred = label_tobinary[pred]
        truelabel = label_tobinary[truelabel]
    elif labelcount == 2:
        pass
    elif labelcount == 1:
        pred = 1 if pred > 5 else 0
        truelabel  = 1 if truelabel > 5 else 0
    return  pred, truelabel


def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)
    mse = mean_squared_error(labels, logits)
    rmse = root_mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    int_preds = [int(logit) for logit in logits.flatten().tolist()]
    binarylabels = []
    binarypreds = []
    for pred, label in zip(int_preds, labels.flatten().tolist()):
        binary_pred, binary_label = Binarize_Label(pred, label)
        binarylabels.append(binary_label)
        binarypreds.append(binary_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(binarylabels, binarypreds, average='weighted')
    acc = accuracy_score(binarylabels, binarypreds)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, 'f1': f1, 'precision': precision, 'recall': recall, 'accuracy': acc}
 
class RegressionTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").type(torch.cuda.FloatTensor) # BREAKS IF NOT SPECIFICALLY CONVERTED TO FLOAT AT THIS EXACT SPOT
        outputs = model(**inputs)
        logits = outputs.logits.flatten()
        loss = torch.nn.functional.huber_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss


def merge_labelcats(labelstring, labelcount = 7):
    """ Function that maps the 7-label granularity labels to the more coarse label granularities. NOTE: the 7-labels are here 0-6 including 0. NOT 1-7.
    """
    try:
        label = int(labelstring)
        if labelcount == 7:
            #recale values to start with 0 as the first label instead of 1
            label= label
        elif labelcount == 5:
            #  merge to 5 point scale
            labeldict = {0:0, 1:0, 2:1, 3:2, 4:3, 5:4, 6:4}
            label = labeldict[label]
        elif labelcount == 3:
            labeldict = {0:0, 1:0, 2: 0, 3:1, 4:2, 5: 2, 6:2}
            label = labeldict[label]
        elif labelcount == 2:
            labeldict = {0:0, 1:0, 2:0, 3:0, 4:1, 5:1, 6:1}
            label = labeldict[label]
        elif labelcount == 1:
            # rescale to values between 0-100
            label = (label)/6*100
    except:
        label = np.nan
    #return label-1 for python 0 (starting from 0 as first index), makes sense for regression that 0 is no irony
    return label

def main():
    access_token = "hf_tokenplaceholder"
    huggingface_hub.login(token=access_token)

    #task type is regression
    task = "regression"
    learning_rate = 5e-6
    label_nums = [5, 7]
    # training for 5 specific seeds and averaging: 42, 69,7, 13 and 666, 420,
    randomseeds = [42, 69,7, 666, 420, 13]
    language = "EN"
    # models we want to train: 
    # monolingual IT:  Musixmatch/umberto-wikipedia-uncased-v1 ,  idb-ita/gilberto-uncased-from-camembert
    # monolingual NL: DTAI-KULeuven/robbert-2023-dutch-base , DTAI-KULeuven/robbert-2023-dutch-large, GroNLP/bert-base-dutch-cased , microsoft/deberta-v3-large, microsoft/deberta-v3-base
    # monolingual EN: microsoft/deberta-v3-large microsoft/deberta-v3-base , FacebookAI/roberta-large, FacebookAI/roberta-base, google-bert/bert-base-cased, vinai/bertweet-base, cardiffnlp/twitter-roberta-base, cardiffnlp/twitter-roberta-large-2022-154m
    # multilingual models: microsoft/mdeberta-v3-base, FacebookAI/xlm-roberta-large, FacebookAI/xlm-roberta-base,  google-bert/bert-base-multilingual-cased,  cardiffnlp/twitter-xlm-roberta-base, 
    modelnames = [ "microsoft/deberta-v3-base" , "FacebookAI/roberta-base", "google-bert/bert-base-cased", "vinai/bertweet-base", "cardiffnlp/twitter-roberta-base",\
                   "microsoft/mdeberta-v3-base", "FacebookAI/xlm-roberta-base",  "google-bert/bert-base-multilingual-cased", "cardiffnlp/twitter-xlm-roberta-base" ] # commented  
    test_result_path = f'test_results/log_test_results_{task}_{language}.tsv'
    with open(test_result_path, "w") as f:
        f.write(f'modelname\tlanguage\ttask\tgranularity\tseed\tf1\taccuracy\tmae\tr2\tf1_EN\taccuracy_EN\tmae_EN\tr2_EN\tf1_NL\taccuracy_NL\tmae_NL\tr2_NL\tf1_IT\taccuracy_IT\tmae_IT\tr2_IT\n')

    for label_experiment_num in label_nums:
        global merge_label_count
        # set paramaters for saving the model
        merge_label_count = label_experiment_num
        # load dataset
        dataset = load_dataset('csv', data_files={'train': f'data/{language}/train.csv',
                                                'val': f'data/{language}/dev.csv',
                                                "test": f'data/{language}/test.csv'})
        dataset = dataset.map(lambda example: {"labels": merge_labelcats(example['labels'], labelcount=merge_label_count)})
        for randomseed in randomseeds:
            for MODEL_NAME in modelnames:
                modelname_out = MODEL_NAME.split("/")[1]
                transformers.set_seed(randomseed)
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
                columns = ['input_ids', 'attention_mask', 'labels']
                def convert_to_features(example_batch):
                    """ Creates the feature representation (i.e. the encodings for each tweet) """
                    inputs = list(example_batch['text'])
                    features = tokenizer.batch_encode_plus(
                        inputs, padding=True)
                    features["labels"] = [ float(label) for label in example_batch["labels"]]
                    return features

                features = {}

                for phase, phase_dataset in dataset.items():
                    features[phase] = phase_dataset.map(
                        convert_to_features,
                        batched=True,
                        load_from_cache_file=False,
                    )
                    features[phase].set_format(
                        type="torch",
                        columns=columns,
                    )

                data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

                training_args = TrainingArguments(
                    output_dir=f'{modelname_out}_{task}_{merge_label_count}_seed{randomseed}_{language}',  # output directory
                    #report_to='wandb',
                    num_train_epochs=10,  # total number of training epochs
                    learning_rate=5e-6,
                    per_device_train_batch_size=8,  # batch size per device during training
                    per_device_eval_batch_size=8,  # batch size for evaluation
                    warmup_steps=200,  # number of warmup steps for learning rate scheduler
                    weight_decay=0.01,  # strength of weight decay
                    logging_dir='./logs',  # directory for storing logs
                    logging_steps=100,
                    evaluation_strategy='steps',
                    eval_steps=100,
                    load_best_model_at_end=True,
                    hub_private_repo=True,
                    metric_for_best_model="eval_loss",
                    )

                trainer = RegressionTrainer(
                    model=model,  # the instantiated 🤗 Transformers model to be trained
                    args=training_args,  # training arguments, defined above
                    train_dataset=features['train'],  # training dataset
                    eval_dataset=features['val'],
                    data_collator=data_collator,  # evaluation dataset
                    compute_metrics= compute_metrics_for_regression
                )
                # looking into early stopping
                trainer.add_callback(transformers.EarlyStoppingCallback(early_stopping_patience=3))
                trainer.train()
                #trainer.save_model(f"{MODEL_NAME}/regression_{merge_label_count}_{learning_rate}_seed{randomseed}")

                test_results = trainer.evaluate(features['test'])
                print("Full test results:", test_results)
                req_evaluation = ["f1", "accuracy", "mae", "r2"]
                # f.write(f'modelname\tlanguage\ttask\tgranularity\tseed\tf1\taccuracy\tmae\tr2\tf1_EN\taccuracy_EN\tmae_EN\tr2_EN\tf1_NL\taccuracy_NL\tmae_NL\tr2_NL\tf1_IT\taccuracy_IT\tmae_IT\tr2_IT\n')
                test_evalstring = ""
                for evaluation_criterium in req_evaluation:
                    if f"eval_{evaluation_criterium}" in test_results.keys():
                        test_evalstring += f"\t{test_results[f'eval_{evaluation_criterium}']}"
                    else:
                        test_evalstring += f"\tNA"

                for eval_lang in ["EN", "NL", "IT"]:
                    if eval_lang in language or "multi" in language:
                        test_subset = load_dataset('csv', data_files={"test": f'data/{eval_lang}/test.csv'})
                        test_subset = test_subset.map(lambda example: {"labels": merge_labelcats(example['labels'], merge_label_count)})
                        test_subset = test_subset.map(convert_to_features, batched=True, load_from_cache_file=False)
                        test_subset.set_format(type="torch", columns=columns)
                        test_results_subset = trainer.evaluate(test_subset)
                        print("Results for", eval_lang)
                        print(test_results_subset)
                    else:
                        test_results_subset = {"eval_test_f1": "NA", "eval_test_accuracy": "NA", "eval_test_mae": "NA", "eval_test_r2": "NA"}
                    # write out results of model for each language
                    for evaluation_criterium in req_evaluation:
                        test_evalstring += f"\t{test_results_subset[f'eval_test_{evaluation_criterium}']}"
                
                print(test_results)
                # push to private hub
                trainer.push_to_hub(token=access_token)
                test_result_path = f'test_results/log_test_results_{task}_{language}.tsv'
                with open(test_result_path, "a") as f:
                    f.write(f'{modelname_out}\t{language}\t{task}\t{merge_label_count}\t{randomseed}{test_evalstring}\n')
                # clean up checkpoint files
                os.system(f"rm -rf {modelname_out}_{task}_{merge_label_count}_seed{randomseed}_{language}")


if __name__ == "__main__":
    main()
    print("Script Complete")