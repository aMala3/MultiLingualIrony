
import transformers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import datasets
from transformers import Trainer, TrainingArguments
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import DataCollatorWithPadding
from datasets import load_metric
from sklearn.metrics import classification_report
import torch.nn as nn
import os

import huggingface_hub


# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"] = "Ordinal-Classification"
# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"] = "false"
# turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def Binarize_Label(pred, truelabel, labelcount =7):
    if our_label_count:
        labelcount = our_label_count

    truelabel_tobinary = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1}
    if labelcount == 7:
        label_tobinary = {0:0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6:1}
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
    return  pred, truelabel


def compute_metrics_for_regression(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)
    #print(labels.shape)
    mse = mean_squared_error(labels, preds)
    rmse = root_mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels,preds)

    binarylabels = []
    binarypreds = []
    for pred, label in zip(preds, labels):
        binary_pred, binary_label = Binarize_Label(pred, label)
        binarylabels.append(binary_label)
        binarypreds.append(binary_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(binarylabels, binarypreds, average='weighted')
    acc = accuracy_score(binarylabels, binarypreds)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, 'f1': f1,'precision': precision, 'recall': recall, 'accuracy': acc}


def merge_labelcats(labelstring, labelcount = 7):
    """ Function to make numerical labels out of the probability annotations
    includes the option to merge the most extreme non-ironic and ironic borderlines.
    merge extremes --> merges the labels at the extreme edges of the annotations scheme
    'yes' merges the 2 extremes,
    1 = not ironic (1-2) in 7-point scale
    2 = rather not ironic with doubt (3 in 7-point scale)
    3 = neutral (4 in 7-point scale)
    4 = rather ironic with doubt ( 5 in 7-point scale)
    5 = ironic (6-7 in 7-point scale)

    'extra' merges the three extremes
    1 = ironic (1-2-3 in 7-point scale)
    2 = doubt --> (only 4 in 7-point scale)
    3 = not ironic ( 5-6-7 in 7-point scale)

    'safe' merges the 2 outside labels and extend the borderline doubt
    1 = not ironic (1-2 in 7-point scale)
    2 = doubt (3-4-5 in 7-point scale)
    3 = ironic (6-7 in 7-point scale)
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

class OrdinalTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        # THIS LOSS FUNCTION IS BASED ON ORDINAL LOG LOSS 2 from https://github.com/glanceable-io/ordinal-log-loss/blob/main/src/loss_functions.py
        ### THIS CAN BE HARDCODED
        num_classes = our_label_count
        #manually setting distance matrix for now
        dist_matrix = OUR_DISTANCE_MATRIX
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[float(dist_matrix[true_labels[j][i]][label_ids[j][i]]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances, device ="cuda:0",requires_grad=True)
        err = -torch.log(1-probas)*abs(distances_tensor)**1.5
        loss = torch.sum(err, axis=1).mean()
        #print(type(loss), loss.dtype, loss)

        return (loss, outputs) if return_outputs else loss


def main():
    access_token = "hf_tokenplaceholder"
    huggingface_hub.login(token=access_token)
    # 
    label_nums = [5, 7]
    randomseeds = [42, 69,7, 666, 420, 13]
    # currently missing microsoft/deberta-v3-large due to some config issues
    modelnames = ["microsoft/deberta-v3-base" , "FacebookAI/roberta-base", "google-bert/bert-base-cased", "vinai/bertweet-base", "cardiffnlp/twitter-roberta-base",\
                   "microsoft/mdeberta-v3-base", "FacebookAI/xlm-roberta-base",  "google-bert/bert-base-multilingual-cased",  "cardiffnlp/twitter-xlm-roberta-base"]
    modelnames = ["vinai/bertweet-base"]
    task = "ordinal"
    learningrate = 5e-6
    language = "EN"
    test_result_path = f'test_results/log_test_results_{task}_{language}.tsv'
    with open(test_result_path, "w") as f:
        f.write(f'modelname\tlanguage\ttask\tgranularity\tseed\tf1\taccuracy\tmae\tr2\tf1_EN\taccuracy_EN\tmae_EN\tr2_EN\tf1_NL\taccuracy_NL\tmae_NL\tr2_NL\tf1_IT\taccuracy_IT\tmae_IT\tr2_IT\n')

    for label_experiment_num in label_nums:
        # get label names
        global our_label_count
        # set paramaters for saving the model
        our_label_count = label_experiment_num
        for randomseed in randomseeds:
            for modelname in modelnames:
                print(f"Model: {modelname}, Seed:", randomseed, "Labelcount:", our_label_count)
                MODEL_NAME = modelname
                modelname_out = modelname.split("/")[1]
                # load dataset
                dataset = load_dataset('csv', data_files={'train': f'data/{language}/train.csv',
                                                        'val': f'data/{language}/dev.csv',
                                                        "test": f'data/{language}/test.csv'})
                
                transformers.set_seed(randomseed)
                print("Number of labels:", our_label_count)
                # distance metric 
                global OUR_DISTANCE_MATRIX
                # labelcount can stay the same because python range 0-7 excludes 7, labels are from 0 up to and including only 6
                OUR_DISTANCE_MATRIX = np.absolute(np.arange(0, our_label_count)[..., np.newaxis] - np.arange(0, our_label_count)[np.newaxis, ...])
                print(OUR_DISTANCE_MATRIX)


                # return label-1 because python indexing and label conventions start with label 0 >> already implemented in label mapping function
                dataset = dataset.map(lambda example: {"labels": merge_labelcats(example['labels'], our_label_count)})
                
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=our_label_count)
                    
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                model = model.to(device)

                columns = ['input_ids', 'attention_mask', 'labels']
                def convert_to_features(example_batch):
                    """ Creates the feature representation (i.e. the encodings for each tweet) """
                    inputs = list(example_batch['text'])
                    features = tokenizer.batch_encode_plus(
                        inputs, padding=True)
                    features["labels"] = example_batch["labels"]
                    return features
                features = {}

                for phase, phase_dataset in dataset.items():
                    features[phase] = phase_dataset.map(
                        convert_to_features,
                        batched=True,
                        load_from_cache_file=False,
                    )
                    print(phase, len(phase_dataset), len(features[phase]))
                    features[phase].set_format(
                        type="torch",
                        columns=columns,
                    )
                    print(phase, len(phase_dataset), len(features[phase]))

                data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

                training_args = TrainingArguments(
                    output_dir=f'{modelname_out}_{task}_{our_label_count}_seed{randomseed}_{language}',  # output directory
                    #report_to='wandb',
                    num_train_epochs=10,  # total number of training epochs
                    learning_rate=learningrate,
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
                    #metric_for_best_model="eval_loss"\
                    )

                if our_label_count == 2:
                    trainer = Trainer(
                        model=model,  # the instantiated 🤗 Transformers model to be trained
                        args=training_args,  # training arguments, defined above
                        train_dataset=features['train'],  # training dataset
                        eval_dataset=features['val'],
                        data_collator=data_collator,  # evaluation dataset
                        compute_metrics=compute_metrics
                    )
                else:
                    trainer = OrdinalTrainer(
                        model=model,  # the instantiated 🤗 Transformers model to be trained
                        args=training_args,  # training arguments, defined above
                        train_dataset=features['train'],  # training dataset
                        eval_dataset=features['val'],
                        data_collator=data_collator,  # evaluation dataset
                        compute_metrics=compute_metrics_for_regression,
                    )
                # looking into early stopping
                trainer.add_callback(transformers.EarlyStoppingCallback(early_stopping_patience=3))
                trainer.train()
                trainer.save_model(f"finetuned_models/{modelname_out}_{task}_{our_label_count}_seed{randomseed}_{language}")

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
                    if eval_lang in language:
                        test_subset = load_dataset('csv', data_files={"test": f'data/{eval_lang}/test.csv'})
                        test_subset = test_subset.map(lambda example: {"labels": merge_labelcats(example['labels'], our_label_count)})
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
                    f.write(f'{modelname_out}\t{language}\t{task}\t{our_label_count}\t{randomseed}{test_evalstring}\n')
                # clean up checkpoint files
                os.system(f"rm -rf {modelname_out}_{task}_{our_label_count}_seed{randomseed}_{language}")
                


if __name__ == "__main__":
    main()
    print("Script Complete")
