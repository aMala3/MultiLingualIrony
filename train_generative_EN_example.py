from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import torch
import os
import pandas as pd
import re
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, AutoConfig, TrainerCallback
from trl import SFTTrainer
from ast import literal_eval
import datasets
import ast
import bitsandbytes as bnb
import huggingface_hub
import wandb
from statistics import mean
from semscore import ModelPredictionGenerator, EmbeddingModelWrapper



def find_all_linear_names(model): # copied from https://github.com/mzbac/llama2-fine-tune/blob/master/utils.py
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)

def print_trainable_parameters(model): # copied from https://github.com/mzbac/llama2-fine-tune/blob/master/utils.py
  """
  Prints the number of trainable parameters in the model.
  """
  trainable_params = 0
  all_param = 0
  for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()
  print(
      f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
  )

def Get_DesiredOutput(dataframe, prompt, task, tokenizer):
    full_explanations = []

    # get an example for each unique label

    for i, row in dataframe.iterrows():
        unique_labels = dataframe['labels'].unique()
        lable_samples = pd.DataFrame()
        # if len(labels) > 2,  we sample one example for three random labels
        if len(unique_labels) > 2:
            unique_labels = unique_labels[:3]
        for label in unique_labels:
            #select a random text for this label
            sample = dataframe[dataframe['labels'] == label].sample(1)
            lable_samples = lable_samples._append(sample)

        # randomize order of lable_samples
        lable_samples = lable_samples.sample(frac=1).reset_index(drop=True)
        chat = []
        
        if task == "binary":
                systemprompt = "You are specialized in detecting irony and sarcasm in social media. This means that you are not only able to identify verbal irony and sarcasm but also situational irony."
                systemprompt += " The possible labels are: 'This text is genuine and does not contain any irony or sarcasm.' and 'This text contains irony or sarcasm.'"
        elif task == "5-grained":
                systemprompt = "You are specialized in detecting irony and sarcasm in social media. This means that you are not only able to identify verbal irony and sarcasm but also situational irony."
                systemprompt += " The possible labels are: 'This text is likely genuine and does not contain any irony or sarcasm.', 'This text is rather genuine and does not contain any irony or sarcasm.', 'It is not clear whether this text contains irony or sarcasm.', 'This text may contain irony or sarcasm.', 'This text likely contains irony or sarcasm.'"
        elif task == "7-grained":
                systemprompt = "You are specialized in detecting irony and sarcasm in social media. This means that you are not only able to identify verbal irony and sarcasm but also situational irony."
                systemprompt += " The possible labels are: 'This text is definitely genuine and does not contain any irony or sarcasm.', 'This text is probably genuine and does not contain any irony or sarcasm.', 'This text is rather genuine and does not contain any irony or sarcasm.', 'It is not clear whether this text contains irony or sarcasm.', 'This text is rather ironic.', 'This text probably contains irony or sarcasm.', 'This text definitely contains irony or sarcasm.'"
        
        # append the examples for each unique label in the prompt template
        for samplenr, sample in lable_samples.iterrows():
            
            example_input = prompt.replace("{PLACEHOLDER_FOR_INPUTTEXT}" , sample['text'])\
            .replace("{PLACEHOLDER_FOR_LABEL}", sample['labels'])
            user_text = re.search(r'\\begin\[user\](.*?)\\end\[user\]', example_input, re.DOTALL)
            system_text = re.search(r'\\begin\[assistant\](.*?)\\end\[assistant\]', example_input, re.DOTALL)
            if samplenr == 0:
                chat.append({"role": "user", "content": systemprompt + user_text.group(0).replace(r"\begin[user]", "").replace(r"\end[user]", "").replace("  ", " ").capitalize()})
            else:
                chat.append({"role": "user", "content": user_text.group(0).replace(r"\begin[user]", "").replace(r"\end[user]", "").replace("  ", " ").capitalize()})
            chat.append({"role": "assistant", "content": system_text.group(0).replace(r"\begin[assistant]", "").replace(r"\end[assistant]", "").replace("  ", " ").capitalize()},)

        
        # get the actual to classify text
        desired_output = prompt.replace("{PLACEHOLDER_FOR_INPUTTEXT}" , row['text'])\
        .replace("{PLACEHOLDER_FOR_LABEL}", row['labels'])

        # get text between \begin[user] and \end[user]
        user_text = re.search(r'\\begin\[user\](.*?)\\end\[user\]', desired_output, re.DOTALL)
        # get text for system , this part is ignored for testing/inference only used as gold for training
        system_text = re.search(r'\\begin\[assistant\](.*?)\\end\[assistant\]', desired_output, re.DOTALL)
        
       
        chat.append({"role": "user", "content": user_text.group(0).replace(r"\begin[user]", "").replace(r"\end[user]", "").replace("  ", " ").capitalize()})
        chat.append({"role": "assistant", "content": system_text.group(0).replace(r"\begin[assistant]", "").replace(r"\end[assistant]", "").replace("  ", " ").capitalize()},)
        desired_output = tokenizer.apply_chat_template(chat, tokenize=False)
        #print(desired_output)
        full_explanations.append(desired_output)
    dataframe["desired_output"] = full_explanations
    return dataframe

def main():

    language = "EN"
    tasks = ["binary", "5-grained", "7-grained"]
    access_token = "hf_placeholder"
    huggingface_hub.login(token=access_token)
    model_name =  "meta-llama/Meta-Llama-3-8B-Instruct"
    save_as_name = model_name.split("/")[1]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    for task in tasks:
        base_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, token=access_token )
        base_model.config.use_cache = False
        base_model = prepare_model_for_kbit_training(base_model)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # Fix for fp16

        prompt = r"\begin[user] Does this text contain irony or sarcasm?\
            \n### Text: {PLACEHOLDER_FOR_INPUTTEXT}\end[user]\
            \n\begin[assistant]### Label:{PLACEHOLDER_FOR_LABEL}\end[assistant]"

        train_data = pd.read_csv(f'data/{language}/train.csv')

        # if the label is a string, lowercase the label
        
        train_data["labels"] = train_data["labels"].apply(lambda x: x.lower() if type(x) == str else x)

        dev_data = pd.read_csv(f'data/{language}/dev.csv')
        dev_data["labels"] = dev_data["labels"].apply(lambda x: x.lower() if type(x) == str else x)


        if task == "binary":
            # labels 0-3 are considered non-ironic, label 4, 5, 6 is considered ironic
            labeldict = {0: "This text is genuine and does not contain any irony or sarcasm.",\
                         1: "This text is genuine and does not contain any irony or sarcasm.",\
                            2: "This text is genuine and does not contain any irony or sarcasm.",\
                            3: "This text is genuine and does not contain any irony or sarcasm.",\
                            4: "This text contains irony or sarcasm.",\
                            5: "This text contains irony or sarcasm.",\
                            6: "This text contains irony or sarcasm."}

        elif task == "5-grained":
            # map from 7 labels to 5 outputs
            labeldict = {0: "This text is likely genuine and does not contain any irony or sarcasm.",\
                            1: "This text is likely genuine and does not contain any irony or sarcasm.",\
                            2: "This text is rather genuine and does not contain any irony or sarcasm.",\
                            3: "It is not clear whether this text contains irony or sarcasm.",\
                            4: "This text may contain irony or sarcasm.",\
                            5: "This text likely contains irony or sarcasm.",\
                            6: "This text likely contains irony or sarcasm."}
        elif task == "7-grained":
            #These labels range include ``definitely not ironic'' (1), ``probably not ironic'' (2), ``rather not ironic'' (3), ``not sure'' (4), ``rather  ironic'' (5), ``probably ironic''(6) and ``definitely ironic''(7). 
            labeldict = { 0: "This text is definitely genuine and does not contain any irony or sarcasm.",\
                          1: "This text is probably genuine and does not contain any irony or sarcasm.",\
                            2: "This text is rather genuine and does not contain any irony or sarcasm.",\
                            3: "It is not clear whether this text contains irony or sarcasm.",\
                            4: "This text is rather ironic.",\
                            5: "This text probably contains irony or sarcasm.",\
                            6: "This text definitely contains irony or sarcasm."
                                }
        train_data["labels"] = train_data["labels"].map(labeldict)
        dev_data["labels"] = dev_data["labels"].map(labeldict)
            
        train_data = Get_DesiredOutput(dataframe=train_data, prompt=prompt, tokenizer=tokenizer, task=task)
        dev_data = Get_DesiredOutput(dataframe=dev_data, prompt=prompt, tokenizer=tokenizer, task=task)
        trainset = datasets.Dataset.from_pandas(train_data)
        devset = dev_data.sample(200)

        print(train_data["desired_output"].to_list()[0])


        # Set up PEFT LoRA for fine-tuning.
        lora_config = LoraConfig(
            lora_alpha=16,
            r=128,
            target_modules=find_all_linear_names(base_model),
            task_type="CAUSAL_LM",
        )
        #max_seq_length = 1024

        trainer = SFTTrainer(
            model=base_model,
            train_dataset=trainset,
            args=TrainingArguments(
                per_device_train_batch_size=4,  # This is actually the global batch size for SPMD.
                num_train_epochs=5,
                output_dir=f"./output_{save_as_name}_{task}_{language}",
                eval_accumulation_steps=10,
                dataloader_drop_last = True,  # Required for SPMD.
            ),
            peft_config=lora_config,
            dataset_text_field="desired_output"
        )

        class SemscoreEvalCallback(TrainerCallback):
            def on_evaluate(self, args, state, control, model, tokenizer, eval_dataloader, **kwargs):
                generator = ModelPredictionGenerator(model = model, tokenizer = tokenizer)
                eval_ds = devset
                results = generator.run(dataset = eval_ds)

                em = EmbeddingModelWrapper()
                similarities = em.get_similarities(
                    em.get_embeddings( [a["desired_output"] for a in results] ),
                    em.get_embeddings( [a["answer_pred"] for a in results] ),
                )
                cosine_sim = mean(similarities)
                trainer.log({"cosine_sim": cosine_sim})
        trainer.add_callback(SemscoreEvalCallback())
        # add early stopping
        class EarlyStopping_Generator(TrainerCallback):
            def __init__(self):
                self.best_cosine = 0
                self.counter = 0
            def on_evaluate(self, args, state, control, model, tokenizer, eval_dataloader, **kwargs):
                generator = ModelPredictionGenerator(model = model, tokenizer = tokenizer)
                eval_ds = devset
                results = generator.run(dataset = eval_ds)

                em = EmbeddingModelWrapper()
                similarities = em.get_similarities(
                    em.get_embeddings( [a["answer_ref"] for a in results] ),
                    em.get_embeddings( [a["answer_pred"] for a in results] ),
                )
                cosine_sim = mean(similarities)
                if cosine_sim > self.best_cosine:
                    self.best_cosine = cosine_sim
                    self.counter = 0
                else:
                    self.counter += 1
                if self.counter > 2:
                    control.should_stop
        trainer.add_callback(EarlyStopping_Generator())
        trainer.train()
        # save
        trainer.save_model(f"finetuned_models/{save_as_name}_{task}_{language}")

        # delete output files
        os.system(f"rm -r output_{save_as_name}_{task}_{language}")

if __name__ == "__main__":
    main()
    print("Complete")