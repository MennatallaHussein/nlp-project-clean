import pandas as pd
import torch
import re
import huggingface_hub
from datasets import Dataset
import transformers
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
import gc
import os 

def remove_paranthhesis(text):
    result=re.sub(r'\(.*?\)' , '', text )
    return result




class CharacterChatBot():
    def __init__(self,
                model_path,
                data_path=r"C:\Users\menah\OneDrive\المستندات\Desktop\AI\nlp\PROJECT\DATA\naruto.csv",
                huggingface_token=None
                ):
    
        self.model_path=model_path
        self.data_path=data_path
        self.huggingface_token = huggingface_token
        self.base_model_path="meta-llama/Meta-Llama-3-8B-Instruct"
        self.device= 'cuda' if torch.cuda.is_available() else 'cpu'   

        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)


        if os.path.exists(self.model_path) and os.listdir(self.model_path):
            print("Loading existing model from:", self.model_path)
            self.model = self.load_model(self.model_path)

        else:
            print('Model not found')
            train_dataset=self.load_data()

            self.train(self.base_model_path , train_dataset)

            self.model=self.load_model(self.model_path)

            


    def load_model(self, model_path):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            pipeline = transformers.pipeline(
                'text-generation',
                model=model_path,
                model_kwargs={
                    "torch_dtype": torch.float16,
                    "quantization_config": bnb_config,
                }
            )
            return pipeline





    def load_data(self):
        naruto_transcript_df = pd.read_csv(self.data_path)
        naruto_transcript_df = naruto_transcript_df.dropna()
        naruto_transcript_df['line'] = naruto_transcript_df['line'].apply(remove_paranthhesis)
        naruto_transcript_df['num_of_words'] = naruto_transcript_df['line'].str.strip().str.split(' ')
        naruto_transcript_df['num_of_words'] = naruto_transcript_df['num_of_words'].apply(lambda x: len(x))
        naruto_transcript_df['naruto_transcript_flag'] = 0
        naruto_transcript_df.loc[
            (naruto_transcript_df['name'] == 'Naruto') & (naruto_transcript_df['num_of_words'] > 5),
            'naruto_transcript_flag'
        ] = 1

        index_to_take = list(naruto_transcript_df[
            (naruto_transcript_df['naruto_transcript_flag'] == 1) & (naruto_transcript_df.index > 0)
        ].index)

        system_prompt = (
            'You are Naruto from the anime "Naruto". '
            'Your responses should reflect his personality and speech patterns.\n'
        )

        prompts = []
        completions = []

        for i in index_to_take:
            # Previous line → context
            prompt = system_prompt + naruto_transcript_df.iloc[i - 1]['line'] + "\n"
            # Naruto’s line → target
            completion = naruto_transcript_df.iloc[i]['line']

            prompts.append(prompt)
            completions.append(completion)

        df = pd.DataFrame({
            "prompt": prompts,
            "completion": completions
        })
        dataset = Dataset.from_pandas(df)
        return dataset



    def train(self,
              base_model_name_or_path,
              dataset,
              output_dir = "./results",
              per_device_train_batch_size = 1,
              gradient_accumulation_steps = 1,
              optim = "paged_adamw_32bit",
              save_steps = 200,
              logging_steps = 10,
              learning_rate = 2e-4,
              max_grad_norm = 0.3,
              max_steps = 300,
              warmup_ratio = 0.3,
              lr_scheduler_type = "constant",
              ):
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model= AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            quantization_config=bnb_config ,
            trust_remote_code=True
        )

        model.config.use_cache=False

        tokenizer =AutoTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.pad_token=tokenizer.eos_token

        lora_alpha=16
        lora_dropout=0.1
        lora_r=64

     



        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM"
        )

        max_seq_len=512


        training_arguments = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size = per_device_train_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            optim = optim,
            save_steps = save_steps,
            logging_steps = logging_steps,
            learning_rate = learning_rate,
            fp16= True,
            max_grad_norm = max_grad_norm,
            max_steps = max_steps,
            warmup_ratio = warmup_ratio,
            group_by_length = True,
            lr_scheduler_type = lr_scheduler_type,
            report_to = "none",
            dataset_text_field="prompt",
            max_length=512,
        )



    

        trainer = SFTTrainer(
            model = model,
            train_dataset=dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
            args = training_arguments,)

        trainer.train()

        trainer.model.save_pretrained('final_ckpt')
        tokenizer.save_pretrained('final_ckpt')

        del trainer , model
        gc.collect()

        
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path,
                                                          return_dict=True,
                                                          quantization_config=bnb_config,
                                                          torch_dtype = torch.float16,
                                                          device_map = self.device
                                                          )

        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

        model = PeftModel.from_pretrained(base_model,"final_ckpt")
        model.push_to_hub(self.model_path)
        tokenizer.push_to_hub(self.model_path)

        # Flush Memory
        del model, base_model
        gc.collect()

    

    def chat(self, message, history):
        system_prompt = """You are Naruto from the anime "Naruto". 
        Your responses should reflect his personality and speech patterns.\n"""

        # Build conversation history
        conversation = system_prompt
        for mess in history:
            conversation += f"User: {mess[0]}\nNaruto: {mess[1]}\n"

        # Add the latest user message
        conversation += f"User: {message}\nNaruto:"

        terminators = [
            self.model.tokenizer.eos_token_id,
            self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        output= self.model(
            conversation,
            max_length=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

        generated_text = output[0]['generated_text']
        output_message = generated_text[len(conversation):].strip()

        return output_message


















            



