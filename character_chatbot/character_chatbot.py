

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import huggingface_hub


class CharacterChatBot:
    def __init__(self,
                 adapter_model_path="mennaGHANAM/Naruto-Llama-3-8B",
                 base_model_path="meta-llama/Meta-Llama-3-8B-Instruct",
                 huggingface_token=None,
                 use_quantization=False):

        self.adapter_model_path = adapter_model_path
        self.base_model_path = base_model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_quantization = use_quantization

        if huggingface_token:
            huggingface_hub.login(huggingface_token)

        print(f"Loading {base_model_path} + adapter {adapter_model_path}")
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)

        if self.use_quantization:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )

        # ✅ Attach your trained Naruto adapter
        model = PeftModel.from_pretrained(base_model, self.adapter_model_path)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        return pipe, tokenizer

    def chat(self, message, history):
        system_prompt = (
            "You are Naruto from the anime 'Naruto'. "
            "Your responses should reflect his personality and speech patterns.\n"
        )

        conversation = system_prompt
        for user_msg, bot_msg in history:
            conversation += f"User: {user_msg}\nNaruto: {bot_msg}\n"
        conversation += f"User: {message}\nNaruto:"

        output = self.model(
            conversation,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        generated_text = output[0]['generated_text']
        reply = generated_text[len(conversation):].strip()
        return reply









            



