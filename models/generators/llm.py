import random
import os
import json
import gc
import torch
import warnings

from peft import AutoPeftModelForCausalLM, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils import left_pad
from models.generators.generator import Generator
from models.generators.prompt import Prompt

class LLM(Generator):
    def __init__(self,
                 model_name = None,
                 batch_size = 1,
                 max_new_tokens = 1,
                 max_doc_len = 10**10,
                 max_length = None,
                 prompt = None,
                 quantization = None,
                 attn_implementation = "sdpa",
                 local_path = False,
                ):
        Generator.__init__(self,
                           model_name=model_name,
                           batch_size=batch_size,
                           max_new_tokens=max_new_tokens,
                           max_doc_len=max_doc_len,
                           max_length=max_length)
        
        self.quantization = quantization
        if quantization == "no":
            tokenizer_name = self.model_name
            model_class = AutoModelForCausalLM
        else:
            try:
                print(f"Loading quantized model from {model_name}")
                config = PeftConfig.from_pretrained(model_name)
                tokenizer_name = config.base_model_name_or_path
                model_class = AutoPeftModelForCausalLM
            except:
                warnings.warn(f"Quantized model {model_name} not found. Loading unquantized model.")
                tokenizer_name = self.model_name
                model_class = AutoModelForCausalLM
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, clean_up_tokenization_spaces=True)
        
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = (
            self.tokenizer.bos_token
            or self.tokenizer.pad_token
            or self.tokenizer.eos_token
        )

        if quantization == "int8":
            quant_config = BitsAndBytesConfig(
                llm_int8_enable_fp32_cpu_offload=True
            )
            self.model = model_class.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=local_path,
            )
        elif quantization == "int4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype='bfloat16',
            )   

            self.model = model_class.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=local_path,
            )
        else:
            self.model = model_class.from_pretrained(
                self.model_name,
                device_map="auto",
            )
        
        self.model = self.model.bfloat16()
        self.model.eval()
        self.model.config.pretraining_tp = 1 # for cpu offloading
        self.prompt = prompt

    
    def generate(self, instr_tokenized):
        input_ids = instr_tokenized["input_ids"].to(self.model.device)
        attention_mask = instr_tokenized["attention_mask"].to(self.model.device)
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
        )

        prompt_len = instr_tokenized["input_ids"].size(1)
        generated_ids = output_ids[:, prompt_len:]
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return decoded
    
    def __del__(self):
        gc.collect()
        torch.cuda.empty_cache()

    def collate_fn(self, examples, **kwargs):
        ignore_index = -100

        q_ids = [ex["q_id"] for ex in examples]
        input_ids_list = [ex["tokenized_input"]["input_ids"][0] for ex in examples]
        attention_mask_list = [ex["tokenized_input"]["attention_mask"][0] for ex in examples]
        
        label = [ex["label"] if isinstance(ex["label"], str) else ex["label"] for ex in examples]
        query = [ex["query"] for ex in examples]
        ranking_label = [ex["ranking_label"] for ex in examples] if "ranking_label" in examples[0] else [None] * len(examples)
        instr = [ex["formatted_instruction"] for ex in examples]

        max_length = max(len(ids) for ids in input_ids_list)

        input_ids_tensor = torch.stack([left_pad(ids, max_length, self.tokenizer.pad_token_id) for ids in input_ids_list])
        attention_mask_tensor = torch.stack([left_pad(mask, max_length, 0) for mask in attention_mask_list])

        model_input = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
        }

        data_dict = {}
        data_dict.update({
            "model_input": model_input,
            "q_id": q_ids,
            "query": query,
            "instruction": instr,
            "label": label,
            "ranking_label": ranking_label,
        })

        return data_dict
        
        

        