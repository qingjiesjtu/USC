# LLMWrapper.py
from .model.LlamaModel import LlamaModel
from .model.OpenaiModel import OpenaiModel
from .model.Llama2Model import Llama2Model
import os
import re
import string
import time
from openai import OpenAI


class LLMWrapper:
    def __init__(self, model_name, device, precision):
        self.model_name = model_name
        self.device = device
        self.precision = precision
        
        if model_name == "llama2-7b-instruct":
            self.model = Llama2Model(device, precision, model_path='meta-llama/Llama-2-7b-chat-hf')
        elif model_name == "llama2-7b-pretrain":
            self.model = Llama2Model(device, precision, model_path='meta-llama/Llama-2-7b-hf')
        elif model_name == "llama2-13b-instruct":
            self.model = Llama2Model(device, precision, model_path='meta-llama/Llama-2-13b-chat-hf')
        elif model_name == "llama2-13b-pretrain":
            self.model = Llama2Model(device, precision, model_path='meta-llama/Llama-2-13b-hf')
        elif model_name == "llama3-8b-instruct":
            self.model = LlamaModel(device, precision, model_path='meta-llama/Meta-Llama-3-8B-Instruct')
        elif model_name == "llama3-8b-pretrain":
            self.model = LlamaModel(device, precision, model_path='meta-llama/Meta-Llama-3-8B')
        elif model_name == "llama3.1-8b-instruct":
            self.model = LlamaModel(device, precision, model_path='meta-llama/Meta-Llama-3.1-8B-Instruct')
        elif model_name == "gpt3":
            self.model = OpenaiModel(device, precision, model_path="davinci-002")
        elif model_name == "gpt3.5-turbo":
            self.model = OpenaiModel(device, precision, model_path="gpt-3.5-turbo-0125")
        elif model_name == "gpt3.5-turbo-instruct":
            self.model = OpenaiModel(device, precision, model_path="gpt-3.5-turbo-instruct")
        elif model_name == "gpt4-turbo":
            self.model = OpenaiModel(device, precision, model_path="gpt-4-turbo-2024-04-09")
        elif model_name == "gpt4":
            self.model = OpenaiModel(device, precision, model_path="gpt-4-0613")
        elif model_name == "gpt4o":
            self.model = OpenaiModel(device, precision, model_path="gpt-4o-2024-08-06")
        elif model_name == "gpt4o-mini":
            self.model = OpenaiModel(device, precision, model_path="gpt-4o-mini-2024-07-18")
        elif model_name == "o1-preview":
            self.model = OpenaiModel(device, precision, model_path="o1-preview")
        elif model_name == "o1-mini":
            self.model = OpenaiModel(device, precision, model_path="o1-mini")
        # fine tuned oepnai model
        elif model_name == "gpt4o-tuned":
            from api_config import OPENAI_GPT4O_TUNED_MODEL_PATH, OPENAI_GPT4O_TUNED_API_KEY
            self.model = OpenaiModel(device, precision, model_path=OPENAI_GPT4O_TUNED_MODEL_PATH,
                                     openai_api_key=OPENAI_GPT4O_TUNED_API_KEY)
        elif model_name == "gpt3.5-turbo-tuned":
            from api_config import OPENAI_GPT35_TURBO_TUNED_MODEL_PATH, OPENAI_GPT35_TURBO_TUNED_API_KEY
            self.model = OpenaiModel(device, precision, model_path=OPENAI_GPT35_TURBO_TUNED_MODEL_PATH,
                                     openai_api_key=OPENAI_GPT35_TURBO_TUNED_API_KEY)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.api_models = ["claude3-opus", "claude3-sonnet", "claude3-haiku", "gemini1.0-pro", "gemini1.5-pro", "gemini1.5-flash", "gpt3.5-turbo", "gpt3.5-turbo-instruct", "gpt4-turbo", "gpt4", "gpt4o", "gpt4o-mini"]

    def generate(self, prompt, max_new_token=None, not_do_sample=None, top_k=None, top_p=None, temperature=None):
        return self.model.generate(prompt, max_new_token, not_do_sample, top_k, top_p, temperature)
    


    def generate_one_text(self, prompt, not_do_sample=None, top_k=None, top_p=None, temperature=None, return_prob=False):
        retry_times = 0
        max_retries = 5 if self.model in self.api_models else 0
        max_retries = 5 if 'gpt' in self.model_name else 0
        while retry_times <= max_retries:
            try:
                if 'gpt' in self.model_name:
                    if return_prob:
                        print('cannot returning prob for gpt, please forbid return_prob')
                    else:
                        res = self.model.generate(prompt,1, top_p=0,temperature=0)
                        first_word = res.split(' ')[0].strip(string.punctuation)
                        # print(f"first_word: {first_word}")
                        return first_word
                else:
                    print('there')
                    if return_prob:
                        res, prob = self.model.generate_with_probs(prompt, 1, not_do_sample, top_k, top_p, temperature)
                        first_word = res.split(' ')[0].strip(string.punctuation)
                        return first_word, prob
                    else:
                        res = self.model.generate(prompt, 1, not_do_sample, top_k, top_p, temperature)
                        first_word = res.split(' ')[0].strip(string.punctuation)
                        # print(f"first_word: {first_word}")
                        return first_word
                
            except Exception as e:
                print(f"Failed to generate text for prompt: {prompt}")
                print(e)

                if retry_times < max_retries:
                    print(f"Retrying... (Attempt {retry_times + 1} of {max_retries})")
                    retry_times += 1
                    time.sleep(10 * retry_times)
                else:
                    print(f"Max retries ({max_retries}) reached. Skipping this prompt.")
                    return "max_retries"
                
    def generate_one_token_probs(self, messages):
        return self.model.generate_one_token_probs(messages)
    
    def perturbation_based_per_seq(self, messages, target):
        return self.model.perturbation_based_per_seq(messages, target)

