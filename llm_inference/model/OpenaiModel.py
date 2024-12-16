# OpenaiModel.py
from .BasicModel import BasicModel
from ..api_config import OPENAI_API_KEY, OPENAI_ENDPOINT
import requests
import json
from openai import OpenAI
import httpx
import torch
import tiktoken
import numpy as np
from captum.attr import LLMAttributionResult

class OpenaiModel(BasicModel):
    
    def __init__(self, device, precision, model_path, openai_api_key=None):
        self.device = device # api call, not used
        self.precision = precision # api call, not used
        self.model_path = model_path
        self.model = model_path
        self.gen_kwargs = {}
        if openai_api_key == None:
            openai_api_key = OPENAI_API_KEY
            base_url = OPENAI_ENDPOINT
            self.client = OpenAI(
                api_key=openai_api_key,
                base_url=base_url
            )
        elif openai_api_key != None:
            self.client = OpenAI(
                api_key=openai_api_key,
            )
        else:
            raise ValueError("openai_api_key and base_url must be both None or both not None")
        


    def generate(self, messages, max_new_tokens=None, not_do_sample = False ,top_k=None, top_p=None, temperature=None):
        self.gen_kwargs["model"] = self.model
        if max_new_tokens:
            self.gen_kwargs["max_tokens"] = max_new_tokens

        if top_p:
            self.gen_kwargs["top_p"] = top_p
            
        if temperature:
            self.gen_kwargs["temperature"] = temperature
        if not_do_sample:
            self.gen_kwargs["temperature"] = 0
        
        # o1-preview and o1-mini not support max_tokens and temperature
        if self.model_path == "o1-preview" or self.model_path == "o1-mini":
            self.gen_kwargs.pop("max_tokens")
            self.gen_kwargs.pop("temperature")
        if self.model == "gpt-3.5-turbo-instruct":
            response = self.client.completions.create(
                prompt = messages[0]['content'],
                **self.gen_kwargs
            )
            return response.choices[0].text
        else:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                **self.gen_kwargs
            )
            response = chat_completion.choices[0].message.content
        return response

    def perturbation_based_per_seq(self, messages, target):
        restoken, _=self.featureAbalationChat(messages,self.model_path,target=target)
        return restoken.seq_attr.cpu().tolist()
        


    def get_logprob_chat(self, input_messages, model_name: str, output_token: str):
        '''
        chat model can only see the effect of a single output token
        '''
        res = [] 
        max_retries = 5 
        retries = 0
        while retries < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=input_messages,
                    max_completion_tokens=1,
                    logprobs=True,
                    top_logprobs=20
                    )
                if response.choices[0].logprobs.content[0].top_logprobs: 
                    logprob_dict={}
                    for logprobInstance in response.choices[0].logprobs.content[0].top_logprobs:
                        logprob_dict[logprobInstance.token]=logprobInstance.logprob
                    if output_token in logprob_dict:
                        print(f"Yes! target token '{output_token}' is in top20: {logprob_dict}")
                        res.append(logprob_dict[output_token])
                    else: # when target token is not in top5 token, use a value empirically, can be optimized by the probability distribution of the output token of the large language model
                        print(f"Oh no! target token '{output_token}' is not in top20: {logprob_dict}")
                        res.append(min(logprob_dict.values())*2)
                    break 
                else:
                    raise ValueError("logprobs.top_logprobs[0] is None")

            except (AttributeError, IndexError, TypeError, ValueError) as e:
                print(f"Error encountered: {e}, retrying... ({retries + 1}/{max_retries})")
                retries += 1 
        if retries == max_retries:
            print("Max retries reached, using default value.")
            raise ValueError("Max retries reached, using default value.")

        return np.array(res)
    def featureAbalationChat(self,input_messages, model_name: str, target=None, base_token='*'):
        '''
        target is the output token
        the first output is token-wise visualization
        the second output is segment-wise visualization averaged by token attribution within segment
        '''
        enc = tiktoken.encoding_for_model(model_name)
        input_segment_length = len(input_messages)
        input_token_id_list_list = [enc.encode(item['content']) for item in input_messages]#list of list of id
        input_token_list_list = []#list of list of token
        input_token_list = []
        for token_id_list_partial in input_token_id_list_list:
            token_list_partial=[enc.decode([token_id]) for token_id in token_id_list_partial]
            input_token_list_list.append(token_list_partial)
            input_token_list+=token_list_partial
        input_token_length = len(input_token_list)
        if target is None:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=input_messages,
                max_completion_tokens=1
            )
            target = response.choices[0].message.content
        attr = np.zeros((1,input_token_length)) # a 2D array
        ref = self.get_logprob_chat(input_messages, model_name, target)
        cnt=0
        for i in range(input_segment_length):
            modified_messages=[dict(item) for item in input_messages]
            input_token_list_partial=input_token_list_list[i]
            for j in range(len(input_token_list_partial)):# j is the index of token to be ablated
                temp_prompt=''
                for k in range(len(input_token_list_partial)): 
                    if j==k: temp_prompt+=base_token
                    else: temp_prompt+=input_token_list_partial[k]
                modified_messages[i]['content']=temp_prompt
                attr[:,cnt]=ref-self.get_logprob_chat(modified_messages, model_name, target)
                cnt+=1
                # print(f"{attr[:,i]}--{np.exp(attr[:,i])}")
                # the output value is too large, use softmax to compress it, see the last line
            seq_attr=[]
        seqs_len = [len(enc.encode(input_messages[j]['content'])) for j in range(len(input_messages))]
        # iterate over each element in tokens_len, sum by group
        print('attr',attr)
        index = 0
        for seq_len in seqs_len:
            group = attr.tolist()[0][index:index + seq_len]  # get the elements of the current group
            seq_attr.append(sum(group))  # sum the current group
            index += seq_len  # update the index
        seq_attr = torch.tensor(seq_attr)

        print('seq_attr',seq_attr)
        attr_segment = np.zeros((1,input_segment_length))
        start=0
        for i in range(input_segment_length):
            end=start+len(input_token_list_list[i])
            attr_segment[0,i]=np.average(attr[0,start:end])
            start=end
        seq_attr_segment=np.sum(attr_segment,axis=0)
        return LLMAttributionResult(torch.tensor(seq_attr), torch.tensor(attr),input_token_list,[target]),LLMAttributionResult(torch.tensor(seq_attr_segment),torch.tensor(attr_segment),[item['content'] for item in input_messages],[target])
