import bitsandbytes as bnb
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from captum.attr import (
    FeatureAblation,
    ShapleyValues,
    LayerIntegratedGradients,
    LLMAttribution,
    LLMGradientAttribution,
    TextTokenInput,
    TextTemplateInput,
    ProductBaselines,
)

def res_attribution(model, tokenizer, eval_prompt, target):
    inp = TextTokenInput(
        eval_prompt,
        tokenizer,
        skip_tokens=[1],  # skip the special token for the start of the text <s>
    )

    fa = FeatureAblation(model)

    llm_attr = LLMAttribution(fa, tokenizer)

    attr_res = llm_attr.attribute(inp, target=target)

    return attr_res


def pb_based_per_token(model, tokenizer, input_text, target=None, generate=True):
    # This function uses perturbation-based method to calculate the influence of input tokens on output tokens
    '''
    Parameter explanation
    model: Pass in the loaded model, note: not the model path
    tokenizer: Pass in the tokenizer of the loaded model
    input_text: Input text, note that its format should be interpretable by apply_chat_template
    target: Target text
    generate: Whether model generation is needed
    '''
    if generate and target == None:
        # Conduct dialogue and output results
        tokenized_chat = tokenizer.apply_chat_template(input_text, tokenize=True, add_generation_prompt=True,
                                                       return_tensors="pt")
        # Model generation
        outputs = model.generate(
            tokenized_chat,
            max_new_tokens=16,  # Control generation length
            pad_token_id=tokenizer.eos_token_id
        )

        # Convert output tokens to text
        new_chat = tokenizer.decode(outputs[0])

        # Exception handling: Prevent truncated output
        if new_chat[len(new_chat) - 4:] != "</s>":
            new_chat += "</s>"
        matches = re.findall(r'\[\/INST\].*?<\/s>', new_chat)

        # Find the most recent response
        for i, match in enumerate(matches):
            match = match[8:-4]
            matches[i] = match
            generated_response = matches[len(matches) - 1]

        # Plot
        attr_res = res_attribution(model, tokenizer, input_text, generated_response)
        attr_res.plot_token_attr_sorted(cnum=None, show=True)
    else:
        input_text = tokenizer.apply_chat_template(input_text, tokenize=False)
        attr_res = res_attribution(model, tokenizer, input_text, target)
        attr_res.plot_token_attr_sorted(cnum=None, show=True)


def pb_based_per_seq(model, tokenizer, inp, target):
    # This function uses perturbation-based method to calculate the influence of input sentences on output tokens
    # This input is special, inp needs to be designed, inp format is as follows:
    '''
    inp = TextTemplateInput(
        template="<s>[INST]{}[/INST]{}</s><s>[INST]{}[/INST]{}</s><s>[INST]{}[/INST]",
        values=[f"{Question}", "No", "Are you sure about your answer? Answer again","No", "Are you sure about your answer? Answer again"],
        baselines=["","",]
    )
    '''
    '''
    Parameter explanation
    model: Pass in the loaded model, note: not the model path
    tokenizer: Pass in the tokenizer of the loaded model
    inp: Input text
    target: Target text
    '''
    fa = FeatureAblation(model)
    llm_attr = LLMAttribution(fa, tokenizer)

    attr_res = llm_attr.attribute(inp, target=target)
    attr_res.plot_token_attr(show=True)


def ig_based_per_seq(model, tokenizer, input_text, target, layername):
    '''
    Parameter explanation
    model: Pass in the loaded model, note: not the model path
    tokenizer: Pass in the tokenizer of the loaded model 
    inp: Input text
    target: Target text
    layername: Name of the layer to analyze, e.g. for llama2 can analyze layers like model.model.layers[0].self_attn.k_proj
    '''

    lig = LayerIntegratedGradients(model, layername)  # This step is layer-by-layer analysis

    llm_attr = LLMGradientAttribution(lig, tokenizer)

    eval_prompt = tokenizer.apply_chat_template(input_text, tokenize=False)

    inp = TextTokenInput(
        eval_prompt,
        tokenizer,
        skip_tokens=[1],  # skip the special token for the start of the text <s>
    )

    attr_res = llm_attr.attribute_sliced(inp, target=target)

    attr_res.plot_seq_attr_sliced(show=True)


