from transformers import AutoTokenizer, pipeline
from transformers import AutoModelForCausalLM


from keybert.llm import TextGeneration
from keybert import KeyBERT
from keybert import KeyLLM


def load_huggingface_model(model_name="Open-Orca/Mistral-7B-OpenOrca", device_map="auto", max_new_tokens=50):
    # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Pipeline
    generator = pipeline(
        model=model, 
        tokenizer=tokenizer,
        task='text-generation',
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.1
    )

    return generator


def keyllm(generator, doc):
    # Load it in KeyLLM
    llm = TextGeneration(generator)
    kw_model = KeyLLM(llm)

    keywords = kw_model.extract_keywords(doc)
    return keywords

def keyllm_hf(doc, model_name="Open-Orca/Mistral-7B-OpenOrca", device_map="auto", max_new_tokens=50):
    generator = load_huggingface_model(model_name, device_map, max_new_tokens)

    keywords = keyllm(generator, doc)

    return keywords

def keybert(doc):
    kw_model = KeyBERT()

    keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)

    return keywords


