from transformers import AutoTokenizer, AutoModelForCausalLM
from InstructorEmbedding import INSTRUCTOR
from io import StringIO

import pandas as pd
import numpy as np
import wordsegment
import torch
import spacy
import csv


from db import connect_db, insert_into_db


# Initialize the wordsegment library
wordsegment.load()
# Load the English language model in spaCy
nlp = spacy.load("en_core_web_sm")


def load_model():
    model_id = 'RUCKBReasoning/TableLLM-7b'
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

    return model, tokenizer

def load_embedding_model():
    model = INSTRUCTOR('hkunlp/instructor-large')
    return model


def encode_text(model, text):
    embeddings = model.encode([text])
    return embeddings


def convert_to_embeddings(text, model, tokenizer):
    tokens = tokenizer.tokenize(text)
    token_id = tokenizer.convert_tokens_to_ids(tokens)
    embeddings = model.model.embed_tokens(torch.tensor(token_id).to('cuda'))
    embeddings = embeddings.to('cpu').detach().numpy()
    return np.mean(embeddings, axis=0)


def split_words(word="examples word"):
    # Tokenize a word into subwords
    subwords = wordsegment.segment(word)
    return subwords

def tokenize_to_lemmas(word):
    doc = nlp(word)
    lemmas = [token.lemma_ for token in doc]
    return lemmas

def generate_keywords(word="examples word"):
    words = split_words(word)
    all_words = []
    for word in words:
        lemmas = tokenize_to_lemmas(word)
        all_words.append(lemmas)

    all_words += words
    keywords = [item if isinstance(item, str) else sublist for sublist in all_words for item in (sublist if isinstance(sublist, list) else [sublist])]
    return list(set(keywords))



def save_csv_to_db(df, cur, model):
    embed_df = pd.DataFrame(columns=["text", "row_index", "col_index", "embeddings", "is_header"])

    for col_index, word_header in enumerate(df.columns):
        embedding = encode_text(model, word_header)
        embed_df.at[len(embed_df), "embeddings"] = embedding
        embed_df.at[len(embed_df)-1, "text"] = word_header
        embed_df.at[len(embed_df)-1, "row_index"] = 0
        embed_df.at[len(embed_df)-1, "col_index"] = col_index
        embed_df.at[len(embed_df)-1, "is_header"] = True
        keywords = generate_keywords(word_header)
        insert_into_db(cur, word_header, keywords, embedding, 0, col_index, True)


    for c, col in enumerate(df.columns):
        for r, value in enumerate(df[col].astype(str)):
            words = value.split(',')
            for k, word in enumerate(words):
                embedding = encode_text(model, word)
                embed_df.at[len(embed_df), "embeddings"] = embedding
                embed_df.at[len(embed_df)-1, "text"] = word
                embed_df.at[len(embed_df)-1, "row_index"] = r + 1
                embed_df.at[len(embed_df)-1, "col_index"] = c
                embed_df.at[len(embed_df)-1, "is_header"] = False
                keywords = generate_keywords(word)
                insert_into_db(cur, word, keywords, embedding, r+1, c, False)

    embed_df.to_csv('test_df.csv', index=False)

def run_save_to_db_for_csv_without_header(csv_data, cur, model):
    lines = csv_data.strip().split('\n')

    reader = csv.reader(lines)

    for row_index, row in enumerate(reader):
        is_header = eval(row[-2])
        for col_index, word in enumerate(row[:-3]):
            embedding = encode_text(model, word)
            keywords = generate_keywords(word)
            if is_header == 1:
                insert_into_db(cur, word, keywords, embedding, row_index, col_index, True)
            else:
                insert_into_db(cur, word, keywords, embedding, row_index, col_index, False)


if __name__ == "__main__":
    # model, tokenizer = load_model()
    model = load_embedding_model()

    df = pd.read_csv('test1.csv')

    cur, con = connect_db()
    save_csv_to_db(df, cur, model)