import streamlit as st 
import pandas as pd 

from main import *
from keyllm import * 
from db import connect_db
from ngram import split_into_multiple_gram
from embedding_model import load_embedding_model


def streamlit(): 
    
    st.title("Hybrid search engine")

    st.write("Provide csv and questions")
    # hf_model = load_huggingface_model(model_name="Open-Orca/Mistral-7B-OpenOrca", device_map="auto", max_new_tokens=50)


    # key1 = st.text_input("key1: ", "2019")
    # key2 = st.text_input("Key2: ", "bill to")
    # df = pd.read_csv('test1.csv')

    question = st.text_input("Enter your question here: ", "What is bill to for 2019?")

    file = st.file_uploader("Upload CSV file...", type="csv")



    if file: 
        df = pd.read_csv(file)

        st.write(df)

        if st.button("Save to database"):
            model = load_embedding_model()
            cur, con = connect_db()
            save_csv_to_db(df, cur, model)
            st.write("Successfully saved to datase")
    else:
        st.write('upload csv file')


    if st.button("Extract Data"):
        # option = st.selectbox("Model to extract keywords from questions", ("ngram", "keybert", "keyllm"))
        
        # if option == "ngram":
        #     keywords = split_into_multiple_gram(question)
        # elif option == "keyllm":
        #     keys = keyllm(hf_model, question)
        #     keywords = keys[0]
        # elif option == "keybert":
        #     keys = keybert(question)
        #     keywords = [key[0] for key in keys] 

        print(question)
        keys = keybert(question)
        print("keywords", keys)
        keywords = [key[0] for key in keys] 

        st.write("Extracted keywords")
        st.write(keywords)

        main(df, question, keywords)


# def main(df, key1, key2):
#     index1, index1_count = search(cur, key1)
#     index2, index2_count = search(cur, key2)

#     result = run(df, index1, index1_count, index2, index2_count)
#     print(result)
#     st.write(result)
        
def main(df, question, keys):

    extracted_values = []
    for key in keys:
        extracted_values.append(search(cur, key))

    results = inference(df, extracted_values)

    st.write("Actual Result")

    print(results)
    st.write(results)

    filtered_results = remove_keys_in_question(question, results)

    filtered_results = remove_duplicates(filtered_results)

    top_k_result = extract_top_k_values(filtered_results, top_k=15)

    st.write("Result After Postprocessing")

    print(top_k_result)
    print(top_k_result)
    st.write(top_k_result)


if __name__ == "__main__":
    # df = pd.read_csv('test1.csv')
    # key1 = "2018"
    # key2 = "bill by"
    # index1, index1_count = search(cur, key1)
    # index2, index2_count = search(cur, key2)

    # result = run(df, index1, index1_count, index2, index2_count)
    # print(result)

    streamlit()
    # question = "What is the bill to 2019?"
    # keys = split_into_multiple_gram(question)


    # results = main(df, keys)
