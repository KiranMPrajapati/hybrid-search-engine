import streamlit as st 
import pandas as pd 
from io import StringIO

from keyllm import * 
from tqdm import tqdm 

import csv
import io


if __name__ == "__main__":
    ans_df = pd.read_csv('final.csv')
    parquet_file_path = "data/sample.parquet"

    df = pd.read_parquet(parquet_file_path)

    # embed_df = pd.DataFrame(columns=["csv", "question", "extracted_keywords", "ground_truth", "prediction"])
    embed_df = pd.DataFrame(columns=["index", "question", "ground_truth", "prediction"])


    for index, value in tqdm(ans_df.iterrows()):
        if value['is_correct'] == False:
            # data = df['csv_string'][value['index']].replace('\n', ',\n')  # Add comma after each newline for CSV parsing

            # # Read data using CSV reader
            # csv_data = csv.reader(io.StringIO(data))

            # # # Print data in readable format
            # csv_str_data = ''
            # import pdb; pdb.set_trace()
            # for row in csv_data:

            #     csv_str_data += ','.join(row) + '\n'

            # keys = keybert(value[1]['question'])
            # embed_df.at[len(embed_df), "extracted_keywords"] = keys
            embed_df.at[len(embed_df), "question"] = value['question']
            embed_df.at[len(embed_df)-1, "ground_truth"] = value['ground_truth']
            embed_df.at[len(embed_df)-1, "index"] = value['index']

            embed_df.at[len(embed_df)-1, "prediction"] = value['prediction']
            # embed_df.at[len(embed_df)-1, "csv"] = csv_str_data
    embed_df.to_csv('streamlit_final.csv', index=False)
    

    st.dataframe(embed_df, hide_index=True)