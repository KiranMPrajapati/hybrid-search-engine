import streamlit as st 


from db import *
from embedding_model import * 
from collections import Counter


cur, conn = connect_db()


model = load_embedding_model()

def remove_duplicates_and_count(input_list):
    unique_dicts = []
    index_counts = {}
    for d in input_list:
        index = d['row_col_index']
        if index not in index_counts:
            index_counts[index] = 1
            unique_dicts.append(d)
        else:
            index_counts[index] += 1
    return unique_dicts, index_counts

def search(cur, key):
    exact_similarity = search_exact_value(cur, target_exact_string=key)
    embedding_similarity = retrieve_top_k_embeddings(cur, embeddings=encode_text(model,key))
    array_similarity = search_particular_value_from_array_column(cur, target_string=key)
    distance_similarity = search_particular_value(cur, target_string=key)
    exact_partial_similarity = search_partial_value_from_text_column(cur, target_string=key)
    total_similar_rows = exact_similarity + embedding_similarity + array_similarity + distance_similarity + exact_partial_similarity

    
    # row_col_index = []
    # for rows in total_similar_rows:
    #     row_col_index.append((rows[4], rows[5]))

    extracted_info = []
    for rows in total_similar_rows:
        extracted_info.append({"row_col_index": (rows[4], rows[5]), "is_header": rows[6]})

    unique_index, unique_index_count = remove_duplicates_and_count(extracted_info)
    # unique_index_count = Counter(row_col_index)

    transformed_list = []

    for item in unique_index:
        index = item['row_col_index']
        count = unique_index_count[index]
        transformed_item = {'unique_index': index, 'is_header': item['is_header'], 'unique_index_count': count}
        transformed_list.append(transformed_item)


    return transformed_list

def run_for_two_keys(df, index1, index1_count, index2, index2_count):
    result = []
    for r1, c1 in index1:
        for r2, c2 in index2:
            result.append({"index": df.iloc[r2-1, c1],"score": (index1_count[(r1, c1)] + index2_count[r2, c2])/2})
    return result


def extract_row_values_exclude_column(df, row_index):
    return df.loc[row_index].to_list()[1:]


def inference(df, index_info):

    header_list = []
    not_header_list = []

    for sublist in index_info:
        for item in sublist:
            if item['is_header']:
                header_list.append(item)
            else:
                not_header_list.append(item)

    result = []


    if len(header_list) != 0: 
        for i in range(len(header_list)):
            for j in range(len(not_header_list)):
                value1_col = header_list[i]["unique_index"][1]
                value2_row = not_header_list[j]["unique_index"][0]
                result.append({"index": df.iloc[value2_row-1, value1_col],
                            "score": (header_list[i]["unique_index_count"] + not_header_list[j]["unique_index_count"])/2})
    else:
        for j in range(len(not_header_list)):
            value2_row = not_header_list[j]["unique_index"][0]
            extracted_row = extract_row_values_exclude_column(df, value2_row)
            result.append({"index": extracted_row,
                        "score": not_header_list[j]["unique_index_count"]})

    return result

def inference_multi_header(df, index_info):

    header_list = []
    not_header_list = []

    for sublist in index_info:
        for item in sublist:
            if item['is_header']:
                header_list.append(item)
            else:
                not_header_list.append(item)

    result = []


    if len(header_list) != 0: 
        for i in range(len(header_list)):
            for j in range(len(not_header_list)):
                value1_col = header_list[i]["unique_index"][1]
                value2_row = not_header_list[j]["unique_index"][0]
                result.append({"index": df.iloc[value2_row, value1_col],
                            "score": (header_list[i]["unique_index_count"] + not_header_list[j]["unique_index_count"])/2})
    else:
        for j in range(len(not_header_list)):
            value2_row = not_header_list[j]["unique_index"][0]
            extracted_row = extract_row_values_exclude_column(df, value2_row)
            result.append({"index": extracted_row,
                        "score": not_header_list[j]["unique_index_count"]})

    return result

def extract_top_k_values(results, top_k=5):
    # Sort the data based on the 'score' values in descending order
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

    # Extract the top 5 values
    top_5_values = sorted_results[:top_k]

    return top_5_values

def remove_duplicates(results):
    # Keep track of seen 'index' values
    seen_indexes = set()

    # List to store filtered dictionaries
    filtered_results = []

    for item in results:
        index = item['index']
        if not isinstance(index, list):
            if index not in seen_indexes:
                seen_indexes.add(index)
                filtered_results.append(item)
        else:
            filtered_results.append(item)

    return filtered_results


def remove_keys_in_question(question, result):
    # Split the question into words
    words_in_question = question.split()

    # Remove dictionaries from the list where any word in 'index' matches any word in the question
    filtered_result = [item for item in result if not any(word in item['index'] for word in words_in_question)]

    return filtered_result


def main(df, question, keys):

    extracted_values = []
    for key in keys:
        extracted_values.append(search(cur, key))

    results = inference_multi_header(df, extracted_values)

    # print("Actual Result")

    # print(results)

    # results = remove_keys_in_question(question, results)

    results = remove_duplicates(results)

    top_k_result = extract_top_k_values(results, top_k=15)

    # print("Result After Postprocessing")

    # print(top_k_result)

    return top_k_result


if __name__ == "__main__":
    df = pd.read_csv('test1.csv')
    key1 = "2018"
    key2 = "bill by"
    index1, index1_count = search(cur, key1)
    index2, index2_count = search(cur, key2)

    result = inference(df, index1, index1_count, index2, index2_count)
    print(result)