import argparse
from tqdm import tqdm
from io import StringIO


from main import *
from keyllm import * 
from generate_question_from_csv import *

def read_parquet_file(parquet_file_path = "data/sample.parquet"):
    df = pd.read_parquet(parquet_file_path)
    return df

def run_save_to_db(csv_df):
    model = load_embedding_model()
    cur, con = connect_db()
    save_csv_to_db(csv_df, cur, model)
    # print("Successfully saved to database")

def run_main():
    df = read_parquet_file()
    cur, con = connect_db()

    embed_df = pd.DataFrame(columns=["index", "question", "ground_truth", "is_correct", "prediction"])
    total_count = 0 
    correct_count = 0
    for i in tqdm(range(len(df))):
        if df['multiple_tables'][i] == 0:
            csv_strings = df['csv_string_with_header'][i]
            results = generate_question_and_answer_from_csv(csv_strings)
            run_save_to_db_for_csv_without_header(df['csv_string_with_header'][i], cur, model)

            csv_df = pd.read_csv(StringIO(df['csv_string'][i]), header=None)

            for result in results:
                keys = keybert(result["question"])
                keywords = [key[0] for key in keys] 
                top_k_result = main(csv_df, result["question"], keywords)

                # print(top_k_result)
                # print(result["question"])
                # print(result["answer"])

                is_correct = False
                indices = [item['index'] for item in top_k_result]
                for ind, index in enumerate(indices):
                    if index is None or (isinstance(index, float) and index != index): 
                        indices[ind] = ''

                if result['answer'] in indices:
                    correct_count += 1 
                    is_correct = True
                total_count += 1 

                embed_df.at[len(embed_df), "question"] = result['question']
                embed_df.at[len(embed_df)-1, "index"] = df['index'][i]
                embed_df.at[len(embed_df)-1, "ground_truth"] = result['answer']
                embed_df.at[len(embed_df)-1, "prediction"] = top_k_result
                embed_df.at[len(embed_df)-1, "is_correct"] = is_correct

                # embed_df.at[len(embed_df)-1, "csv"] = df['csv_string'][i]

            
            del_query = """delete from embedding_table;"""
            cur.execute(del_query)

        print('acc', correct_count/total_count)
        embed_df.to_csv('final.csv', index=False)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Hybrid search evaluation')

    # parser.add_argument('save_to_db', type=str, help='save csv to database or not')

    # args = parser.parse_args()

    # # Access the arguments
    # print("save_to_db:", args.save_to_db)
    run_main()