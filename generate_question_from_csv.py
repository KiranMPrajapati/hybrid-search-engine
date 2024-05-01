import csv
import pandas as pd



def generate_question_and_answer_from_csv(csv_data_with_header):
    lines = csv_data_with_header.strip().split('\n')

    reader = csv.reader(lines)

    header_data = []
    non_header_data = [] 
    for row in reader:
        if eval(row[-2]) == 1:
            header_data.append(row[:-3])
        else:
            non_header_data.append(row)

    final_header_data = []
    for sublist in zip(*header_data):
        combined = ' '.join(item.strip() for item in sublist if item.strip() or item == '')
        final_header_data.append(combined)

    results = [] 
    for value in non_header_data:
        for index_header, each_header in enumerate(final_header_data[1:]):
            question = f"How much is {value[0]} for {each_header}?"
            answer = value[index_header+1]
            # print('question', question, 'answer', answer)
            # print('*********')
            results.append({"question": question, "answer": answer})
    return results  


if __name__ == "__main__":
    parquet_file_path = "data/sample.parquet"

    df = pd.read_parquet(parquet_file_path)

    results = generate_question_and_answer_from_csv(df['csv_string_with_header'][0])
    print(results)