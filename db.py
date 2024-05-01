import psycopg2 
import numpy as np   
from pgvector.psycopg2 import register_vector


def connect_db():
    # Connect to the PostgreSQL database
    conn = psycopg2.connect(
        host='localhost',
        database='vectordb',
        user='psql_user',
        password='psql_password'
    )

    # Create a cursor object
    cur = conn.cursor()

    # Roll back the current transaction
    conn.rollback()

    # Start a new transaction
    conn.set_session(autocommit=True)
    try:
        cur.execute('CREATE EXTENSION if not exists vector')
        cur.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm')


        register_vector(conn)
        # Execute the SQL command to create the table
        cur.execute("""
            CREATE TABLE if not exists embedding_table (
                id SERIAL PRIMARY KEY,
                exact_string TEXT,
                edit_distance_string TEXT[],
                embedding vector(768),
                row_index INTEGER,
                col_index INTEGER,
                is_header BOOL
            )
        """)

        # Commit the transaction
        conn.commit()
    except psycopg2.Error as e:
        print("Error:", e)

    return cur, conn

def insert_into_db(cur, exact_string, edit_distance_string, embedding_values, row_index, col_index, is_header):
    if len(edit_distance_string) == 0: 
        edit_distance_string.append("NULL")
    exact_string = exact_string.replace("'", '')
    query = f"""
        INSERT INTO embedding_table (exact_string, edit_distance_string, embedding, row_index, col_index, is_header)
        VALUES ('{exact_string}', ARRAY{edit_distance_string}, '{embedding_values[0].tolist()}', {row_index}, {col_index}, {is_header});
    """

    cur.execute(query)

def search_exact_value(cur, column_name="exact_string", target_exact_string="total payment"):
    # Search for rows where any element in the exact_string array matches a specific value
    cur.execute(f"""SELECT * FROM embedding_table WHERE %s = {column_name};""", 
                (target_exact_string,))

    # Fetch and print the results
    rows = cur.fetchall()
    return rows

def retrieve_top_k_embeddings(cur, embeddings=np.random.rand(1, 768), top_k=3):
    cur.execute(f'SELECT * FROM embedding_table ORDER BY embedding <=> %s LIMIT {top_k}', (embeddings[0],))
    rows = cur.fetchall()
    return rows


def search_particular_value_from_array_column(cur, column_name="edit_distance_string", target_string='tota'):
    query = f"""SELECT * FROM embedding_table 
        WHERE EXISTS (
            SELECT 1
            FROM unnest({column_name}) AS keyword
            WHERE keyword LIKE '{target_string}%' OR
            keyword LIKE '%{target_string}%' OR
            keyword LIKE '%{target_string}'  
        );"""

    cur.execute(query)

    # Fetch and print the results
    rows = cur.fetchall()
    return rows

def search_partial_value_from_text_column(cur, column_name="exact_string", target_string='tota'):
    query = f"""SELECT * FROM embedding_table WHERE '{column_name}' LIKE '%{target_string}%';"""

    cur.execute(query)

    # Fetch and print the results
    rows = cur.fetchall()
    return rows

def search_particular_value(cur, column_name="exact_string", target_string='total as'):
    query = f"SELECT * FROM embedding_table WHERE similarity({column_name}, '{target_string}') > 0.5;"

    cur.execute(query)

    # Fetch and print the results
    rows = cur.fetchall()
    return rows


if __name__ == "__main__":
    cur, con = connect_db()

    exact_string = 'hello'
    edit_distance_string = ['helo', 'helllo']
    embedding_values = np.random.rand(1,768)  # Example embedding values from 1 to 4096
    row_index = 1
    col_index = 1
    is_header = False


    # insert_into_db(cur, exact_string, edit_distance_string, embedding_values, row_index, col_index, is_header)
    # rows = search_exact_string(cur)
    # for row in rows:
    #     print(row)
    # print('I am here')

    rows = retrieve_top_k_embeddings(cur)
    for row in rows:
        print(row)