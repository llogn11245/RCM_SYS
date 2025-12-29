import sqlite3
import pandas as pd
import os 

def upload_data(file_path, db_path):
    if not os.path.exists(file_path):
        print("File does not exist.")
        return
    conn = sqlite3.connect(db_path)
    df = pd.read_csv(file_path)
    df.to_sql('data_table', conn, if_exists='replace', index=False)
    conn.close()


if __name__ == "__main__":
    upload_data('./data/hotel_with_id.csv', './database.db')