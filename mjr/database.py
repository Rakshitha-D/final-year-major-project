import psycopg2
import os

def connect_to_database():
    conn = psycopg2.connect(
        dbname="User",
        user="postgres",
        password="postgresql",
        host="localhost",
        port="5432"
    )
    return conn

def get_all_records():
    conn = connect_to_database()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM register")
    records = cursor.fetchall()
    cursor.close()
    conn.close()
    return records

def insert_record(label, place, date, image_path):
    conn = connect_to_database()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(id) FROM register")
    max_id = cursor.fetchone()[0]
    next_id = max_id + 1 if max_id is not None else 1
    cursor.execute("INSERT INTO register (id, label, place, date, url) VALUES (%s, %s, %s, %s, %s)", (next_id, label, place, date, image_path))
    conn.commit()
    cursor.close()
    conn.close()

def update_record(record_id, label, place, date, image_path=None):
    conn = connect_to_database()
    cursor = conn.cursor()
    if image_path:
        cursor.execute("UPDATE register SET label = %s, place = %s, date = %s, url = %s WHERE id = %s", (label, place, date, image_path, record_id))
    else:
        cursor.execute("UPDATE register SET label = %s, place = %s, date = %s WHERE id = %s", (label, place, date, record_id))
    conn.commit()
    cursor.close()
    conn.close()

def delete_record(record_id):
    conn = connect_to_database()
    cursor = conn.cursor()
    cursor.execute("SELECT url FROM register WHERE id = %s", (record_id,))
    image_path = cursor.fetchone()[0]
    if image_path:
        os.remove(image_path)
        os.rmdir(os.path.dirname(image_path))
    cursor.execute("DELETE FROM register WHERE id = %s", (record_id,))
    conn.commit()
    cursor.close()
    conn.close()
