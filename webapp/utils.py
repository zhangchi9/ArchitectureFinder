import config as creds
import psycopg2

class_index = {0: 'Achaemenid architecture',
 1: 'American Foursquare architecture',
 2: 'American craftsman style',
 3: 'Ancient Egyptian architecture',
 4: 'Art Deco architecture',
 5: 'Art Nouveau architecture',
 6: 'Baroque architecture',
 7: 'Bauhaus architecture',
 8: 'Beaux-Arts architecture',
 9: 'Byzantine architecture',
 10: 'Chicago school architecture',
 11: 'Colonial architecture',
 12: 'Deconstructivism',
 13: 'Edwardian architecture',
 14: 'Georgian architecture',
 15: 'Gothic architecture',
 16: 'Greek Revival architecture',
 17: 'International style',
 18: 'Novelty architecture',
 19: 'Palladian architecture',
 20: 'Postmodern architecture',
 21: 'Queen Anne architecture',
 22: 'Romanesque architecture',
 23: 'Russian Revival architecture',
 24: 'Tudor Revival architecture'}

def connect():
    
    # Set up a connection to the postgres server.
    conn_string = "host="+ creds.PGHOST +" port="+ "5432" +" dbname="+ creds.PGDATABASE +" user=" + creds.PGUSER \
                  +" password="+ creds.PGPASSWORD
    
    conn = psycopg2.connect(conn_string)
    print("Connected to AWS DataBase!")

    # Create a cursor object
    cursor = conn.cursor()
    
    return conn, cursor