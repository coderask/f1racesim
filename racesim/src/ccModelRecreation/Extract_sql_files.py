import sqlite3
import pandas as pd


path = 'racesim/input/vse/F1_timingdata_2014_2019.sqlite'

#connect database
conn = sqlite3.connect(path)


query = "SELECT id FROM races"


read = conn.cursor()
read.execute(query)
race_ids = read.fetchall()
for id in race_ids:
    print(id)


conn.close()
