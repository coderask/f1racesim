import sqlite3
# import pandas as pd
import pickle as pkl
path = 'racesim/input/vse/F1_timingdata_2014_2019.sqlite'

#connect database
conn = sqlite3.connect(path)

query = '''SELECT id,comment
FROM races
'''

read = conn.cursor()
read.execute(query)
used_race_ids_tuple = read.fetchall()
used_race_ids = []

for entry in used_race_ids_tuple:
    if entry[1] == None:
        pass
    else:
        used_race_ids_tuple.remove(entry)
#convert into a list from tuples 
for nextEntry in used_race_ids_tuple:
    used_race_ids.append(nextEntry[0])

with open('racesim/src/ccModelRecreation/used_race_ids.pkl', 'wb') as file:
    pkl.dump(used_race_ids, file)

# print(used_race_ids)
placeholders = ','.join(['?'] * len(used_race_ids))

raceNamesquery = f'''
SELECT season, location
FROM races
WHERE id IN ({placeholders});
'''
read.execute(raceNamesquery, used_race_ids)
raceNames = read.fetchall()


# print(raceNames)

races = []
for entry in raceNames:
    races.append( "racesim/input/parameters/"+ "pars_" + entry[1] + "_" + str(entry[0]) + ".ini")
with open('racesim/src/ccModelRecreation/raceNames.pkl', 'wb') as file:
    pkl.dump(races, file)
# print(races)
conn.close()  
