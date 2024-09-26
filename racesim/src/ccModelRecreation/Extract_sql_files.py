import sqlite3
import pandas as pd
import pickle as pkl

path = 'racesim/input/vse/F1_timingdata_2014_2019.sqlite'
racePath = 'racesim/input/parameters/pars_Catalunya_2019.ini'
#connect database
conn = sqlite3.connect(path)

#lapnumber: driver 
lapdict = {}
pitdict = {}
query = "SELECT id, location, season FROM races"

#returns tuple of ('racename', year(int))
def findrace(path_to_race):
   

    newstring = path_to_race.split("pars_")

    #after "pars_"
    target = newstring[1].split(".ini")[0]

    target = target.split("_")
    
    racename = target[0]
    raceYear = target[1]

    return (racename, raceYear)  

# print(findTargetRaceId('racesim/input/parameters/pars_Spa_2018.ini'))

read = conn.cursor()
read.execute(query)
race_ids = read.fetchall()
desir_race_id = 1
for id in race_ids:
    tpl = findrace(racePath)
    if id[1] == tpl[0] and id[2] == int(tpl[1]):
        desir_race_id = id[0]


laptime_query = '''
SELECT * 
FROM laps
WHERE laptime > 200 AND race_id = ''' + str(desir_race_id) + ''' 
ORDER BY driver_id'''
#print(laptime_query)

read.execute(laptime_query)
lapquery = read.fetchall()


pitstop_query =  '''
SELECT * 
FROM laps
WHERE pitstopduration > 50 AND race_id = ''' + str(desir_race_id) + ''' 
ORDER BY driver_id
'''
read.execute(pitstop_query)
pitquery = read.fetchall()

for entry in lapquery:
    lapdict[entry[1]] = entry[3]
for pit in pitquery:
    lapdict[pit[1]] = pit[3]

t10_query = '''
WITH race_info AS (
    SELECT nolaps
    FROM races
    WHERE id = ?
),
final_lap_positions AS (
    SELECT driver_id, position
    FROM laps
    WHERE race_id = ? AND lapno = (SELECT nolaps FROM race_info)
)
SELECT driver_id, position
FROM final_lap_positions
ORDER BY position ASC
LIMIT 10;
'''
read.execute(t10_query, (desir_race_id, desir_race_id))

t10= read.fetchall()
#print(t10)

driver_idx_t10 = []
for entry in t10:
    driver_idx_t10.append(entry[0])

# print(driver_idx_t10)

order_query = '''
SELECT id, initials
FROM drivers;
'''
read.execute(order_query)

sqlOrder = read.fetchall()

# print(sqlOrder)

conn.close()

with open('racesim/src/ccModelRecreation/lapdict.pkl', 'wb') as lapDictFile:
    pkl.dump(lapdict, lapDictFile)
with open('racesim/src/ccModelRecreation/driver_idx_t10.pkl', 'wb') as driverIdxFile:
    pkl.dump(driver_idx_t10, driverIdxFile) 
with open('racesim/src/ccModelRecreation/SqlOrder.pkl', 'wb') as orderFile:
    pkl.dump(sqlOrder, orderFile)
if (lapdict):
    with open('racesim/src/ccModelRecreation/lapquery.pkl', 'wb') as lapFile:
        print("lapfile", lapdict)
        pkl.dump(lapdict, lapFile)
if (pitdict):
    with open('racesim/src/ccModelRecreation/pitquery.pkl', 'wb') as pitFile:
        print("pitfile", pitdict)
        pkl.dump(pitdict, pitFile)