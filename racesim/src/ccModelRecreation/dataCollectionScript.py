import fileinput
import subprocess
import pickle as pkl  
import random  
import sklearn
import numpy
import sklearn.model_selection
with open('racesim/src/ccModelRecreation/used_race_ids.pkl', 'rb') as file:
    used_race_ids = pkl.load(file)
with open('racesim/src/ccModelRecreation/raceNames.pkl', 'rb') as file:
    raceNames = pkl.load(file)
print(raceNames)
#filter out races into 10% testing 90% training
train =[]
test = []
raceNames_copy = raceNames.copy()
numTest = int((len(raceNames_copy)-1)*0.1)
print(numTest)
for race in raceNames:
    print(race)
    if 'SaoPaulo_2019' in race:
        test.append(race)
        # raceNames.remove(race)
        raceNames_copy.remove(race)
        print("entered brazil 2019")
races_toAdd = random.sample(raceNames_copy, numTest-1)
for race in races_toAdd:
    test.append(race)
for race in raceNames_copy:
    if race not in test:
        train.append(race)
print(len(test)) 
print(len(raceNames))   
print(len(train))
# #export training and testing data 
# with open('racesim/src/ccModelRecreation/trainingDATA.pkl', 'wb') as file:
#     pkl.dump(train, file)
# with open('racesim/src/ccModelRecreation/testingDATA.pkl', 'wb') as file:
#     pkl.dump(test, file)
test = numpy.array(test)
train = numpy.array(train)
def modify_file(file_path, target_line, new_content):
    with fileinput.FileInput(file_path, inplace=True) as file:
        for line in file:
            if target_line in line:
                print(new_content, end='')
            else:
                print(line, end='')

def run_script(script_path):
    try:
        subprocess.run(['python', script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")

folds = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=42)

trainRaces = []
valRaces = []
for train_index, val_index in folds.split(train):
    trainRaces.append(train[train_index])
    valRaces.append(train[val_index])
    print("TRAIN:", train_index, "validate:", val_index)
#modify all relavent files to collect data for this split 
# for race in trainRaces:
    modify_file('main_racesim.py', '259', "race_pars_file_ = " + race.split('racesim/input/parameters/')[-1]  + '\n')
    modify_file('racesim/src/vse_supervised.py', "47", "path = " + race.split('racesim/input/parameters/')[-1]  + '\n')
    modify_file('racesim/src/ccModelRecreation/Extract_sql_files.py', "6", "racePath = " + race.split('racesim/input/parameters/')[-1]  + '\n')
    run_script('main_racesim.py')
    run_script('Extract_sql_files.py')
#     run_script('racesim/src/ccModelRecreation/vse_supervisedReplica.py')

