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
# print(raceNames)
#filter out races into 10% testing 90% training
train =[]
test = []
raceNames_copy = raceNames.copy()
numTest = int((len(raceNames_copy)-1)*0.1)
print(numTest)
for race in raceNames:
    # print(race)
    if 'SaoPaulo_2019' in race:
        test.append(race)
        # raceNames.remove(race)
        raceNames_copy.remove(race)
        # print("entered brazil 2019")
races_toAdd = random.sample(raceNames_copy, numTest-1)
for race in races_toAdd:
    test.append(race)
for race in raceNames_copy:
    if race not in test:
        train.append(race)
# print(len(test)) 
# print(len(raceNames))   
# print(len(train))
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

trainRacesSet1 = []
valRacesSet1 = []
trainRacesSet2 = []
valRacesSet2 = []
trainRacesSet3 = []
valRacesSet3 = []
trainRacesSet4 = []
valRacesSet4 = []
trainRacesSet5 = []
valRacesSet5 = []
trainRacesSet6 = []
valRacesSet6 = []
trainRacesSet7 = []
valRacesSet7 = []
trainRacesSet8 = []
valRacesSet8 = []
trainRacesSet9 = []
valRacesSet9 = []
trainRacesSet10 = []
valRacesSet10 = []
# print("raceNames", raceNames)
indexing = 1 
for train_index, val_index in folds.split(train):
    # trainRaces.append(raceNames[i] for i in train_index)
    # valRaces.append(raceNames[i] for i in val_index)
    # print(raceNames[i] for i in train_index)
    # print(raceNames[i] for i in val_index)
    def switch(indexing):
        if indexing == 1:
            for i in train_index:
                trainRacesSet1.append(train[i])
            for i in val_index:
                valRacesSet1.append(train[i])
            indexing+=1 
            
        elif indexing == 2:
            print("entered set 2")
            for i in train_index:
                trainRacesSet2.append(train[i])
            for i in val_index:
                valRacesSet2.append(train[i])
            indexing+=1 
            
        elif indexing == 3:
            for i in train_index:
                trainRacesSet3.append(train[i])
            for i in val_index:
                valRacesSet3.append(train[i])
            indexing+=1
            
        elif indexing == 4:
            for i in train_index:
                trainRacesSet4.append(train[i])
            for i in val_index:
                valRacesSet4.append(train[i])
            indexing+=1
            
        elif indexing == 5:
            for i in train_index:
                trainRacesSet5.append(train[i])
            for i in val_index:
                valRacesSet5.append(train[i])
            indexing+=1
            
        elif indexing == 6:
            for i in train_index:
                trainRacesSet6.append(train[i])
            for i in val_index:
                valRacesSet6.append(train[i])
            indexing+=1
            
        elif indexing == 7:
            for i in train_index:
                trainRacesSet7.append(train[i])
            for i in val_index:
                valRacesSet7.append(train[i])
            indexing+=1
            
        elif indexing == 8:
            for i in train_index:
                trainRacesSet8.append(train[i])
            for i in val_index:
                valRacesSet8.append(train[i])
            indexing+=1
            
        elif indexing == 9:
            for i in train_index:
                trainRacesSet9.append(train[i])
            for i in val_index:
                valRacesSet9.append(train[i])
            indexing+=1
            
        elif indexing == 10:
            for i in train_index:
                trainRacesSet10.append(train[i])
            for i in val_index:
                valRacesSet10.append(train[i])
            indexing+=1
        return indexing     
    indexing = switch(indexing)
print(len(trainRacesSet1))
print(len(valRacesSet1))
print(len(trainRacesSet2))
print(len(valRacesSet2))
print(len(trainRacesSet3))
print(len(valRacesSet3))
print(len(trainRacesSet4))
print(len(valRacesSet4))
print(len(trainRacesSet5))
print(len(valRacesSet5))
# print("\n")
# print("trainraces", trainRaces, len(trainRaces))
# print("\n")
# print("valraces", valRaces, len(valRaces))
# print(len(raceNames))
    # print("TRAIN:", train_index, "validate:", val_index)
# print(trainRaces, valRaces)
#modify all relavent files to collect data for this split 
# for race in trainRaces:
#     modify_file('main_racesim.py', '259', "race_pars_file_ = " + race.split('racesim/input/parameters/')[-1]  + '\n')
#     modify_file('racesim/src/vse_supervised.py', "47", "path = " + race.split('racesim/input/parameters/')[-1]  + '\n')
#     modify_file('racesim/src/ccModelRecreation/Extract_sql_files.py', "6", "racePath = " + race.split('racesim/input/parameters/')[-1]  + '\n')
#     run_script('main_racesim.py')
#     run_script('Extract_sql_files.py')
#     run_script('racesim/src/ccModelRecreation/vse_supervisedReplica.py')

