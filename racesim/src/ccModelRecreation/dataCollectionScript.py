import fileinput
import subprocess
import pickle as pkl    
with open('racesim/src/ccModelRecreation/used_race_ids.pkl', 'rb') as file:
    used_race_ids = pkl.load(file)
with open('racesim/src/ccModelRecreation/raceNames.pkl', 'rb') as file:
    raceNames = pkl.load(file)
print(raceNames)
#filter out races
test = []
for race in raceNames:
    print(race)
    if 'SaoPaulo_2019' in race:
        test.append(race)
        print("entered brazil 2019")
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

# Modify files
# for race in raceNames:
# # race = raceNames[raceNames.index("racesim/input/parameters/pars_Catalunya_2019.ini")]
#     modify_file('main_racesim.py', '259', "race_pars_file_ = " + race.split('racesim/input/parameters/')[-1]  + '\n')
#     modify_file('racesim/src/vse_supervised.py', "47", "path = " + race.split('racesim/input/parameters/')[-1]  + '\n')
#     modify_file('racesim/src/ccModelRecreation/Extract_sql_files.py', "6", "racePath = " + race.split('racesim/input/parameters/')[-1]  + '\n')
#     run_script('main_racesim.py')
#     run_script('racesim/src/ccModelRecreation/vse_supervisedReplica.py')

