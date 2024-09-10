import fileinput
import subprocess

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
modify_file('main_racesim.py', '259', "race_pars_file_ = 'pars_Catalunya_2019.ini'\n")
modify_file('racesim/src/vse_supervised.py', "47", "path = 'racesim/input/parameters/pars_Catalunya_2019.ini'")
modify_file('racesim/src/ccModelRecreation/Extract_sql_files.py', "6", "racePath = 'racesim/input/parameters/pars_Catalunya_2019.ini'")
run_script('main_racesim.py')
run_script('racesim/src/ccModelRecreation/vse_supervisedReplica.py')
# modify_file('script2.py', 'another line to modify', 'new content for line 2\n')

# Run the scripts in a specific order

# run_script('script2.py')