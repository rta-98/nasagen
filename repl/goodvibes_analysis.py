from pathlib import Path 
import goodvibes as gv
import os
import pandas as pd

#|%%--%%| <HrkrefHnDD|lOP23PBo07>
os.chdir('/home/yang/projects/nasagen/') 
base = Path.cwd()
storage = base / './storage'
log = storage / './log'
dat = storage / './dat'
csv = storage / './csv'
txt = storage / './txt'

#|%%--%%| <lOP23PBo07|ZZ1AYp3a0E>
nasa7_202_df = pd.read_csv( csv / './nasa7_202_clean.csv' )   
names_202 = list(nasa7_202_df['Molecule'])

i = 0
seen = set() 
for nlog in log.iterdir():
    nlog_stem = Path(nlog).stem
    seen.add(nlog_stem)
    for n202 in names_202: 
        if n202.lower() == nlog_stem: 
            i += 1 

#|%%--%%| <ZZ1AYp3a0E|PPQ6Dm78jp>
# grabbing all frequency calculation log files via search term "opt freq" 
i = 0
file_set = set() 
for file in log.iterdir():
    with open(file, 'r') as log_file:
        for line in log_file:
            if "opt freq" in line:
                file_set.add(Path(file).stem)

file_list = list(file_set)

#|%%--%%| <PPQ6Dm78jp|yMpfwBGdC7>
# grabbing all 291 log files 
log_files_291 = set() 
for file in log.iterdir():
    log_files_291.add(Path(file).stem)

#|%%--%%| <yMpfwBGdC7|hb4TimfGH3>
# determining from these 200 frequency calculation output files the chk files 
# that were generated simulatenously
chk_files = []
for file_name in file_list:
    for file_path in log.iterdir():
        file = Path(file_path).stem
        if file_name == file:
            with open(file_path, 'r') as log_file:
                for line in log_file: 
                    if "%chk" in line:
                        chk_file = line.split("=")[1] 
                        chk_files.append(Path(chk_file).stem)

#|%%--%%| <hb4TimfGH3|6zC6JENG58>
# creating a list out of the 2164 chk files that I've amalgamated from cclear 
chk_file_cclear = txt / "file_list.txt"
cclr_chks = set()
with open(chk_file_cclear, "r") as txt_file:
    lines = txt_file.read().splitlines()
    for line in lines:
        cclr_chks.add(Path(line).stem)

cclr_chks_list = list(cclr_chks)
#|%%--%%| <6zC6JENG58|J5BomXIclK>
# determining, from all 290 log files, if there's a correlated chk file 
i = 0
all_chk_file_matches = set() 
for log_file in log_files_291:
    for chk_file in cclr_chks_list:
        if log_file.lower() == chk_file.lower():
            i += 1 # 74 out of the 2164 chk files match
            all_chk_file_matches.add(chk_file.lower())

#|%%--%%| <J5BomXIclK|I3xt5z4gdN>
# determining out of these amalgamated 2164 files which ones correlate with the chk files produced from the log files 
i = 0
matched_log_chks = set()
for cclear_chk in cclr_chks_list:
    for log_chk in chk_files:
        if cclear_chk.lower() == log_chk.lower():
            matched_log_chks.add(log_chk.lower())
            i += 1 # 41 out of the 2164 chk files match 
            
matched_log_chks_list = list(matched_log_chks)
left_over = [cf for cf in chk_files if cf not in matched_log_chks] # 164 unmatched 
#|%%--%%| <i3xt5z4gdn|z1JjcuiI39>
# is there overlap between the 72 and 41?
for file in all_chk_file_matches: 
    matched_log_chks.add(file)

#|%%--%%| <z1JjcuiI39|tUIFSdoEol>
files_119 = txt / 'file_list_119.txt'

chks_119 = set() 
with open(files_119, 'r') as f:
    lines = f.read().splitlines() 
    for line in lines:
        chks_119.add(Path(line).stem)

for file in matched_log_chks:
    chks_119.add(file)

chks_167 = chks_119


chks_167
#|%%--%%| <tUIFSdoEol|PbxkMuLE1y>
files_155 = txt / 'file_list_155.txt'
chks_155 = set() 
with open(files_119, 'r') as f:
    lines = f.read().splitlines() 
    for line in lines:
        chks_155.add(Path(line).stem)


for file in chks_167:
    chks_155.add(file)

len(chks_155)

