Welcome to the scheduling code for Rapid Ascent.

There are 4 files that are important to understand this repository

1: data:
within this file is the printers dataset outlined

2: STL files:
within this file is the STL files the scheduling was based off of

3: results:
within this file is all generated csv files, and a script that summarises experiments A and B

4: src:
this contains the code base. strategies.py contains the 4 strategies and their code, run_strategy.py will run the selected strategy

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% How to operate the code / Reproduce the experiments within the report:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Open run_strategy.py and you will see:
strategy = "satiation_aware"
run_id = "run_001"

Change these to the desired strategy and run id, and run the file there is a strategies dictionary as a comment above these variables.
If reproducing the experiments, input the run_id and strategy outlined in the table within the report, and run the file each time you change run_id and strategy.

CSV files will be created and added to the results folder. when you are ready to analyse the results, open analysis.py and run the file. graphs and two more csv's labelled Experiment A and Experiment B summary will be created.

Thank you for taking the time to view my project, I hope it provides you with insight into my code and rapid ascent's scheduling section.
