#!/usr/bin/env python3

import sys
from re import search
import csv
from counterfactuals.counterfactualprogram import CounterfactualProgram
import aspmc.config as config
from aspmc.main import logger as aspmc_logger
import itertools
import time


# Main execution
def main(input_csv, input_cbn, output_csv):

    print("Input CSV file:", input_csv, flush=True)
    print("Input PL file:", input_cbn, flush=True)
    print("Output CSV file:", output_csv, flush=True)    
    
    actions = ['cruise', 'keep', 'change_to_left', 'change_to_right', 'swerve_left', 'swerve_right']
    iaction_truth = [('True','False','False','False','False','False'), ('False','True','False','False','False','False'),('False','False','True','False','False','False'), ('False','False','False','True','False','False'),('False','False','False','False', 'True','False'),('False','False','False','False', 'False','True')]
    # Do not use the following instruction (kept as a reference only)   
    #iaction_truth = list(set(list(itertools.permutations(['True', 'False', 'False', 'False']))))
    
    # Write to file
    with open(output_csv, 'w', newline='') as outfile:
     writer = csv.writer(outfile, delimiter=',')
     writer.writerow(['action','curr_lane','free_E','free_NE',     
     'free_NW','free_SE','free_SW','free_W', 'latent_collision',
     'iaction', 'probability', 'elapsed_time'])
    
    with open (input_csv, mode = 'r', newline='') as file:
         reader = csv.DictReader(file)

          # Counterfactuals data
         program_files = []
         program_str = ""
         evidence = {}
         interventions = {}
         queries = []
 
         # Loop through each row in the CSV
         row_num = 1
         for row in reader:       
          # Counterfactuals data
          program_files.clear()
          program_str = ""
          evidence.clear()
          interventions.clear()
          queries.clear()

          # Add evidence 
          name = 'action(' + row['action'] + ')' 
          value = 'True'
          phase = False if value == "True" else True
          evidence[name] = phase
     
          name = 'curr_lane'
          value = row['curr_lane']
          phase = False if value == "True" else True
          evidence[name] = phase
     
          name = 'free_E'
          value = row['free_E']
          phase = False if value == "True" else True
          evidence[name] = phase     
          
          name = 'free_NE'
          value = row['free_NE']
          phase = False if value == "True" else True
          evidence[name] = phase          
          
          name = 'free_NW'
          value = row['free_NW']
          phase = False if value == "True" else True
          evidence[name] = phase

          name = 'free_SE'
          value = row['free_SE']
          phase = False if value == "True" else True
          evidence[name] = phase

          name = 'free_SW'
          value = row['free_SW']
          phase = False if value == "True" else True
          evidence[name] = phase

          name = 'free_W'
          value = row['free_W']
          phase = False if value == "True" else True
          evidence[name] = phase
          
          name = 'latent_collision'
          value = row['latent_collision']
          phase = False if value == "True" else True
          evidence[name] = phase

          # Add interventions 
          iaction = row['iaction']
          iaction_str = ""
          action_idx = actions.index(iaction)
          for i in range(len(actions)):
            name = 'action(' + actions[i] + ')'
            value = iaction_truth[action_idx][i]
            phase = False if value == "True" else True
            interventions[name] = phase
            iaction_str = iaction_str + " -i " + 'action\(' + actions[i] + '\)' + "," + value 
          
          # Config
          config.config["knowledge_compiler"] = "sharpsat-td"
          aspmc_logger.setLevel("ERROR")
          
          # Add query
          queries.append("latent_collision")

          # Load the BN
          try:
            start_time = time.time() 
            program_files.append(input_cbn)
            program = CounterfactualProgram(program_str, program_files)
            # Do the query  
            output_query = []
            output_query = program.single_query(interventions, evidence, queries, strategy=config.config["knowledge_compiler"])
            end_time = time.time()             
            elapsed_time = end_time - start_time 
          except Exception as inst:
            print(type(inst))    
            print(inst.args)     
            print(inst) 
            print("Evidence: ", evidence, flush=True)
            print("Intervention: ", interventions, flush=True)
            print("Query: ", queries, flush=True)
 
          prob = float(output_query[0])      
          out_str = "Row: " + str(row_num) + " Probability: " + str(prob) + " Elapsed time " + str(elapsed_time)
          print(out_str, flush=True)
                    
          # Write to file
          with open(output_csv, 'a', newline='') as outfile:
           writer = csv.writer(outfile, delimiter=',')
           writer.writerow([row['action'],row['curr_lane'],
           row['free_E'],row['free_NE'],row['free_NW'],
           row['free_SE'],row['free_SW'],row['free_W'],
           row['latent_collision'],row['iaction'],prob, elapsed_time])
           
           whatif_call = "WhatIf -q latent_collision" + " -e action\(" + row['action'] + "\)," + 'True' + " -e curr_lane," + row['curr_lane'] + " -e free_E," + row['free_E'] + " -e free_NE," + row['free_NE'] + " -e free_NW," + row['free_NW'] + " -e free_SE," + row['free_SE'] + " -e free_SW," + row['free_SW'] + " -e free_W," + row['free_W'] +  " -e latent_collision," + row['latent_collision'] + " " + iaction_str + " " + input_cbn 

           print(whatif_call, flush=True)
           
           row_num += 1
 
if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: python3 run_WhatIf_v2.py input.csv input.pl output.csv")
        sys.exit(1)
        
    input_csv = sys.argv[1]
    input_cbn = sys.argv[2]    
    output_csv = sys.argv[3]  

    print("Received paths:", input_csv, input_cbn, output_csv)
          
    main(input_csv, input_cbn, output_csv)
    
    



