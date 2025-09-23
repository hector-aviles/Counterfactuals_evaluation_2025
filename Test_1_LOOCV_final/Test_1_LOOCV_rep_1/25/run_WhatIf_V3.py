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
def main(input_csv, input_cbn, output_csv, models_subdir, output_found_actions):

    print("Input CSV file:", input_csv, flush=True)
    print("Input PL file:", input_cbn, flush=True)
    print("Output CSV file:", output_csv, flush=True)    
    print("Model subdirectory:", models_subdir, flush=True)        

    # Initialize actions
    actions = []

    # Read the input file and find all action patterns
    try:
        with open(input_cbn, 'r') as file:
            content = file.read()
            import re
            # Find all patterns like action(something) :- 
            found_actions = re.findall(r'action\(([^)]+)\)\s*:-', content)
            # Add new unique actions to the list
            for action in found_actions:
                if action not in actions:
                    actions.append(action)
                    
            with open(output_found_actions, "a") as file:     
                file.write(str(actions)) 
                file.write("\n")    
                   
    except FileNotFoundError:
        print(f"Warning: Could not read file {input_cbn}, using default actions only", flush=True)
             

    # Generate iaction_truth based on the actions list
    iaction_truth = []
    for i in range(len(actions)):
        truth_values = ['False'] * len(actions)
        truth_values[i] = 'True'
        iaction_truth.append(tuple(truth_values))

    # Print the results for verification
    print("Detected actions:", actions, flush=True)
    print("Generated truth tuples:", iaction_truth, flush=True)    
    
    output_time = models_subdir + "/elapsed_time.csv"
    str_tmp = "%%%%" + output_csv + "\n"
    with open(output_time, "w") as file:     
        file.write(str_tmp)    
    
    # Write to file
    with open(output_csv, 'w', newline='') as outfile:
     writer = csv.writer(outfile, delimiter=',')
     writer.writerow(['action','curr_lane','free_E','free_NE',     
     'free_NW','free_SE','free_SW','free_W', 'labeled_lc', 'latent_collision',
     'iaction', 'probability', 'elapsed_time'])
     
    # Load the BN
    program_files = []
    program_str = ""
    program_files.append(input_cbn)
    program = CounterfactualProgram(program_str, program_files)               
    
    # Initialize counterfactual query data
    evidence = {}
    interventions = {}
    queries = []
    action_idx = -1
    
    # Main loop 
    with open (input_csv, mode = 'r', newline='') as file:
         reader = csv.DictReader(file)
 
         # Loop through each row in the CSV
         row_num = 0
         for row in reader:
         
          row_num += 1       
          # Counterfactuals data
          evidence.clear()
          interventions.clear()
          queries.clear()

          # Add evidence 
          try:
            action_idx = actions.index(row['action'])
          except ValueError:
            print("Observed action: ", row['action'], end=" ", flush = True)
            print("for querying not found in the actions list", flush = True)
            continue  
               
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
          try: 
             action_idx = actions.index(iaction)
          except ValueError:
             print("Intervention: ", iaction, end=" ", flush = True)
             print("for querying not found in the actions list", flush = True) 
             # Write a query not actually performed 
             # but assign to the result a probability of 1 
             # to mantain all groups with 6 queries
             with open(output_csv, 'a', newline='') as outfile: 
                 prob = 1.0           
                 elapsed_time = 0.0
                 writer = csv.writer(outfile, delimiter=',')            
                 writer.writerow([row['action'],row['curr_lane'],            
                 row['free_E'],row['free_NE'],row['free_NW'],            
                 row['free_SE'],row['free_SW'],row['free_W'],  
                 row['labeled_lc'],row['latent_collision'],
                 row['iaction'],prob, elapsed_time])
                 continue

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
            
          whatif_call = "WhatIf -q latent_collision" + " -e action\(" + row['action'] + "\)," + 'True' + " -e curr_lane," + row['curr_lane'] + " -e free_E," + row['free_E'] + " -e free_NE," + row['free_NE'] + " -e free_NW," + row['free_NW'] + " -e free_SE," + row['free_SE'] + " -e free_SW," + row['free_SW'] + " -e free_W," + row['free_W'] +  " -e latent_collision," + row['latent_collision'] + " " + iaction_str + " " + input_cbn             
            
          print(whatif_call, flush=True)                
           
          try:            
            # Query the PLTNs              
            output_query = []            
            start_time = time.time()             
            output_query = program.single_query(interventions, evidence, queries, strategy=config.config["knowledge_compiler"])            
            end_time = time.time()                         
            elapsed_time = end_time - start_time            
                           
            prob = float(output_query[0])                  
            out_str = "Row: " + str(row_num) + " Probability: " + str(prob) + " Elapsed time " + str(elapsed_time)            
            print(out_str, flush=True)                        
                                        
            # Write to file            
            with open(output_csv, 'a', newline='') as outfile:            
              writer = csv.writer(outfile, delimiter=',')            
              writer.writerow([row['action'],row['curr_lane'],            
              row['free_E'],row['free_NE'],row['free_NW'],            
              row['free_SE'],row['free_SW'],row['free_W'],
              row['labeled_lc'],row['latent_collision'],
              row['iaction'],prob, elapsed_time])
            
            with open(output_time, "a") as file:            
              string = str(elapsed_time) + "\n"            
              file.write(string)                          
                                                      
          except Exception as inst:            
            print(type(inst))                
            print(inst.args)                 
            print(inst)             
            print("Evidence: ", evidence, flush=True)            
            print("Intervention: ", interventions, flush=True)            
            print("Query: ", queries, flush=True)                 
 
 
if __name__ == "__main__":

    if len(sys.argv) != 6:
        print("Usage: python3 run_WhatIf_v2.py input.csv input.pl output.csv  models_subdir")
        sys.exit(1)
        
    input_csv = sys.argv[1]
    input_cbn = sys.argv[2]    
    output_csv = sys.argv[3]
    models_subdir = sys.argv[4]  
    output_found_actions = sys.argv[5]   

    print("Received paths:", input_csv, input_cbn, output_csv, models_subdir, output_found_actions)
          
    main(input_csv, input_cbn, output_csv, models_subdir, output_found_actions)
    
    



