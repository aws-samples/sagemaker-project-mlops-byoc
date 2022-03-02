import json
import random
import ast
import sys
import subprocess
import os

import base64

subprocess.run("pip install numpy", shell=True)
import numpy as np

subprocess.run("echo $dataset_source", shell=True)

def write_to_file(log, filename):
    with open(f"/opt/ml/processing/output/{filename}.log", "w") as f:
        f.write(log + '\n')

def preprocess_handler(inference_record):
    #*********************
    # a single inference implementation
    #*********************
    input_enc_type = inference_record.endpoint_input.encoding
    write_to_file(f"{input_enc_type}", "log2")
    input_data = inference_record.endpoint_input.data.rstrip("\n")
    write_to_file("3", "log3")
    output_data = inference_record.endpoint_output.data.rstrip("\n")
    write_to_file(f"{output_data}", "log4")
    output_data = ast.literal_eval(output_data)[0]
    write_to_file(f"{output_data}", "log5")
    
    input_data = json.loads(input_data)
    input_data = input_data['data'].encode()
    
    r = base64.decodebytes(input_data)
    q = np.frombuffer(r).astype('float64')
    q = np.reshape(q, (-1, 32, 64))
    q_list = list(q[-1,-1,:])
    
    data_dict = {f'feature_{i}': str(d) for i, d in enumerate(q_list)}
    write_to_file(f"{data_dict}", "log6")
    if input_enc_type == "CSV":
        #TODO: add code for csv data
        pass
    elif input_enc_type == "JSON":  
        outputs = {**{"pred": output_data}, **data_dict}
        write_to_file(str(outputs), "log")
        
        outputs = {'Churn': 0, 'Account Length': '134', 'VMail Message': '500', 'Day Mins': '2.5553671354794947', 'Day Calls': '2', 'Eve Mins': '0.1370484071345021', 'Eve Calls': '3', 'Night Mins': '5.550331656408224', 'Night Calls': '300', 'Intl Mins': '7.097685494397488', 'Intl Calls': '7', 'CustServ Calls': '8', 'State_AK': '0', 'State_AL': '0', 'State_AR': '0', 'State_AZ': '0', 'State_CA': '0', 'State_CO': '0', 'State_CT': '0', 'State_DC': '0', 'State_DE': '0', 'State_FL': '0', 'State_GA': '0', 'State_HI': '0', 'State_IA': '0', 'State_ID': '0', 'State_IL': '0', 'State_IN': '0', 'State_KS': '0', 'State_KY': '0', 'State_LA': '0', 'State_MA': '0', 'State_MD': '0', 'State_ME': '0', 'State_MI': '0', 'State_MN': '0', 'State_MO': '1', 'State_MS': '0', 'State_MT': '0', 'State_NC': '0', 'State_ND': '0', 'State_NE': '0', 'State_NH': '0', 'State_NJ': '0', 'State_NM': '0', 'State_NV': '0', 'State_NY': '0', 'State_OH': '0', 'State_OK': '0', 'State_OR': '0', 'State_PA': '0', 'State_RI': '0', 'State_SC': '0', 'State_SD': '0', 'State_TN': '0', 'State_TX': '0', 'State_UT': '0', 'State_VA': '0', 'State_VT': '0', 'State_WA': '0', 'State_WI': '0', 'State_WV': '0', 'State_WY': '0', 'Area Code_657': '0', 'Area Code_658': '0', 'Area Code_659': '0', 'Area Code_676': '0', 'Area Code_677': '0', 'Area Code_678': '0', 'Area Code_686': '0', 'Area Code_707': '0', 'Area Code_716': '0', 'Area Code_727': '0', 'Area Code_736': '0', 'Area Code_737': '0', 'Area Code_758': '0', 'Area Code_766': '0', 'Area Code_776': '0', 'Area Code_777': '1', 'Area Code_778': '0', 'Area Code_786': '0', 'Area Code_787': '0', 'Area Code_788': '0', 'Area Code_797': '0', 'Area Code_798': '0', 'Area Code_806': '0', 'Area Code_827': '0', 'Area Code_836': '0', 'Area Code_847': '0', 'Area Code_848': '0', 'Area Code_858': '0', 'Area Code_866': '0', 'Area Code_868': '0', 'Area Code_876': '0', 'Area Code_877': '0', 'Area Code_878': '0', "Int'l Plan_no": '1', "Int'l Plan_yes": '0', 'VMail Plan_no': '0', 'VMail Plan_yes': '1'}
        
        
        return {str(i).zfill(20) : outputs[d] for i, d in enumerate(outputs)}
    else:
        raise ValueError(f"encoding type {input_enc_type} is not supported") 
