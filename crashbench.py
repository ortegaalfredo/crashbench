import re
import time
import os
import fnmatch
import os,openai,argparse,sys
import configparser
import argparse

from neuroengine import Neuroengine
# example usage for OpenAI:
#       python benchmark-oai.py --oai --model gpt-4o
# example for Neuroengine:
#       python benchmark-oai.py--model Neuroengine-Medium

# Create a ConfigParser object
config = configparser.ConfigParser()
def readConfig(filename):
    global config
    # Read the INI file 
    config.read(filename)
    # Get the SETTINGS section
    settings_section = config['SETTINGS']
    # Retrieve the prompt and systemprompt settings
    prompt = settings_section.get('Prompt')
    system_prompt = settings_section.get('SystemPrompt')
    return (system_prompt,prompt)

# ---------- OpenAI interface

temperature = 0.0

def check_api_key_validity(api_key):
   try:
        openai.api_key = api_key
        ml=openai.Model.list()
        print("\t[I] OpenAI API key is valid")
   except openai.OpenAIError as e:
        print("\t[E] Invalid OpenAI API key")
        exit(-1)

def call_AI_chatGPT(prompt,model):
        #model = "gpt-3.5-turbo"
        #model="gpt-4-0613"
        response = openai.ChatCompletion.create(
            model=model,
            messages =[
                {'role':'system','content':'You are an expert security researcher, programmer and bug finder. You analize every code you see and are capable of finding programming bugs at an expert or super-human level.'},
                {'role':'user','content':prompt}
                ],
            temperature=temperature,
            max_tokens=1024)
        return response.choices[0]['message']['content']

# import api key
def read_apikey():
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None: api_key=""
    if (len(api_key)==0): # try to load apikey from file
        try:
            api_key=open('api-key.txt','rb').read().strip().decode()
        except:
            print("\t[E] Couldn't load OpenAI Api key, please load it in OPENAI_API_KEY env variable, or alternatively in 'api-key.txt' file.")
            exit(-1)
    check_api_key_validity(api_key)

service_name = 'Neuroengine-Large'
def call_neuroengine(code,prompt):
    global service_name
    hub=Neuroengine(service_name=service_name)
    answer=hub.request(prompt=f"{prompt}:\n{code}",raw=False,temperature=temperature,max_new_len=512,seed=5)
    return(answer)


def extract_function_bodies(c_code):
    # Define a regular expression pattern to match C functions
    function_pattern = r'\s*(\w+\s+\w+\s*\([^)]*\))\s*{([^}]*)}'

    # Use re.findall to find all matching functions in the C code
    functions = re.findall(function_pattern, c_code, re.DOTALL)

    # Return a list of tuples containing function name and body
    return functions

fc=0
def findBug(file_path,bugline):
        global fc
        global service_name
        # Read configuration
        prompt,systemprompt=readConfig('config.ini')
        prompt=systemprompt+"\n"+prompt
        try:
            a=open(file_path)
            c_code=a.read()
            a.close()
        except: return 0
        fc+=1
        # Extract function bodies
        function_bodies = extract_function_bodies(c_code)
        # Print the function bodies
        report=""
        count=0
    
        for function in function_bodies:
            code=f"{function[0]} {{ {function[1]} }}"
            if use_openai:
                prompt=f"{prompt}:\n{code}"
                report=call_AI_chatGPT(prompt,service_name)
            else:
                report=call_neuroengine(code,prompt)
            report+=f'-----{function[0]}---------------: {report}'
            count+=1
            time.sleep(0.5)
        if (count>0):
            print(f'\t[I] Test file: {file_path} report:\n{report}')
            # Find bug line
            pattern = r"bugline=(\d+)"
            match = re.search(pattern, report.lower())
            if match:
                print(f'\t[I] Bug found on line {int(match.group(1))}')
                diff = abs(bugline-int(match.group(1))) 
            else:
                print("\t[I] No match found")
                diff = abs(bugline-0) 
            # We give it a +/- 2 lines range
            if diff<3:
                print("\t[I] Bug line number matches!")
                return 1
            print("\t[I] Bug not found")
            return 0
        return 0

def main():
    global service_name
    global use_openai
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', '-r', type=int, default=5, help='Number of test repetitions to average')
    parser.add_argument('--model', type=str, default='Neuroengine-Large', help='Model name')
    parser.add_argument('--oai', action='store_true', help='Use OpenAI. Need api-key.txt file')
    parser.add_argument('--endpoint', type=str, default="https://api.openai.com/v1",help='OpenAI-style endpoint to use')
    args = parser.parse_args()
    print(f'\t[I] Model: {args.model}')
    print(f'\t[I] Repeat: {args.repeat}')
    if args.oai:
        use_openai=True
        if args.model=="Neuroengine-Large": # change default model name if using OpenaI
            args.model="gpt-4o"
        openai.api_base=args.endpoint
        print(f'\t[I] Using OpenAI API, Endpoint: {openai.api_base}')
        read_apikey()
    else: use_openai=False

    service_name=args.model

    # Read configuration
    prompt,systemprompt=readConfig('config.ini')
    print(f"\t[I] System:{systemprompt}\nPrompt:{prompt}")
    totalscore=0
    # Calculate max possible score
    maxscore=0
    for section in config:
        if section=="DEFAULT" or section=="SETTINGS":
            continue
        if section=="real":
              maxscore+=10*len(config[section])
        else: maxscore+=len(config[section])
    print(f"\t[I] Max possible score: {maxscore}")
    # Read test files
    for section in config:
        if section=="DEFAULT" or section=="SETTINGS":
            continue
        files = config[section]
        print(f"\t[I] Section: {section}")
        for s in files:
            filename,bugline=files[s].split(',')
            tmpscore=0.0
            for q in range(args.repeat):
                score=findBug(filename,int(bugline))
                # real tests have much more weight
                if section=="real":
                    score*=10.0
                tmpscore+=score
            totalscore+=(tmpscore/args.repeat)
            print(f"\t[I] Score:{totalscore}")
    print(f"\t[I] Final score: {totalscore}/{maxscore}")
    

if __name__ == "__main__":
    main()

