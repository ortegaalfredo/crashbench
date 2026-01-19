"""
CrashBench: A benchmarking tool for evaluating AI models on bug detection tasks.

This script runs bug detection tests on various code files using different AI engines
(OpenAI, Claude, Neuroengine) and reports the results with scoring and timing statistics.

Example usage:
    python crashbench.py --oai --model gpt-4o
    python crashbench.py --claude --model claude-3-opus-20240229
    python crashbench.py --model Neuroengine-Medium
"""

import argparse
import configparser
import os
import re
import sys
import time
import anthropic
import openai
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from neuroengine import Neuroengine
from openai import OpenAI
from threading import Lock


# Global configuration
config = configparser.ConfigParser()

# Token tracking for usage estimation
totalTokens = 0
tokenizer = tiktoken.get_encoding("cl100k_base")

# Thread-safe locks for global variables
totalTokens_lock = Lock()
fc_lock = Lock()

# API configuration
temperature = 1.0
api_key = ""
service_name = 'Neuroengine-Large'

# File counter for tracking processed files
fc = 0


class EngineType(Enum):
    """Enumeration of supported AI engines."""
    OPENAI = "openai"
    CLAUDE = "claude"
    NEUROENGINE = "neuroengine"


def readConfig(filename):
    """
    Read configuration from an INI file.
    
    Args:
        filename (str): Path to the configuration file.
        
    Returns:
        tuple: (prompt, system_prompt) strings from the SETTINGS section.
    """
    global config
    config.read(filename)
    settings_section = config['SETTINGS']
    prompt = settings_section.get('Prompt')
    system_prompt = settings_section.get('SystemPrompt')
    return (prompt, system_prompt)


def check_api_key_validity(key):
    """
    Validate an OpenAI API key by attempting to list available models.
    
    Args:
        key (str): The API key to validate.
        
    Exits:
        If the API key is invalid.
    """
    global api_key
    try:
        client = OpenAI(api_key=key, base_url=openai.api_base)
        ml = client.models.list()
        print("\t[I] OpenAI API key is valid")
        api_key = key
    except Exception as e:
        print(e)
        print("\t[E] Invalid OpenAI API key")
        exit(-1)


def call_AI_claude(systemprompt, prompt, model="claude-3-opus-20240229", max_tokens=8192, temperature=1.0):
    """
    Call the Claude AI API to analyze code for bugs.
    
    Args:
        systemprompt (str): The system prompt to set the AI's behavior.
        prompt (str): The prompt to send to the AI.
        model (str): The Claude model to use.
        max_tokens (int): Maximum tokens in the response.
        temperature (float): Temperature for response generation (0.0-2.0).
        
    Returns:
        str: The AI's response text.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key is None:
        print("[E] Cannot read environment variable ANTHROPIC_API_KEY")
        exit(-1)
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=systemprompt,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.content[0].text


def call_AI_chatGPT(systemprompt, prompt, model, verbose=False, max_tokens=8192, temperature=1.0):
    """
    Call the OpenAI ChatGPT API to analyze code for bugs.
    
    Args:
        systemprompt (str): The system prompt to set the AI's behavior.
        prompt (str): The user prompt to send to the AI.
        model (str): The OpenAI model to use.
        verbose (bool): Whether to print the AI response stream.
        max_tokens (int): Maximum tokens in the response.
        temperature (float): Temperature for response generation (0.0-2.0).
        
    Returns:
        str: The AI's full response text.
    """
    if verbose:
        print("-"*100)
        print(prompt)
        print("-"*100)
    client = OpenAI(api_key=api_key, base_url=openai.api_base)
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': systemprompt},
            {'role': 'user', 'content': prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )
    
    full_response = ""
    if verbose:
        print("[AI Response]: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            if verbose:
                print(content, end="", flush=True)
            full_response += content
    if verbose:
        print()  # New line after the response
    return full_response


def read_apikey():
    """
    Read the OpenAI API key from environment variable or file.
    
    First tries to read from OPENAI_API_KEY environment variable.
    If not found, tries to read from api-key.txt file.
    
    Exits:
        If the API key cannot be loaded.
    """
    global api_key
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        api_key = ""
    if len(api_key) == 0:
        try:
            api_key = open('api-key.txt', 'rb').read().strip().decode()
        except Exception:
            print("\t[E] Couldn't load OpenAI Api key, please load it in OPENAI_API_KEY env variable, or alternatively in 'api-key.txt' file.")
            exit(-1)
    check_api_key_validity(api_key)


def call_neuroengine(code, prompt, max_tokens=8192, temperature=1.0):
    """
    Call the Neuroengine API to analyze code for bugs.
    
    Args:
        code (str): The code to analyze.
        prompt (str): The prompt to send to the AI.
        max_tokens (int): Maximum tokens in the response.
        temperature (float): Temperature for response generation (0.0-2.0).
        
    Returns:
        str: The AI's response.
    """
    global service_name
    hub = Neuroengine(service_name=service_name)
    answer = hub.request(prompt=f"{prompt}:\n{code}", raw=False, temperature=temperature, max_new_len=max_tokens, seed=5)
    return answer


def findBug(file_path, bugline, service_name, engine, verbose=False, max_tokens=8192, temperature=1.0):
    """
    Analyze a code file to find bugs using the specified AI engine.
    
    Args:
        file_path (str): Path to the code file to analyze.
        bugline (int): The actual line number where the bug is located.
        service_name (str): The name of the AI service/model to use.
        engine (EngineType): The AI engine to use (OPENAI, CLAUDE, or NEUROENGINE).
        verbose (bool): Whether to print verbose output.
        max_tokens (int): Maximum tokens for AI response.
        temperature (float): Temperature for AI response (0.0-2.0).
        
    Returns:
        int: 1 if the bug was found within +/- 2 lines, 0 otherwise.
    """
    global fc
    global totalTokens
    
    # Read configuration
    prompt, systemprompt = readConfig('config.ini')
    
    # Read the code file
    try:
        with open(file_path, 'r') as f:
            c_code = f.read()
    except Exception:
        return 0
    
    fc += 1
    
    # Add line numbers to the entire file
    lines = c_code.split('\n')
    numbered_lines = []
    for line_num, line in enumerate(lines, start=1):
        numbered_lines.append(f"{line_num} {line}")
    code = '\n'.join(numbered_lines)
    code = '\n### BEGIN CODE ###\n'+code+'\n### END CODE ###\n'
    report = ""
    tokens = 0
    print(f"[I]\tProcessing file {file_path}")
    
    # Call the appropriate AI engine
    if engine == EngineType.OPENAI:
        prompt = f"{prompt}:\n{code}"
        report = call_AI_chatGPT(systemprompt, prompt, service_name, verbose, max_tokens, temperature)
    elif engine == EngineType.CLAUDE:
        prompt = f"{prompt}:\n{code}"
        report = call_AI_claude(systemprompt, prompt, service_name, max_tokens, temperature)
    elif engine == EngineType.NEUROENGINE:
        report = call_neuroengine(code, prompt, max_tokens, temperature)
    
    # Estimate amount of used tokens
    try:
        tokens += len(tokenizer.encode(prompt)) + 34
        tokens += len(tokenizer.encode(report))
        if engine == EngineType.NEUROENGINE:
            tokens += len(tokenizer.encode(code))
    except Exception:
        pass
    
    report += f'-----{file_path}---------------: {report}'
    time.sleep(0.5)
    
    with totalTokens_lock:
        totalTokens += tokens
    
    print(f'\t[I] Used tokens on this stage: {tokens} total Tokens: {totalTokens}')
    
    # Find bug line in the report
    pattern = r"bugline=(\d+)"
    match = re.search(pattern, report.lower())
    if match:
        print(f'\t[I] Bug found on line {int(match.group(1))}')
        diff = abs(bugline - int(match.group(1)))
    else:
        print("\t[I] No match found")
        diff = abs(bugline - 0)
    
    # We give it a +/- 2 lines range
    if diff < 3:
        print("\t[I] Bug line number matches!")
        return 1
    print("\t[I] Bug not found")
    return 0

def main():
    """
    Main function to run the crashbench bug detection benchmark.
    
    Parses command-line arguments, configures the AI engine, runs tests in parallel,
    and reports results with scoring and timing statistics.
    """
    global service_name
    global engine
    global totalTokens
    
    # Track start time for performance measurement
    start_time = time.time()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='CrashBench: AI bug detection benchmarking tool')
    parser.add_argument('--repeat', '-r', type=int, default=5,
                        help='Number of test repetitions to average')
    parser.add_argument('--model', type=str, default='Neuroengine-Large',
                        help='Model name to use')
    parser.add_argument('--oai', action='store_true',
                        help='Use OpenAI. Need api-key.txt file')
    parser.add_argument('--claude', action='store_true',
                        help='Use Claude. Need API key on environment variable ANTHROPIC_API_KEY')
    parser.add_argument('--endpoint', type=str, default="https://api.openai.com/v1",
                        help='OpenAI-style endpoint to use')
    parser.add_argument('--parallel', '-p', type=int, default=5,
                        help='Number of parallel requests to launch')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print the AI response stream')
    parser.add_argument('--max-tokens', type=int, default=8192,
                        help='Maximum tokens for AI response')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for AI response (0.0-2.0)')
    args = parser.parse_args()
    
    # Set default engine
    engine = EngineType.NEUROENGINE
    
    # Configure engine based on arguments
    if args.claude:
        engine = EngineType.CLAUDE
        if args.model == "Neuroengine-Large":
            args.model = "claude-3-opus-20240229"
    
    if args.oai:
        engine = EngineType.OPENAI
        if args.model == "Neuroengine-Large":
            args.model = "gpt-4o"
        openai.api_base = args.endpoint
        print(f'\t[I] Using OpenAI API, Endpoint: {openai.api_base} model {args.model}')
        read_apikey()
    
    # Print configuration
    print(f'\t[I] Model: {args.model}')
    print(f'\t[I] Repeat: {args.repeat}')
    print(f'\t[I] Parallel: {args.parallel}')
    print(f'\t[I] Max tokens: {args.max_tokens}')
    print(f'\t[I] Temperature: {args.temperature}')

    service_name = args.model

    # Read configuration
    prompt, systemprompt = readConfig('config.ini')
    print(f"\t[I] System:{systemprompt}\nPrompt:{prompt}")
    
    # Calculate max possible score
    maxscore = 0
    for section in config:
        if section == "DEFAULT" or section == "SETTINGS":
            continue
        if section == "real":
            maxscore += 10 * len(config[section])
        else:
            maxscore += len(config[section])
    print(f"\t[I] Max possible score: {maxscore}")
    
    # Reset token counter
    totalTokens = 0
    
    # Collect all tasks to run
    tasks = []
    for section in config:
        if section == "DEFAULT" or section == "SETTINGS":
            continue
        files = config[section]
        print(f"\t[I] Section: {section}")
        for s in files:
            filename, bugline = files[s].split(',')
            for q in range(args.repeat):
                tasks.append((filename, int(bugline), section))
    
    # Run tasks in parallel
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(findBug, filename, bugline, service_name, engine, args.verbose, args.max_tokens, args.temperature): (filename, bugline, section)
            for filename, bugline, section in tasks
        }
        
        # Collect all individual scores
        all_scores = []
        
        # Process results as they complete
        for future in as_completed(future_to_task):
            filename, bugline, section = future_to_task[future]
            try:
                score = future.result()
                # Real tests have much more weight
                if section == "real":
                    score *= 10.0
                
                # Append individual score to list
                all_scores.append(score)
                
                # Print running total after each test (divided by repeat)
                totalscore = sum(all_scores) / args.repeat
                print(f"\t[I] Test complete. Current score: {totalscore}/{maxscore}")
                
            except Exception as e:
                print(f"\t[E] Error processing {filename}: {e}")
    
    # Calculate final total score (divided by repeat)
    totalscore = sum(all_scores) / args.repeat
    print(f"\t[I] Final score: {totalscore}/{maxscore}")
    print(f"\t[I] Total tokens used: {totalTokens}")
    
    # Calculate and print total time and tokens per second
    end_time = time.time()
    total_time = end_time - start_time
    tokens_per_second = totalTokens / total_time if total_time > 0 else 0
    print(f"\t[I] Total time: {total_time:.2f} seconds")
    print(f"\t[I] Tokens per second: {tokens_per_second:.2f}")


if __name__ == "__main__":
    main()

