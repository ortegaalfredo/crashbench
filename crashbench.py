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
import time
import anthropic
import openai
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from neuroengine import Neuroengine
from openai import OpenAI


# Global configuration
config = configparser.ConfigParser()

# API configuration
api_key = ""
service_name = 'Neuroengine-Large'

# Global token and time tracking
total_time_seconds = 0.0
total_tokens = 0  # Includes reasoning tokens, excludes judge tokens


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
    postprompt = settings_section.get('PostPrompt')
    return (prompt, system_prompt,postprompt)


def check_api_key_validity(key):
    """
    Validate an OpenAI API key by attempting to list available models.
    
    Args:
        key (str): The API key to validate.
        
    Raises:
        SystemExit: If the API key is invalid.
    """
    global api_key
    try:
        client = OpenAI(api_key=key, base_url=openai.api_base)
        client.models.list()
        api_key = key
    except Exception:
        print("Error: Invalid OpenAI API key")
        exit(1)


def call_AI_claude(systemprompt, prompt, model="claude-3-opus-20240229", max_tokens=8192, temperature=1.0, track_usage=False):
    """
    Call the Claude AI API to analyze code for bugs.
    
    Args:
        systemprompt (str): The system prompt to set the AI's behavior.
        prompt (str): The prompt to send to the AI.
        model (str): The Claude model to use.
        max_tokens (int): Maximum tokens in the response.
        temperature (float): Temperature for response generation (0.0-2.0).
        track_usage (bool): Whether to track token usage globally.
        
    Returns:
        str: The AI's response text.
    """
    global total_tokens
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        exit(1)
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
    # Track token usage if requested (input + output tokens)
    if track_usage and hasattr(response, 'usage'):
        total_tokens += response.usage.input_tokens + response.usage.output_tokens
    return response.content[0].text


def call_AI_chatGPT(systemprompt, prompt, model, max_tokens=8192, temperature=1.0, reasoning_effort="high", judge_endpoint=None, judge_model=None, judge_apikey=None, return_reasoning=False, verbose=False, track_usage=False):
    """
    Call the OpenAI ChatGPT API to analyze code for bugs.
    
    Args:
        systemprompt (str): The system prompt to set the AI's behavior.
        prompt (str): The user prompt to send to the AI.
        model (str): The OpenAI model to use.
        max_tokens (int): Maximum tokens in the response.
        temperature (float): Temperature for response generation (0.0-2.0).
        reasoning_effort (str): Reasoning effort for o1/o3 models ("low", "medium", "high").
        judge_endpoint (str): Optional judge API endpoint.
        judge_model (str): Optional judge model.
        judge_apikey (str): Optional judge API key.
        return_reasoning (bool): Whether to include reasoning content in response.
        verbose (bool): Whether to stream output to stdout.
        track_usage (bool): Whether to track token usage globally.
        
    Returns:
        str: The AI's response text.
    """
    global total_tokens
    # Use judge-specific configuration if provided
    if judge_endpoint is not None:
        base_url = judge_endpoint
    else:
        base_url = openai.api_base
        
    if judge_apikey is not None:
        api_key_to_use = judge_apikey
    else:
        api_key_to_use = api_key
    
    client = OpenAI(api_key=api_key_to_use, base_url=base_url)
    model_to_use = judge_model if judge_model is not None else model
    
    # Build API parameters
    api_params = {
        'model': model_to_use,
        'messages': [
            {'role': 'system', 'content': systemprompt},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': temperature,
        'max_tokens': max_tokens,
        'stream': True
    }
    
    # Add stream_options to include usage data (for token tracking)
    if track_usage:
        api_params['stream_options'] = {'include_usage': True}
    
    # Add reasoning_effort parameter for o1/o3 models if not default
    if reasoning_effort is not None:
        api_params['reasoning_effort'] = reasoning_effort
    
    stream = client.chat.completions.create(**api_params)
    
    full_response = ""
    reasoning_response = ""
    usage_data = None
    
    for chunk in stream:
        # Check if chunk has usage information (some APIs provide this)
        if hasattr(chunk, 'usage') and chunk.usage is not None:
            usage_data = chunk.usage
            continue  # Skip usage chunks
        
        # Collect reasoning tokens if present (for o1/o3 models) - stream them in verbose mode
        if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content is not None:
            reasoning_response += chunk.choices[0].delta.reasoning_content
            # Stream reasoning tokens to stdout with R prefix (verbose mode only)
            if verbose:
                print(f"R{chunk.choices[0].delta.reasoning_content}", end='', flush=True)
        # Collect regular content - stream it in verbose mode
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
            # Stream regular content to stdout (verbose mode only)
            if verbose:
                print(chunk.choices[0].delta.content, end='', flush=True)
    
    if return_reasoning:
        full_response = reasoning_response + "\n" + full_response
    
    # Remove analysis tags if present (for final output only)
    analysis_end = full_response.find("</analysis>")
    if analysis_end > -1:
        full_response = full_response[analysis_end + 11:]
    
    # Track token usage if requested (input + output + reasoning tokens)
    if track_usage and usage_data is not None:
        reasoning_tokens = getattr(usage_data, 'reasoning_tokens', 0)
        total_tokens += usage_data.prompt_tokens + usage_data.completion_tokens + reasoning_tokens
    
    return full_response


def read_apikey():
    """
    Read the OpenAI API key from environment variable or file.
    
    First tries to read from OPENAI_API_KEY environment variable.
    If not found, tries to read from api-key.txt file.
    
    Raises:
        SystemExit: If the API key cannot be loaded.
    """
    global api_key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        try:
            api_key = open('api-key.txt', 'rb').read().strip().decode()
        except Exception:
            print("Error: Could not load OpenAI API key")
            print("Please set OPENAI_API_KEY environment variable or create api-key.txt file.")
            exit(1)
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
        str: The AI's response text.
    """
    global service_name
    hub = Neuroengine(service_name=service_name)
    answer = hub.request(prompt=f"{prompt}:\n{code}", raw=False, temperature=temperature, max_new_len=max_tokens, seed=5)
    return answer


def findBug(file_path, bugline, service_name, engine, max_tokens=8192, temperature=1.0, reasoning_effort="high", judge_endpoint=None, judge_model=None, judge_apikey=None, verbose=False, track_usage=False):
    """
    Analyze a code file to find bugs using the specified AI engine.
    
    Args:
        file_path (str): Path to the code file to analyze.
        bugline (int): The actual line number where the bug is located.
        service_name (str): The name of the AI service/model to use.
        engine (EngineType): The AI engine to use (OPENAI, CLAUDE, or NEUROENGINE).
        max_tokens (int): Maximum tokens for AI response.
        temperature (float): Temperature for AI response (0.0-2.0).
        reasoning_effort (str): Reasoning effort for OpenAI o1/o3 models ("low", "medium", "high").
        judge_endpoint (str): Optional judge API endpoint.
        judge_model (str): Optional judge model.
        judge_apikey (str): Optional judge API key.
        verbose (bool): Whether to stream output to stdout.
        track_usage (bool): Whether to track token usage globally.
        
    Returns:
        int: 1 if bug found within +/- 2 lines, 0 otherwise
    """
    # Read configuration
    prompt, systemprompt, postprompt = readConfig('config.ini')
    
    # Read the code file
    try:
        with open(file_path, 'r') as f:
            c_code = f.read()
    except Exception:
        return 0
    
    # Add line numbers to the entire file
    lines = c_code.split('\n')
    numbered_lines = [f"{line_num} {line}" for line_num, line in enumerate(lines, start=1)]
    code = '\n'.join(numbered_lines)
    code = '\n### BEGIN CODE ###\n' + code + '\n### END CODE ###\n'
    
    # Build initial prompt
    prompt = f"{prompt}{code}{postprompt}\n"
    
    # Call the appropriate AI engine (track usage for main report, not judge)
    if engine == EngineType.OPENAI:
        report = call_AI_chatGPT(systemprompt, prompt, service_name, max_tokens, temperature, reasoning_effort, return_reasoning=True, verbose=verbose, track_usage=track_usage)
    elif engine == EngineType.CLAUDE:
        report = call_AI_claude(systemprompt, prompt, service_name, max_tokens, temperature, track_usage=track_usage)
    elif engine == EngineType.NEUROENGINE:
        report = call_neuroengine(code, prompt, max_tokens, temperature)
    
    # Build judge prompt to extract bug line number
    judge_prompt = (
        f"We have a vulnerability report:\n-- BEGIN REPORT --\n{report}\n-- END REPORT --\n"
        "Your task is to interpret the report and find the line number where the root cause of "
        "the bug happens according to the report. If there is a range, choose the middle. "
        "Your output must follow strictly this format:\n\nbugline=N\n\nDo not write anything else except that line."
    )
    
    judge_report = call_AI_chatGPT(systemprompt, judge_prompt, service_name, max_tokens, temperature, reasoning_effort='low',
                     judge_endpoint=judge_endpoint, judge_model=judge_model, judge_apikey=judge_apikey)
    
    # Find bug line in the judge report
    pattern = r"bugline\s*[=:]\s*(\d+)"
    match = re.search(pattern, judge_report.lower())
    
    if match:
        detected_line = int(match.group(1))
        diff = abs(bugline - detected_line)
    else:
        diff = abs(bugline - 0)
    
    # We give it a +/- 2 lines range
    if diff < 3:
        return 1
    return 0

def format_time(seconds):
    """Format time in a human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f} sec"
    elif seconds < 3600:
        return f"{seconds/60:.1f} min"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def main():
    """
    Main function to run the crashbench bug detection benchmark.
    
    Parses command-line arguments, configures the AI engine, runs tests in parallel,
    and reports results with scoring and timing statistics.
    """
    global service_name
    global engine
    global total_time_seconds
    global total_tokens
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='CrashBench: AI bug detection benchmarking tool')
    parser.add_argument('--repeat', '-r', type=int, default=5,
                        help='Number of test repetitions to average')
    parser.add_argument('--model', type=str, default='Neuroengine-Large',
                        help='Model name to use')
    parser.add_argument('--oai', action='store_true',
                        help='Use OpenAI (requires OPENAI_API_KEY or api-key.txt file)')
    parser.add_argument('--claude', action='store_true',
                        help='Use Claude (requires ANTHROPIC_API_KEY environment variable)')
    parser.add_argument('--endpoint', type=str, default="https://api.openai.com/v1",
                        help='OpenAI-style endpoint to use')
    parser.add_argument('--judge-endpoint', type=str, default=None,
                        help='Judge LLM API endpoint (default: same as regular endpoint)')
    parser.add_argument('--judge-model', type=str, default=None,
                        help='Judge LLM model name (default: same as regular model)')
    parser.add_argument('--judge-apikey', type=str, default=None,
                        help='Judge LLM API key (default: same as regular API key)')
    parser.add_argument('--parallel', '-p', type=int, default=5,
                        help='Number of parallel requests to launch')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed progress information')
    parser.add_argument('--max-tokens', type=int, default=8192,
                        help='Maximum tokens for AI response')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for AI response (0.0-2.0)')
    parser.add_argument('--reasoning-effort', type=str, default='high', choices=['low', 'medium', 'high'],
                        help='Reasoning effort for OpenAI o1/o3 models (low, medium, high)')
    args = parser.parse_args()
    
    # Track start time for ETA calculation
    start_time = time.time()
    
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
        read_apikey()
    
    service_name = args.model

    # Read configuration
    prompt, systemprompt, postprompt = readConfig('config.ini')
    
    # Calculate max possible score
    max_score = 0
    for section in config:
        if section == "DEFAULT" or section == "SETTINGS":
            continue
        if section == "real":
            max_score += 10 * len(config[section])
        else:
            max_score += len(config[section])
    
    # Collect all tasks to run
    tasks = []
    for section in config:
        if section == "DEFAULT" or section == "SETTINGS":
            continue
        files = config[section]
        for file_key in files:
            filename, bugline = files[file_key].split(',')
            for _ in range(args.repeat):
                tasks.append((filename, int(bugline), section))
    
    total_tasks = len(tasks)
    
    print(f"\nCrashBench v3: Running {total_tasks} tests with {args.model}")
    print(f"Engine: {engine.value}, Parallel workers: {args.parallel}")
    print(f"{'='*60}\n")
    
    # Run tasks in parallel
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(findBug, filename, bugline, service_name, engine, max_tokens=args.max_tokens,
                          temperature=args.temperature, reasoning_effort=args.reasoning_effort,
                          judge_endpoint=args.judge_endpoint, judge_model=args.judge_model,
                          judge_apikey=args.judge_apikey, verbose=args.verbose, track_usage=True): (filename, bugline, section)
            for filename, bugline, section in tasks
        }
        
        # Collect scores
        all_scores = []
        passed_tests = 0  # Track number of passed tests
        completed = 0
        
        # Process results as they complete
        for future in as_completed(future_to_task):
            filename, bugline, section = future_to_task[future]
            completed += 1
            try:
                score = future.result()
                # Real tests have much more weight
                if section == "real":
                    score *= 10.0
                
                # Track passed tests (score > 0 means the bug was found)
                if score > 0:
                    passed_tests += 1
                
                # Append individual score to list
                all_scores.append(score)
                
                # Calculate current progress
                current_score = sum(all_scores) / args.repeat
                progress = (completed / total_tasks) * 100
                
                # Calculate pass percentage
                pass_percentage = (passed_tests / completed) * 100 if completed > 0 else 0
                
                # Calculate ETA
                elapsed = time.time() - start_time
                if completed > 0:
                    avg_time_per_task = elapsed / completed
                    remaining_tasks = total_tasks - completed
                    eta_seconds = avg_time_per_task * remaining_tasks
                    eta_minutes = eta_seconds / 60
                    eta_str = f"{eta_minutes:.0f} min" if eta_minutes >= 1 else f"{eta_seconds:.0f} sec"
                else:
                    eta_str = "calculating..."
                
                # Show progress update with ETA and pass percentage
                print(f"[{completed:>3}/{total_tasks}] ({progress:>5.1f}%) ETA: {eta_str:>8s} {filename:30s} Score: {current_score:>6.2f}/{max_score} ({pass_percentage:>5.1f}%)")
                
            except Exception as e:
                print(f"[{completed:>3}/{total_tasks}] Error processing {filename}: {e}")
    
    # Calculate final totals (divided by repeat)
    total_score = sum(all_scores) / args.repeat
    total_time_seconds = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Benchmark Results")
    print(f"{'='*60}")
    percentage = (total_score / max_score) * 100 if max_score > 0 else 0
    print(f"Total Score: {total_score:.2f}/{max_score} ({percentage:.1f}%)")
    print(f"Tests Completed: {len(all_scores)}/{total_tasks}")
    print(f"Total Time: {format_time(total_time_seconds)}")
    print(f"Total Tokens: {total_tokens} (including reasoning tokens, excluding judge tokens)")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
