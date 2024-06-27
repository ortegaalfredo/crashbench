# crashbench
Crashbench is a LLM benchmark to measure bug-finding and reporting capabilities of LLMs

## Usage:
Example usage for OpenAI:
       python benchmark-oai.py --oai --model gpt-4o

Example using Neuroengine ai provider:
       python benchmark-oai.py--model Neuroengine-Medium

## Configuration

Test cases are stored in the config.ini file. The test format is:

```
[Basic]
file1=tests/stack1.c,6
file2=tests/stack2.c,6
```

Each entry just have the test filename and expected line of the bug, if found. Every found bug increases the final score.

## Current leaderboard (V1)

![Leaderboard](https://raw.githubusercontent.com/ortegaalfredo/crashbench/main/models-scores.png)

