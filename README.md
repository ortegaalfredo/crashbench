# crashbench
Crashbench is a LLM benchmark to measure bug-finding and reporting capabilities of LLMs

## Usage:
Example usage for OpenAI:

    python crashbench.py --oai --model gpt-4o

Example using Neuroengine ai provider:

    python crashbench.py --model Neuroengine-Medium

Example using custom openai-style provider (for tabby-api/vllm/aphrodite engine):

    python crashbench.py --oai --endpoint http://127.0.0.1:5000/v1

Also the API key can be written in the api-key.txt file or passed though an environment variable:

    OPENAI_API_KEY="xxxx" python crashbench.py  --oai

## Article

This benchmark was presented at Off-by-One conf 2024.
Article describing it can be downloaded here:

[AI powered bughunting](https://github.com/ortegaalfredo/autokaker/blob/main/doc/AI-powered-bughunting-aortega-paper.pdf)

## Configuration

Test cases are stored in the config.ini file. The test format is:

```
[Basic]
file1=tests/stack1.c,6
file2=tests/stack2.c,6
```

Each entry just have the test filename and expected line of the bug, if found. If no bug is expected to be found (for a false positive test), adjust the line number to 0. Every found bug increases the final score.

## Current leaderboard (V2)

![Leaderboard V2](https://raw.githubusercontent.com/ortegaalfredo/crashbench/main/models-scores-v2.png)

