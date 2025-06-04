# Local LLM Notebooks and Utilities

This repository contains a collection of local LLM notebooks and helper scripts. Most models use the [`transformers`](https://github.com/huggingface/transformers) library by Hugging Face and are designed to run locally.

## Structure and Usage

- **Training**: Notebooks (`.ipynb`) are typically converted to Python scripts (`.py`) and run in the background using `tmux`. Logs are saved for later review.
- **Control**: Scripts are included to stop training by terminating the corresponding `tmux` session.

## Search-Based Models

Some models require a running instance of a [SearXNG](https://github.com/searxng/searxng) container for web based seatching. Make sure this container is set up and running before using search-integrated models.

## Environment Note

These scripts assume you are using a Conda environment named `llm-env`.  
You will need to manually set up your own environment with the necessary dependencies. Automatic setup is not currently provided.

## Coming Soon

- Setup instructions
- Docker container for full environment and dependency setup
