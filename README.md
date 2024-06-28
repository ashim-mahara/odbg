# Open Data Badger (ODB)

## Overview

Open Data Badger (ODB) is a powerful data batching utility designed for OpenAI endpoints, with plans for future support for local language models (LLMs). ODB provides robust tools for managing, submitting, and processing large batches of data efficiently.

## Features

- **BatchProcessor**:
  - Initializes a SQLite database to store job details.
  - Reads system prompts from a string or file.
  - Creates batch files from input data.
  - Submits batch jobs to OpenAI API.
  - Estimates and processes batches based on maximum batch size and request limits.
  - Supports CSV and Parquet input files.
  - Handles random sampling and dry runs.
  - Verbose mode for detailed batch information.

- **ResultDownloader**:
  - Retrieves the status of batch jobs.
  - Downloads and saves results from completed batch jobs.
  - Supports output in CSV and Parquet formats.

## Requirements

- Python 3.6+
- Poetry for dependency management

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/odb.git
   cd odb
   ```

2. Install the required dependencies using Poetry:

   ```sh
   poetry install
   ```

## Usage

### BatchProcessor

Use the `BatchProcessor` to create and submit batch jobs to the OpenAI API.

#### Example

```sh
poetry run python batch_processor.py \
    --base_url https://your-openai-proxy.com \
    --system_prompt "Your system prompt here" \
    --input_file data/input.csv \
    --data_path data/output \
    --text_field text \
    --task_name example_task \
    --model text-davinci-003 \
    --id_field id \
    --description "Example batch processing task" \
    --random_samples 100 \
    --dry_run \
    --verbose
```

### ResultDownloader

Use the `ResultDownloader` to retrieve and save results from completed batch jobs.

#### Example

```sh
poetry run python result_downloader.py \
    --task_name example_task \
    --task_run_id 1 \
    --output_format csv
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.
