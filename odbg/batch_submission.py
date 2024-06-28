import argparse
import os
import logging
import pandas as pd
import sqlite3
import json
from tqdm import tqdm
from openai import OpenAI
from typing import List, Dict, Optional, Union
import tiktoken
from tabulate import tabulate

class BatchProcessor:
    def __init__(self, api_key: str, base_url: str, max_batch_size_mb: int = 100, max_requests_per_batch: int = 50000, db_name: str = 'jobs.db'):
        """
        Initializes the BatchProcessor with API key, base URL, batch size limits, and database name.

        Args:
            api_key (str): API key for OpenAI.
            base_url (str): Base URL for OpenAI or compatible server.
            max_batch_size_mb (int, optional): Maximum batch size in megabytes. Defaults to 100.
            max_requests_per_batch (int, optional): Maximum number of requests per batch. Defaults to 50000.
            db_name (str, optional): Name of the SQLite database. Defaults to 'jobs.db'.
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.db_name = db_name
        self.max_batch_size_mb = max_batch_size_mb * 1024 * 1024  # Convert to bytes
        self.max_requests_per_batch = max_requests_per_batch
        self.init_db()

    def init_db(self) -> None:
        """Initializes the SQLite database for storing job information."""
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY,
            task_name TEXT,
            task_run_id INTEGER,
            status TEXT,
            job_id TEXT,
            file_path TEXT,
            description TEXT
        )
        ''')
        conn.commit()
        conn.close()

    def read_system_prompt(self, system_prompt: Optional[Union[str, None]] = None, system_prompt_file: Optional[str] = None) -> Optional[str]:
        """
        Reads the system prompt either directly or from a file.

        Args:
            system_prompt (Optional[Union[str, None]], optional): System prompt as a string. Defaults to None.
            system_prompt_file (Optional[str], optional): Path to the system prompt file. Defaults to None.

        Returns:
            Optional[str]: The system prompt content.
        """
        if system_prompt:
            return system_prompt
        elif system_prompt_file:
            with open(system_prompt_file, 'r') as file:
                return file.read()
        return None

    def create_batch_file(self, df_chunk: pd.DataFrame, data_path: str, system_prompt: Optional[str], batch_index: int, task_name: str, task_run_id: int, text_field: str, model: str, id_field: Optional[str]) -> str:
        """
        Creates a batch file in JSONL format for a chunk of data.

        Args:
            df_chunk (pd.DataFrame): DataFrame chunk to be processed.
            data_path (str): Path to save the batch file.
            system_prompt (Optional[str]): System prompt content.
            batch_index (int): Index of the current batch.
            task_name (str): Name of the task.
            task_run_id (int): Run ID of the task.
            text_field (str): Name of the text field in the DataFrame.
            model (str): Model to use for OpenAI API.
            id_field (Optional[str]): Name of the ID field in the DataFrame.

        Returns:
            str: Path to the created batch file.
        """
        input_file_path = os.path.join(data_path, f"{task_name}/{task_run_id}/batch_{batch_index}.jsonl")
        os.makedirs(os.path.dirname(input_file_path), exist_ok=True)
        logging.info(f"Created directory for batch files: {os.path.dirname(input_file_path)}")
        with open(input_file_path, "w") as f:
            for idx, row in df_chunk.iterrows():
                user_prompt = row[text_field]
                custom_id = row[id_field] if id_field else row['index_col']
                messages = [{"role": "user", "content": user_prompt}]
                if system_prompt:
                    messages.insert(0, {"role": "system", "content": system_prompt})
                prompt = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": messages,
                    },
                }
                f.write(json.dumps(prompt) + "\n")
        return input_file_path

    def submit_batch_job(self, input_file_path: str, description: str) -> Dict:
        """
        Submits a batch job to OpenAI API.

        Args:
            input_file_path (str): Path to the input batch file.
            description (str): Description of the batch job.

        Returns:
            Dict: Response from the batch job submission.
        """
        file_response = self.client.files.create(file=open(input_file_path, "rb"), purpose="batch")
        file_id = file_response.id
        batch_response = self.client.batches.create(
            input_file_id=file_id,
            completion_window="24h",
            endpoint="/v1/chat/completions",
            metadata={"description": description},
        )
        return batch_response

    def get_next_task_run_id(self, task_name: str) -> int:
        """
        Gets the next task run ID for a given task name.

        Args:
            task_name (str): Name of the task.

        Returns:
            int: Next task run ID.
        """
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('SELECT MAX(task_run_id) FROM jobs WHERE task_name = ?', (task_name,))
        max_run_id = c.fetchone()[0]
        conn.close()
        return (max_run_id or 0) + 1

    def estimate_batches(self, df: pd.DataFrame, system_prompt: Optional[str], text_field: str, model: str) -> List[Dict]:
        """
        Estimates the number of batches needed for the given data.

        Args:
            df (pd.DataFrame): DataFrame to be processed.
            system_prompt (Optional[str]): System prompt content.
            text_field (str): Name of the text field in the DataFrame.
            model (str): Model to use for OpenAI API.

        Returns:
            List[Dict]: List of batch information.
        """
        enc = tiktoken.encoding_for_model(model)
        batch_size = 0
        request_count = 0
        start_idx = 0
        batch_info = []

        total_tokens = 0
        batch_tokens = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Estimating Batches"):
            text_tokens = len(enc.encode(row[text_field]))
            total_tokens += text_tokens
            batch_tokens += text_tokens
            text_size = len(row[text_field].encode('utf-8'))
            if request_count + 1 > self.max_requests_per_batch or batch_size + text_size > self.max_batch_size_mb:
                batch_info.append({
                    "batch_index": len(batch_info),
                    "start_idx": start_idx,
                    "end_idx": idx,
                    "num_requests": request_count,
                    "batch_size_mb": batch_size / (1024 * 1024),
                    "batch_tokens": batch_tokens
                })
                batch_size = 0
                request_count = 0
                batch_tokens = 0
                start_idx = idx

            batch_size += text_size
            request_count += 1

        if request_count > 0:
            batch_info.append({
                "batch_index": len(batch_info),
                "start_idx": start_idx,
                "end_idx": len(df),
                "num_requests": request_count,
                "batch_size_mb": batch_size / (1024 * 1024),
                "batch_tokens": batch_tokens
            })

        return total_tokens, batch_info

    def process_batches(self, df: pd.DataFrame, system_prompt: Optional[str], data_path: str, task_name: str, task_run_id: int, text_field: str, model: str, id_field: Optional[str], description: str) -> None:
        """
        Processes and submits batches for the given data.

        Args:
            df (pd.DataFrame): DataFrame to be processed.
            system_prompt (Optional[str]): System prompt content.
            data_path (str): Path to save the batch files.
            task_name (str): Name of the task.
            task_run_id (int): Run ID of the task.
            text_field (str): Name of the text field in the DataFrame.
            model (str): Model to use for OpenAI API.
            id_field (Optional[str]): Name of the ID field in the DataFrame.
            description (str): Description of the task run.
        """
        batch_info = []
        batch_size = 0
        request_count = 0
        start_idx = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Batches"):
            text_size = len(row[text_field].encode('utf-8'))
            if request_count + 1 > self.max_requests_per_batch or batch_size + text_size > self.max_batch_size_mb:
                df_chunk = df.iloc[start_idx:idx]
                input_file_path = self.create_batch_file(df_chunk, data_path, system_prompt, len(batch_info), task_name, task_run_id, text_field, model, id_field)
                batch_response = self.submit_batch_job(input_file_path, description)
                batch_info.append({
                    "batch_id": batch_response.id,
                    "input_file": input_file_path,
                    "status": batch_response.status,
                })
                self.save_job_details(task_name, task_run_id, 'submitted', batch_response.id, input_file_path, description)
                batch_size = 0
                request_count = 0
                start_idx = idx

            batch_size += text_size
            request_count += 1

        if request_count > 0:
            df_chunk = df.iloc[start_idx:]
            input_file_path = self.create_batch_file(df_chunk, data_path, system_prompt, len(batch_info), task_name, task_run_id, text_field, model, id_field)
            batch_response = self.submit_batch_job(input_file_path, description)
            batch_info.append({
                "batch_id": batch_response.id,
                "input_file": input_file_path,
                "status": batch_response.status,
            })
            self.save_job_details(task_name, task_run_id, 'submitted', batch_response.id, input_file_path, description)

        self.save_batch_info(data_path, task_name, task_run_id, batch_info)
        logging.info("Done with submitting batches.")

    def save_job_details(self, task_name: str, task_run_id: int, status: str, job_id: str, file_path: str, description: str) -> None:
        """
        Saves job details to the database.

        Args:
            task_name (str): Name of the task.
            task_run_id (int): Run ID of the task.
            status (str): Status of the job.
            job_id (str): ID of the job.
            file_path (str): Path to the batch file.
            description (str): Description of the task run.
        """
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('''
        INSERT INTO jobs (task_name, task_run_id, status, job_id, file_path, description)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (task_name, task_run_id, status, job_id, file_path, description))
        conn.commit()
        conn.close()

    def save_batch_info(self, data_path: str, task_name: str, task_run_id: int, batch_info: List[Dict]) -> None:
        """
        Saves batch information to a CSV file.

        Args:
            data_path (str): Path to save the batch information.
            task_name (str): Name of the task.
            task_run_id (int): Run ID of the task.
            batch_info (List[Dict]): List of batch information.
        """
        batch_info_df = pd.DataFrame(batch_info)
        output_dir = os.path.join(data_path, f"{task_name}/{task_run_id}")
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Created directory for batch info: {output_dir}")
        batch_info_df.to_csv(os.path.join(output_dir, "batch_info.csv"), index=False)
        logging.info("Batch information saved.")

    def run(self, system_prompt: Optional[Union[str, None]], system_prompt_file: Optional[str], input_file: str, data_path: str, text_field: str, task_name: str, model: str, id_field: Optional[str], description: str, random_samples: Optional[int], dry_run: bool, verbose: bool) -> Union[Dict, None]:
        """
        Main method to run the batch processing.

        Args:
            system_prompt (Optional[Union[str, None]]): System prompt as a string.
            system_prompt_file (Optional[str]): Path to the system prompt file.
            input_file (str): Path to the input CSV or Parquet file.
            data_path (str): Path to save data.
            text_field (str): Name of the text field in the DataFrame.
            task_name (str): Task name to prefix batch files and output.
            model (str): Model to use for OpenAI API.
            id_field (Optional[str]): Name of the ID field in the DataFrame.
            description (str): Description for the task run.
            random_samples (Optional[int]): Number of rows to sample from the DataFrame.
            dry_run (bool): Perform a dry run without executing the batch jobs.
            verbose (bool): Print detailed batch information.

        Returns:
            Union[Dict, None]: Task information if not a dry run, otherwise None.
        """
        logging.info("Starting the batch processing script.")

        task_run_id = self.get_next_task_run_id(task_name)
        logging.info(f"Using task_run_id: {task_run_id}")

        system_prompt_content = self.read_system_prompt(system_prompt, system_prompt_file)

        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith('.parquet'):
            df = pd.read_parquet(input_file)
        else:
            logging.error("Unsupported file format. Use CSV or Parquet.")
            return

        if random_samples:
            df = df.sample(random_samples)
        df['index_col'] = df.index

        total_tokens, batch_info = self.estimate_batches(df, system_prompt_content, text_field, model)

        print(f"Total tokens for the run: {total_tokens}")

        if verbose:
            table = [[info["batch_index"], info["start_idx"], info["end_idx"], info["num_requests"], info["batch_size_mb"], info["batch_tokens"]] for info in batch_info]
            print(tabulate(tabular_data=table, headers=["Batch Index", "Start Index", "End Index", "Num Requests", "Batch Size (MB)", "Batch Tokens"]))

        total_requests = sum(info["num_requests"] for info in batch_info)
        total_batch_size = sum(info["batch_size_mb"] for info in batch_info)
        total_batches = len(batch_info)
        total_batch_tokens = sum(info["batch_tokens"] for info in batch_info)

        summary_table = [
            ["Total Requests", total_requests],
            ["Total Batch Size (MB)", total_batch_size],
            ["Total Batches", total_batches],
            ["Total Batch Tokens", total_batch_tokens]
        ]

        print(tabulate(summary_table, headers=["Metric", "Value"]))

        if not dry_run:
            self.process_batches(df, system_prompt_content, data_path, task_name, task_run_id, text_field, model, id_field, description)

            logging.info(f"To check the status and download results, run the following command:")
            logging.info(f"python -m odbg.download_results.py --task_name {task_name} --task_run_id {task_run_id}")

            return {"task_name": task_name, "task_run_id": task_run_id}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch job processing script")
    parser.add_argument('--base_url', type=str, default=None, help='Base URL for OpenAI Compatible Servers and proxies, example: https://localuser:localpassword@localhost:9090 or https://localhost:8080')
    parser.add_argument('--system_prompt', type=str, help='System prompt as a string')
    parser.add_argument('--system_prompt_file', type=str, help='Path to the system prompt file')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV or Parquet file')
    parser.add_argument('--data_path', type=str, required=True, help="Path to save data")
    parser.add_argument('--text_field', type=str, required=True, help='Name of the text field in the dataframe')
    parser.add_argument('--task_name', type=str, required=True, help='Task name to prefix batch files and output')
    parser.add_argument('--model', type=str, required=True, help='Model to use for OpenAI API')
    parser.add_argument('--id_field', type=str, required=False, help='Name of the ID field in the dataframe')
    parser.add_argument('--description', type=str, required=False, default="Playing with ODBG!", help='Description for the task run')
    parser.add_argument('--random_samples', type=int, help='Number of rows to sample from the dataframe')
    parser.add_argument('--dry_run', action='store_true', help='Perform a dry run without executing the batch jobs')
    parser.add_argument('--verbose', action='store_true', help='Print detailed batch information')

    args = parser.parse_args()
    processor = BatchProcessor(api_key=os.getenv("OPENAI_KEY"), base_url=args.base_url, max_batch_size_mb=100, max_requests_per_batch=50000) # type: ignore
    processor.run(args.system_prompt, args.system_prompt_file, args.input_file, args.data_path, args.text_field, args.task_name, args.model, args.id_field, args.description, args.random_samples, args.dry_run, args.verbose)
