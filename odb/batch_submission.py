import argparse
import os
import logging
import pandas as pd
import sqlite3
import json
from tqdm import tqdm
from openai import OpenAI
from typing import List, Dict, Optional, Union

class BatchProcessor:
    def __init__(self, api_key: str, db_name: str = 'jobs.db'):
        self.client = OpenAI(api_key=api_key)
        self.db_name = db_name
        self.init_db()

    def init_db(self) -> None:
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
        if system_prompt:
            return system_prompt
        elif system_prompt_file:
            with open(system_prompt_file, 'r') as file:
                return file.read()
        return None

    def create_batch_file(self, df_chunk: pd.DataFrame, system_prompt: Optional[str], batch_index: int, task_name: str, task_run_id: int, text_field: str, model: str, id_field: str) -> str:
        input_file_path = f"data/openai_jobs/{task_name}/{task_run_id}/batch_{batch_index}.jsonl"
        os.makedirs(os.path.dirname(input_file_path), exist_ok=True)
        with open(input_file_path, "w") as f:
            for _, row in df_chunk.iterrows():
                user_prompt = row[text_field]
                custom_id = f"{task_name}_{task_run_id}_{row[id_field]}"
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

    def submit_batch_job(self, input_file_path: str) -> Dict:
        file_response = self.client.files.create(file=open(input_file_path, "rb"), purpose="batch")
        file_id = file_response.id
        batch_response = self.client.batches.create(
            input_file_id=file_id,
            completion_window="24h",
            endpoint="/v1/chat/completions",
            metadata={"description": "developing batching script"},
        )
        return batch_response

    def get_next_task_run_id(self, task_name: str) -> int:
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('SELECT MAX(task_run_id) FROM jobs WHERE task_name = ?', (task_name,))
        max_run_id = c.fetchone()[0]
        conn.close()
        return (max_run_id or 0) + 1

    def process_batches(self, df: pd.DataFrame, system_prompt: Optional[str], task_name: str, task_run_id: int, text_field: str, model: str, id_field: str, description: str) -> None:
        batch_info = []
        max_requests_per_batch = 50000
        max_batch_size_mb = 100 * 1024 * 1024
        batch_size = 0
        request_count = 0
        start_idx = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Batches"):
            text_size = len(row[text_field].encode('utf-8'))
            if request_count + 1 > max_requests_per_batch or batch_size + text_size > max_batch_size_mb:
                df_chunk = df.iloc[start_idx:idx]
                input_file_path = self.create_batch_file(df_chunk, system_prompt, len(batch_info), task_name, task_run_id, text_field, model, id_field)
                batch_response = self.submit_batch_job(input_file_path)
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
            input_file_path = self.create_batch_file(df_chunk, system_prompt, len(batch_info), task_name, task_run_id, text_field, model, id_field)
            batch_response = self.submit_batch_job(input_file_path)
            batch_info.append({
                "batch_id": batch_response.id,
                "input_file": input_file_path,
                "status": batch_response.status,
            })
            self.save_job_details(task_name, task_run_id, 'submitted', batch_response.id, input_file_path, description)

        self.save_batch_info(task_name, task_run_id, batch_info)
        logging.info("Done with submitting batches.")

    def save_job_details(self, task_name: str, task_run_id: int, status: str, job_id: str, file_path: str, description: str) -> None:
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('''
        INSERT INTO jobs (task_name, task_run_id, status, job_id, file_path, description)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (task_name, task_run_id, status, job_id, file_path, description))
        conn.commit()
        conn.close()

    def save_batch_info(self, task_name: str, task_run_id: int, batch_info: List[Dict]) -> None:
        batch_info_df = pd.DataFrame(batch_info)
        os.makedirs(f"data/openai_jobs/{task_name}/{task_run_id}", exist_ok=True)
        batch_info_df.to_csv(f"data/openai_jobs/{task_name}/{task_run_id}/batch_info.csv", index=False)
        logging.info("Batch information saved.")

    def run(self, system_prompt: Optional[Union[str, None]], system_prompt_file: Optional[str], input_file: str, text_field: str, task_name: str, model: str, id_field: str, description: str) -> None:
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

        df = df.sample(100)
        self.process_batches(df, system_prompt_content, task_name, task_run_id, text_field, model, id_field, description)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch job processing script")
    parser.add_argument('--system_prompt', type=str, help='System prompt as a string')
    parser.add_argument('--system_prompt_file', type=str, help='Path to the system prompt file')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV or Parquet file')
    parser.add_argument('--text_field', type=str, required=True, help='Name of the text field in the dataframe')
    parser.add_argument('--task_name', type=str, required=True, help='Task name to prefix batch files and output')
    parser.add_argument('--model', type=str, required=True, help='Model to use for OpenAI API')
    parser.add_argument('--id_field', type=str, required=True, help='Name of the ID field in the dataframe')
    parser.add_argument('--description', type=str, required=True, help='Description for the task run')

    args = parser.parse_args()
    processor = BatchProcessor(api_key=os.getenv("OPENAI_KEY"))
    processor.run(args.system_prompt, args.system_prompt_file, args.input_file, args.text_field, args.task_name, args.model, args.id_field, args.description)
