import argparse
import os
import logging
import pandas as pd
import sqlite3
import json
from openai import OpenAI
from typing import List, Dict

class ResultDownloader:
    def __init__(self, api_key: str, db_name: str = 'jobs.db'):
        """
        Initializes the ResultDownloader with API key and database name.

        Args:
            api_key (str): API key for OpenAI.
            db_name (str, optional): Name of the SQLite database. Defaults to 'jobs.db'.
        """
        self.client = OpenAI(api_key=api_key)
        self.db_name = db_name
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

    def download_results(self, task_name: str, task_run_id: int, output_format: str) -> None:
        """
        Downloads results of completed batch jobs.

        Args:
            task_name (str): Name of the task.
            task_run_id (int): Run ID of the task.
            output_format (str): Output format for the results, either 'csv' or 'parquet'.
        """
        logging.info(f"Checking the status of batch jobs for task: {task_name}, task_run_id: {task_run_id}")

        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('SELECT job_id, file_path FROM jobs WHERE task_name = ? AND task_run_id = ?', (task_name, task_run_id))
        jobs = c.fetchall()

        all_results = []

        for job_id, file_path in jobs:
            batch = self.client.batches.retrieve(batch_id=job_id)

            if batch.status == 'completed':
                logging.info(f"Batch job {job_id} is completed. Downloading results.")
                result_file_id = batch.output_file_id
                result_file = self.client.files.content(file_id=result_file_id)  # type: ignore
                all_results.extend([json.loads(line) for line in result_file.read().splitlines()])

                c.execute('UPDATE jobs SET status = ? WHERE job_id = ?', ('completed', job_id))
                conn.commit()
            else:
                logging.info(f"Batch job {job_id} is not completed. Status: {batch.status}")

        conn.close()

        if all_results:
            output_path = f"data/openai_jobs/{task_name}/{task_run_id}/all_results.jsonl"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                for result in all_results:
                    f.write(json.dumps(result) + "\n")
            logging.info(f"All results saved to {output_path}")

            results_df = pd.DataFrame(all_results)
            if output_format == 'csv':
                results_df.to_csv(f"data/openai_jobs/{task_name}/{task_run_id}/results.csv", index=False)
                logging.info("Results DataFrame created and saved as CSV.")
            else:
                results_df.to_parquet(f"data/openai_jobs/{task_name}/{task_run_id}/results.parquet", index=False)
                logging.info("Results DataFrame created and saved as Parquet.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download results of batch jobs")
    parser.add_argument('--task_name', type=str, required=True, help='Task name')
    parser.add_argument('--task_run_id', type=int, required=True, help='Task run ID')
    parser.add_argument('--output_format', type=str, choices=['csv', 'parquet'], default='parquet', help='Output format (default: parquet)')

    args = parser.parse_args()
    downloader = ResultDownloader(api_key=os.getenv("OPENAI_KEY"))
    downloader.download_results(args.task_name, args.task_run_id, args.output_format)
