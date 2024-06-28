import unittest
import sqlite3
from odbg import BatchProcessor, ResultDownloader

class TestBatchProcessor(unittest.TestCase):

    def test_init_db(self):
        processor = BatchProcessor(api_key="test_key")
        processor.init_db()
        conn = sqlite3.connect('jobs.db')
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'")
        self.assertIsNotNone(c.fetchone())
        conn.close()

    # Add more tests for BatchProcessor and ResultDownloader functions

if __name__ == '__main__':
    unittest.main()
