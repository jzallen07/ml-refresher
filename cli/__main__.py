import os

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from cli.main import mlr

if __name__ == "__main__":
    mlr()
