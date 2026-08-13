from dotenv import load_dotenv
import os

load_dotenv()

required = [
    "AZURE_SEARCH_ENDPOINT",
    "AZURE_SEARCH_API_KEY",
    "AZURE_SEARCH_INDEX_NAME",
    "GITHUB_MODELS_TOKEN",
    "GITHUB_MODELS_ENDPOINT",
    "GITHUB_MODELS_MODEL_NAME",
]

for var in required:
    val = os.environ.get(var)
    status = "SET" if val else "MISSING"
    print(f"{var}: {status}")