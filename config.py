"""
Settings for the database and the language model.

This file does not talk to Neo4j. It only reads names, passwords, and
the model address so the other files stay free of hardcoded secrets.

Input: optional .env file, or the real environment variables.
Output: one dict named settings.
"""

import os
from pathlib import Path


# Folder that contains this file. All data paths hang off this folder
# so the script works no matter which directory you launch it from.
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "graph_data.json"
NEW_FILE = ROOT / "data" / "new_tickets.json"


def load_env_file():
    """
    Copy KEY=VALUE lines from .env into the process environment.

    Why:
        A beginner can drop a key into a file instead of learning
        how to set Windows environment variables.

    Input:
        ROOT / ".env" if that file exists. Missing file is fine.

    Output:
        Nothing returned. os.environ may gain new keys.
        A key that is already set in the real environment is left as it is.
    """
    env_path = ROOT / ".env"
    if not env_path.exists():
        return

    # Each line is one setting. Blank lines and comments are skipped.
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def get_settings():
    """
    Build the settings dict the rest of the program reads.

    Input:
        Environment variables, after .env has been loaded.

    Output:
        dict with neo4j connection fields and the model fields.
        api_key is an empty string when the model should stay off.
    """
    load_env_file()

    # Defaults match docker-compose.yml so a first run needs no .env file.
    return {
        "neo4j_uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        "neo4j_user": os.environ.get("NEO4J_USER", "neo4j"),
        "neo4j_password": os.environ.get("NEO4J_PASSWORD", "ticket123"),
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    }
