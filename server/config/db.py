import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Mock MongoDB using JSON file for the MVP
DB_PATH = Path("data/db.json")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

class JsonCollection:
    def __init__(self, name):
        self.name = name
        self.db_path = DB_PATH
        self._load_db()
        
        if name not in self.db:
            self.db[name] = []
            self._save_db()

    def _load_db(self):
        if self.db_path.exists():
            with open(self.db_path, "r") as f:
                try:
                    self.db = json.load(f)
                except json.JSONDecodeError:
                    self.db = {}
        else:
            self.db = {}

    def _save_db(self):
        with open(self.db_path, "w") as f:
            json.dump(self.db, f, indent=4)

    def find_one(self, query):
        self._load_db()
        collection = self.db.get(self.name, [])
        for item in collection:
            # Simple exact match for all query keys
            if all(item.get(k) == v for k, v in query.items()):
                return item
        return None

    def insert_one(self, document):
        self._load_db()
        if self.name not in self.db:
            self.db[self.name] = []
        
        self.db[self.name].append(document)
        self._save_db()
        return True

# Mock the database object
class MockDB:
    def __getitem__(self, name):
        return JsonCollection(name)

# Expose same interface as before
users_collection = JsonCollection("users")
