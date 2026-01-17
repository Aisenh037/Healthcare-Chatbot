from server.config.db import users_collection
from server.auth.hash_utils import hash_password

def seed_users():
    users = [
        {"username": "admin", "password": "admin123", "role": "admin"},
        {"username": "doctor", "password": "doc123", "role": "doctor"},
        {"username": "nurse", "password": "nurse123", "role": "nurse"},
        {"username": "patient", "password": "patient123", "role": "patient"}
    ]

    print("🌱 Seeding Users...")
    for u in users:
        if not users_collection.find_one({"username": u["username"]}):
            users_collection.insert_one({
                "username": u["username"],
                "password": hash_password(u["password"]),
                "role": u["role"]
            })
            print(f"✅ Created {u['username']} ({u['role']})")
        else:
            print(f"ℹ️  {u['username']} already exists")

    print("\nDone! Login with these credentials.")

if __name__ == "__main__":
    seed_users()
