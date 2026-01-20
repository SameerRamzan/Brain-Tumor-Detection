import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from getpass import getpass

# Configuration (Matches api.py)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

async def create_admin():
    print("\n--- 🛡️ Create Admin User Script ---")
    username = input("Enter Admin Username: ").strip()
    if not username:
        print("❌ Username cannot be empty.")
        return

    # Connect to DB
    client = AsyncIOMotorClient(MONGO_URI)
    db = client.brain_tumor_db
    users_collection = db.users

    # Check if user exists
    existing_user = await users_collection.find_one({"username": username})
    
    if existing_user:
        print(f"⚠️ User '{username}' already exists.")
        update = input("Do you want to promote this user to Admin? (y/n): ").lower()
        if update == 'y':
            await users_collection.update_one(
                {"username": username},
                {"$set": {"is_admin": True}}
            )
            print(f"✅ User '{username}' has been promoted to Admin.")
        else:
            print("Operation cancelled.")
        return

    # Create new user
    password = getpass("Enter Password: ")
    confirm_password = getpass("Confirm Password: ")
    
    if password != confirm_password:
        print("❌ Passwords do not match.")
        return
        
    recovery_key = getpass("Enter Recovery Key (for password reset): ")

    hashed_password = get_password_hash(password)
    hashed_recovery = get_password_hash(recovery_key)

    user_doc = {
        "username": username,
        "hashed_password": hashed_password,
        "recovery_key_hash": hashed_recovery,
        "is_admin": True
    }

    await users_collection.insert_one(user_doc)
    print(f"✅ Admin user '{username}' created successfully!")

if __name__ == "__main__":
    try:
        asyncio.run(create_admin())
    except KeyboardInterrupt:
        print("\nScript cancelled.")