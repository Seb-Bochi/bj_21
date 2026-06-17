from dotenv import load_dotenv
load_dotenv()
import os

api_key = os.getenv("WANDB_API_KEY")

if api_key:
    print(f"✅ WANDB_API_KEY loaded (starts with: {api_key[:6]}...)")
else:
    print("❌ WANDB_API_KEY not found — check your .env file")