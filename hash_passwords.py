import pandas as pd
from werkzeug.security import generate_password_hash
import os

def hash_csv(file):
    if not os.path.exists(file): return
    df = pd.read_csv(file)
    if 'password' in df.columns:
        # Check if already hashed
        def safe_hash(pwd):
            if pd.isna(pwd): return pwd
            pwd_str = str(pwd).strip()
            # werkzeug hashes usually start with 'scrypt:' or 'pbkdf2:'
            if pwd_str.startswith('scrypt:') or pwd_str.startswith('pbkdf2:'): return pwd_str
            return generate_password_hash(pwd_str)
            
        df['password'] = df['password'].apply(safe_hash)
        df.to_csv(file, index=False)
        print(f"Secured passwords in {file}")

print("Securing Data Lake...")
hash_csv("data/teachers.csv")
hash_csv("data/students.csv")
hash_csv("data/parents.csv")
print("SUCCESS: All passwords encrypted with PBKDF2/Scrypt hashing.")
