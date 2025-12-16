
import json
from cryptography.fernet import Fernet

# key = Fernet.generate_key()

master_key = b"pCt-V4cMUKFvqrWFTD6YjoaXmioK20H3zHzXn1dfd_o="


def save_dict(filename:str, data: dict, key: bytes):
    fernet = Fernet(key)

    json_data = json.dumps(data, indent=2).encode()
    encrypted = fernet.encrypt(json_data)

    with open(filename, "wb") as f:
        f.write(encrypted)

def load_dict(filename: str, key: bytes) -> dict:
    fernet = Fernet(key)

    with open(filename, "rb") as f:
        encrypted = f.read()

    decrpyted = fernet.decrypt(encrypted)
    return json.loads(decrpyted.decode())


data = {"Profile1": {"username": None, "password": None, "Saved stocks": ["aapl"]},
        "Profile2": {"username": None, "password": None, "Saved stocks": ["nvda"]}}


temp_key = Fernet.generate_key()
temp_key_s = temp_key.decode("utf-8")
save_dict("master", {"Profile1": temp_key_s}, master_key)

keys = load_dict("keys", master_key)


for key in keys.values():
    print(key)

key = keys["Profile1"].encode("utf-8")


save_dict("test", data, key)

print(load_dict("test", key))