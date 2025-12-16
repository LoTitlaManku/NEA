
import json
from cryptography.fernet import Fernet


master_key = b"pCt-V4cMUKFvqrWFTD6YjoaXmioK20H3zHzXn1dfd_o="

def save_dict(filename: str, data: dict, key: bytes):
    fernet = Fernet(key)
    json_data = json.dumps(data, indent=2).encode()
    encrypted = fernet.encrypt(json_data)

    with open(filename, "wb") as f:
        f.write(encrypted)

def load_dict(filename: str, key: bytes):
    fernet = Fernet(key)

    with open(filename, "rb") as f:
        encrypted = f.read()

    decrypted = fernet.decrypt(encrypted)
    return json.loads(decrypted.decode())

########################################################################################################################
# TEMP INITIALISING DATA
initial = True
if initial:
    initial_data = {
        "username1": {"password": "ABCD", "Saved stocks": ["aapl", "mu"]},
        "username2": {"password": "DCBA", "Saved stocks": ["nvda", "tsla"]}
    }
    profile_keys = {profile_id: Fernet.generate_key().decode("utf-8") for profile_id in initial_data.keys()}


    save_dict("keys.dat", profile_keys, master_key)

    encrypted_profile_data = {}
    for profile_id, data in initial_data.items():
        profile_key_bytes = profile_keys[profile_id].encode("utf-8")

        fernet = Fernet(profile_key_bytes)

        json_data = json.dumps(data, indent=2).encode()
        encrypted_bytes = fernet.encrypt(json_data)

        encrypted_profile_data[profile_id] = encrypted_bytes.decode("utf-8")


    with open("data.dat", "w") as f:
        json.dump(encrypted_profile_data, f, indent=2)
########################################################################################################################
keys_dict = load_dict("keys.dat", master_key) # file data decrypted {profile: key string, profile: key string}
with open("data.dat", "r") as f: profile_data = json.load(f) # profile identifier normal, profile data encrypted {profile: encrypted, profile: encrypted}

TARGET_PROFILE = "username1"

target_key = keys_dict.get(TARGET_PROFILE) # key as string
target_key = target_key.encode("utf-8") # key string to bytes

target_data = profile_data.get(TARGET_PROFILE) # target data as string
target_data = target_data.encode("utf-8") # data string to bytes

fernet_profile = Fernet(target_key)
target_data = fernet_profile.decrypt(target_data) # decrypt target data

target_data = json.loads(target_data.decode()) # finally convert back into dict
########################################################################################################################
print(f"SUCCESSFULLY decrypted data for {TARGET_PROFILE} using its unique key.")
print("-" * 40)
print(f"Decrypted Data for {TARGET_PROFILE}:")
print(json.dumps(target_data, indent=4))
print("-" * 40)