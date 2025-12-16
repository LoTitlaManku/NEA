
import json
from cryptography.fernet import Fernet


class Profile:
    def __init__(self, manager: 'DataManager', username: str, key: bytes, data: dict):
        self.__manager = manager
        self.__username = username
        self.__key = key
        self.__data = data

    def update_data(self, new_data: dict):
        new_data = {key: value for key,value in new_data.items() if key != "password"}
        self.__data.update(new_data)
        self.__manager.save_profile_data(self.__username, self.__key, self.__data)

    def get_data(self) -> dict:
        return {key: value for key,value in self.__data.items() if key != "password"}

    def get_username(self) -> str:
        return self.__username


class DataManager:
    def __init__(self):
        self.__master_key = b"pCt-V4cMUKFvqrWFTD6YjoaXmioK20H3zHzXn1dfd_o="
        self.__key_file = "keys.dat"
        self.__data_file = "data.dat"

        self.__keys = self.load_decrypt_file(self.__key_file, self.__master_key)
        with open(self.__data_file, "r") as f: self.__profile_datas = json.load(f)

    def load_decrypt_file(self, filename: str, key: bytes) -> dict:
        fernet = Fernet(key)
        with open(filename, "rb") as f:
            encrypted_data = f.read()

        decrypted_data = fernet.decrypt(encrypted_data)
        return json.loads(decrypted_data.decode())

    def save_encrypt_file(self, filename: str, data: dict, key: bytes):
        fernet = Fernet(key)
        json_data = json.dumps(data).encode()
        encrypted_data = fernet.encrypt(json_data)

        with open(filename, "wb") as f:
            f.write(encrypted_data)

    def get_profile(self, username, password) -> Profile | str:

        target_key = self.__keys.get(username) # key as string
        if target_key is None: return "Non-existent profile"
        target_key = target_key.encode("utf-8")  # key string to bytes

        target_data = self.__profile_datas.get(username)  # target data as string
        target_data = target_data.encode("utf-8")  # data string to bytes

        fernet = Fernet(target_key)
        target_data = fernet.decrypt(target_data)  # decrypt target data
        target_data = json.loads(target_data.decode())  # convert back into dict

        if target_data["password"] != password: return "Incorrect password"

        return Profile(self, username, target_key, target_data)

    def create_profile(self, username, password) -> str:

        if username in self.__profile_datas.keys():
            return "Profile already exists"

        new_key = Fernet.generate_key()

        fernet = Fernet(new_key)
        data = fernet.encrypt(json.dumps({"password": password, "Saved stocks": []}).encode()).decode("utf-8")
        self.__profile_datas[username] = data
        with open(self.__data_file, "w") as f: json.dump(self.__profile_datas, f)

        self.__keys[username] = new_key.decode("utf-8")
        self.save_encrypt_file(self.__key_file, self.__keys, self.__master_key)

        return "Profile created"


    def save_profile_data(self, username: str, key: bytes, data: dict):
        data_to_save = data
        fernet = Fernet(key)
        json_data = json.dumps(data_to_save).encode()
        encrypted_data = fernet.encrypt(json_data).decode("utf-8")

        self.__profile_datas[username] = encrypted_data
        with open(self.__data_file, "w") as f: json.dump(self.__profile_datas, f)




if __name__ in "__main__":
    manage = DataManager()
    data = manage.get_profile("username3", "password")
    if data == "Non-existent profile":
        print("Profile data is non-existent")
    elif data == "Incorrect password":
        print("Incorrect password")
    else:
        print(data.get_data())

    manage.create_profile("username3", "password") # print( ... ) -> profile already exists

    # profile = manage.get_profile("username3", "password")
    # print(profile.get_data())
    #
    # profile.update_data({"Saved stocks": ["TSLA"]})
    #
    # print(profile.get_data())
