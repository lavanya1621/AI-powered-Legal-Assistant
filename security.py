from cryptography.fernet import Fernet

def encrypt_file(file_path, key):
    f = Fernet(key)
    with open(file_path, 'rb') as file:
        data = file.read()
    encrypted_data = f.encrypt(data)
    with open(file_path + '.encrypted', 'wb') as file:
        file.write(encrypted_data)