import numpy as np


def encrypt_message(message, key_matrix):
    message_vector = np.array([ord(char) for char in message])
    ##Перетворення повідомлення у вектор ASCII-кодів символів:
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    ##Діагоналізація матриці ключа:
    encrypted_vector = np.dot(diagonalized_key_matrix, message_vector)
    ##Шифрування вектору повідомлення:
    return encrypted_vector


def decrypt_message(encrypted_vector, inverse_key_matrix):
    decrypted_vector = np.dot(inverse_key_matrix, encrypted_vector)
    decrypted_message = ''.join([chr(int(round(char.real))) for char in decrypted_vector])
    return decrypted_message


def task3():
    message = input("Write your message: ")
    key_matrix = np.random.randint(1, 10, (len(message), len(message)))  # Генеруємо менші числа для ключової матриці

    encrypted_vector = encrypt_message(message, key_matrix)
    print("Encrypted Message:", encrypted_vector)

    inverse_key_matrix = np.linalg.inv(key_matrix)

    decrypted_message = decrypt_message(encrypted_vector, inverse_key_matrix)
    print("Decrypted Message:", decrypted_message)