import numpy as np
from task2 import task2

matrix = np.array([[1, 2],
                   [2, 1]])

def calculate_eigenvectors_and_values(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)  # функція для обчислення власних значень та власних векторів

    for i in range(len(eigenvalues)):
        eigenvalue = eigenvalues[i]
        eigenvector = eigenvectors[:, i]  # беремо всі рядки у даному стовпці.
        Av = np.dot(matrix, eigenvector)  # matrix_times_eigenvector
        lv = eigenvalue * eigenvector  # eigenvalue_times_eigenvector

        if np.allclose(Av, lv):
            print(f"Власне значення: {eigenvalue}")
            print(f"Власний вектор: {eigenvector}")
            print("Рівність виконується")
        else:
            print("Рівність не виконується: A⋅v ≠ λ⋅v")
    return eigenvalues, eigenvectors

def start_program():
    while True:
        start_program = input("Do you want to start? Write 'y' or 'n': ").strip().lower()
        if start_program == 'y':
            return True
        elif start_program == 'n':
            print("Program exited.")
            exit()
        else:
            print("You entered an invalid answer. Please enter 'y' or 'n'.")

def main():
    start_program()
    while True:
        command = input("Chose task: 1, 2, 3 or exit: ").strip().lower()
        if command == 'exit':
            print('Exit!')
            break
        elif command == '1':
            eigenvalues, eigenvectors = calculate_eigenvectors_and_values(matrix)
        elif command == '2':
            task2()
        else:
            print("Invalid option. Choose between '1', '2', or 'exit'.")


if __name__ == "__main__":
    main()
