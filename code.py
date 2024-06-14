import numpy as np

def calculate_eigenvectors_and_values(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix) #функція для обчислення власних значень та власних векторів

    for i in range(len(eigenvalues)):
        eigenvalue = eigenvalues[i]
        eigenvector = eigenvectors[:, i] # беремо всі рядки у даному стовпці.
        Av = np.dot(matrix, eigenvector) # matrix_times_eigenvector
        lv = eigenvalue * eigenvector # eigenvalue_times_eigenvector

        if np.allclose(Av, lv):
            print(f"Власне значення: {eigenvalue}")
            print(f"Власний вектор: {eigenvector}")
            print("Рівність виконується")
        else:
            print("Рівність не виконується: A⋅v ≠ λ⋅v")

    return eigenvalues, eigenvectors

matrix = np.array([[1, 2], [2, 1]])
eigenvalues, eigenvectors = calculate_eigenvectors_and_values(matrix)
