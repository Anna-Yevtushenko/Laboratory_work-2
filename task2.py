import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt


def task2():
# Завантаження зображення
    image_path = 'v013eb4229134ab21o74j.jpg'
    image_raw = imread(image_path)
    print("Розмір зображення:", image_raw.shape)

    # Відображення початкового зображення
    plt.figure(figsize=(12, 8))
    plt.imshow(image_raw)
    plt.title("Початкове зображення")
    plt.show()


    # Перетворення зображення в чорно-біле
    image_sum = image_raw.sum(axis=2)
    print("Розмір чорно-білого зображення:", image_sum.shape)

    # Нормалізація зображення
    image_bw = image_sum / image_sum.max()
    print("Максимальне значення після нормалізації:", image_bw.max())

    # Відображення чорно-білого зображення
    plt.figure(figsize=(12, 8))
    plt.imshow(image_bw, cmap=plt.cm.gray)
    plt.title("Чорно-біле зображення")
    plt.show()


    from sklearn.decomposition import PCA, IncrementalPCA

    # Застосування PCA
    pca = PCA()
    pca.fit(image_bw)

    # Кумулятивна дисперсія
    var_cumu = np.cumsum(pca.explained_variance_ratio_) * 100

    # Знаходження кількості компонент для покриття 95% дисперсії
    k = np.argmax(var_cumu >95)
    print("Кількість компонент, які пояснюють 95% дисперсії: + " , str(k))

    # Відображення графіку
    plt.figure(figsize=(10, 5))
    plt.title('Cumulative Explained Variance explained by the components')
    plt.ylabel('Cumulative Explained variance (%)')
    plt.xlabel('Principal components')
    plt.axvline(x=k, color="k", linestyle="--")
    plt.axhline(y=95, color="r", linestyle="--")
    plt.plot(var_cumu)
    plt.show()



    ipca = IncrementalPCA(n_components=k)
    image_recon = ipca.inverse_transform(ipca.fit_transform(image_bw))

    # Plotting the reconstructed image
    plt.figure(figsize=(12,8))
    plt.imshow(image_recon,cmap = plt.cm.gray)
    plt.show()


    # Function to reconstruct and plot image for a given number of components

    def plot_at_k(k):
        ipca = IncrementalPCA(n_components=k)
        image_recon = ipca.inverse_transform(ipca.fit_transform(image_bw))
        plt.imshow(image_recon, cmap=plt.cm.gray)


    k = 150
    plt.figure(figsize=(12, 8))
    plot_at_k(100)
    plt.show()



    ks = [10, 25, 50, 100, 150, 250]

    plt.figure(figsize=(15,9))

    for i in range(6):
        plt.subplot(2,3,i+1)
        plot_at_k(ks[i])
        plt.title("Components: "+str(ks[i]))

    plt.subplots_adjust(wspace=0.2, hspace=0.0)
    plt.show()