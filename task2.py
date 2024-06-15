import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA


def task2():
    image_path = 'v013eb4229134ab21o74j.jpg'
    image_raw = imread(image_path) #завантаження зображення
    print("Розмір початкового зображення:", image_raw.shape) ## height, width, кількість каналів

    plt.figure(figsize=(10, 6)) ## встановлюємо розмір на екрані
    plt.imshow(image_raw)
    plt.title("Початкове зображення")
    plt.show()

    ##print("____________________________________________________________")

    # Перетворення зображення в чорно-біле
    image_sum = image_raw.sum(axis=2)
    print("Розмір чорно-білого зображення:", image_sum.shape)
    ## Сумує значення кольорових каналів для кожного пікселя,
    ## перетворюючи тривимірний масив у двовимірний.

    image_bw = image_sum / image_sum.max()
    # Нормалізація зображення
    print("Кількість каналів кольорів:", image_bw.max())


    plt.figure(figsize=(10, 6))
    plt.imshow(image_bw, cmap=plt.cm.gray)##вказуємо палітру чорно-білу
    plt.title("Чорно-біле зображення")
    plt.show()

    ##print("____________________________________________________________")

    def reconstruct_and_plot_image(components_number):
        ipca = IncrementalPCA(n_components=components_number)
        image_recon = ipca.inverse_transform(ipca.fit_transform(image_bw))
        plt.imshow(image_recon, cmap=plt.cm.gray)


    pca = PCA()
    pca.fit(image_bw)
    cumulative_sum = np.cumsum(pca.explained_variance_ratio_) * 100
    num_components_95 = np.argmax(cumulative_sum > 95)
    print("Кількість компонент, які пояснюють 95% дисперсії:", num_components_95)

    plt.figure(figsize=(10, 6))
    plt.title('Кумулятивна пояснена дисперсія пояснена компонентами')
    plt.ylabel('Кумулятивна пояснена дисперсія (%)')
    plt.xlabel('Головні компоненти')
    plt.axvline(x=num_components_95, color="k", linestyle="--")
    plt.axhline(y=95, color="r", linestyle="--")
    plt.plot(cumulative_sum)
    plt.show()

    plt.figure(figsize=(10, 6))
    reconstruct_and_plot_image(num_components_95)
    plt.title(f"Реконструйоване зображення з {num_components_95} компонентами (PCA)")
    plt.axis('off')
    plt.show()
    ##print("____________________________________________________________")

    components_list = [5, 15, 25, 75, 100, 170]
    plt.figure(figsize=(15,9))
    for i in range(6):
        plt.subplot(2,3,i+1)
        reconstruct_and_plot_image(components_list[i])
        plt.title("Components: "+str(components_list[i]))
    plt.subplots_adjust(wspace=0.2, hspace=0.0) # Регулювання відстані між підграфіками
    plt.show()