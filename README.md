# Лабораторна робота №2



### Завдання 1: Обчислення власних значень та власних векторів матриці (0.5 б.).
***Опис:*** Напишіть функцію, яка приймає квадратну матрицю і повертає її власні значення та власні вектори, використовуючи бібліотеку NumPy. Також здійсніть перевірку рівності A⋅ v=λ⋅ v для кожного власного значення та відповідного власного вектора.

----------------------------
### Завдання 2. Реалізація методу зменшення розмірності зображення за допомогою аналізу головних компонент (PCA: Image Compression) (4 б.).

***Опис:*** Реалізувати функцію для зменшення розмірності зображення за допомогою аналізу головних компонент (PCA).

Необхідні бібліотеки для виконання завдання: NumPy, matplotlib, Scikit-learn або їх аналоги.

**Алгоритм виконання завдання:**

1. Вивести початкове кольорове зображення та вектор, що буде містити: розміри зображення в пікселях та кількість основних каналів кольорів, що використовуються

2. Перетворити зображення в чорно-біле та вивести розмір зображення і кількість каналів кольорів **(0.25 б.)**

3. Застосувати PCA для матриці компонентів `image_bw`. Вивести cumulative variance та знайти кількість компонент, які необхідні для покриття 95% of the variance. Використовуйте бібліотечні засоби (рекомендовано NumPy) **(0.75 б.).**

   Вивести графік відповідного процесу: **додаткові 0.5 б.**

4. Провести реконструкцію чорно-білого зображення, використовуючи обмежену кількість компонентів, знайдену в попередньому кроці. Вивести отримане зображення. Для 95% покриття даних очікувано отримати більш чітке зображення, чи не так? Зауважте, що ми точно зафіксували всі основні елементи – ви все ще можете дуже добре ідентифікувати об’єкти. Чого не вистачає, так це чіткості — та, можливо, саме дрібні деталі у візуальних елементах роблять зображення привабливим і чітким. **1б.**

5. Проведіть реконструкцію зображення для різної кількості компонент та виведіть відповідні результати. Спробуйте взяти більшу кількість компонент та виведіть відповідний результат. Чи отримали ви більш чітке зображення? А якщо взяти меншу кількість компонент? **(2 б).**
   


----------------------------

### Завдання 3: Використання діагоналізації, власних значень та векторів в криптографії.

***Опис:*** Використайте діагоналізацію для розшифрування кодів**.** Рекомендується використовувати бібліотеку Numpy. Корисні посилання: [Короткий опис як можна ](https://www.youtube.com/watch?v=S_2MV3ncHj0)застосовувати принципи лінійної алгебри в криптографії,[відео про саму ](https://www.youtube.com/watch?v=-yFZGF8FHSg)криптографію **1.5 б.**

**Завдання:**

1. Створити функцію `decrypt_message(encrypted_vector, key_matrix)`, яка розшифровує зашифрований вектор `encrypted_vector` за допомогою матриці ключа `key_matrix`, використовуючи обернену операцію діагоналізації.
2. Функція кодування задається таким чином:

 ![image](https://github.com/Anna-Yevtushenko/Laboratory_work-2/assets/150729768/a263f8b4-acbe-474b-b869-fbceb436dc91)


3. Перевірити роботу розроблених функцій на прикладі випадково згенерованої матриці ключа та текстового повідомлення.

![image](https://github.com/Anna-Yevtushenko/Laboratory_work-2/assets/150729768/103e2f52-f235-4eac-8479-cd47552f9978)


   **Приклад виконання:**

   > Original Message: Hello, World!

   > Encrypt Message: [118703.+1.04957181e-11j 180926.+1.50897913e-11j 149312.+1.90968912e-11j 161873\.+1.52982299e-11j 188078.+1.70487245e-12j 145036.+1.64536004e-11j 139370.+4.15241913e-11j 195037.+1.51003306e-11j 155629.+1.38144125e-11j 206859.+1.38439864e-11j 145588.+1.95891393e-11j 130163.+1.55598216e-11j 176423.+1.26512259e-11j]

   > Decrypted Message: Hello, World!

-----------------------------------
### Теоретичні питання 2б.:

1. Що таке власне значення і власний вектор матриці? Як вони обчислюються? **0.25 б.**
2. Які властивості мають власні вектори симетричних матриць? **0.25 б.**
3. Які можуть бути недоліки використання PCA, і які стратегії можуть використовуватися для подолання цих недоліків? **0.75 б**.
4. Які переваги має діагоналізація матриці в криптографії? Як вона застосовується для шифрування та дешифрування повідомлень? **0.75 б.**


