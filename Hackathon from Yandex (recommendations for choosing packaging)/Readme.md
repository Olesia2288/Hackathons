**Общее описание задачи:**

Для того, чтобы пользователь получил
заказ, необходимо упаковать заказанные товары в посылки конечному клиенту.
Компания заметила, что сотрудник тратит большое количество времени для
выбора упаковочного материала в который необходимо упаковать товары.
Существует большое количество упаковочного материала (коробочки,
пакетики). Необходимо придумать способ подсказывать пользователю
информацию о выборе упаковочного материала

**Решение задачи:**

Разработана модель на основе исторических данных о выборе пользователя, также добавлены рекомендации на основе имеющихся сведений о размерах упаковки и сведений о типах товаров.

**Описание файлов в папке:**
- [Описание решения задачи по выбору упаковки.ipynb](https://github.com/Olesia2288/Hackathons/blob/main/Hackathon%20from%20Yandex%20(recommendations%20for%20choosing%20packaging)/%D0%9E%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5%20%D1%80%D0%B5%D1%88%D0%B5%D0%BD%D0%B8%D1%8F%20%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B8%20%D0%BE%20%D0%B2%D1%8B%D0%B1%D0%BE%D1%80%D0%B5%20%D1%83%D0%BF%D0%B0%D0%BA%D0%BE%D0%B2%D0%BA%D0%B8.ipynb) - представлен ход решения и анализ данных, выводы по результатам исследования,
- [model.py](https://github.com/Olesia2288/Hackathons/blob/main/Hackathon%20from%20Yandex%20(recommendations%20for%20choosing%20packaging)/model.py) - код с решением, используемый для приложения по выбору упаковки,
- [test.py](https://github.com/Olesia2288/Hackathons/blob/main/Hackathon%20from%20Yandex%20(recommendations%20for%20choosing%20packaging)/test.py) - тестовые данные,
- [app.py](https://github.com/Olesia2288/Hackathons/blob/main/Hackathon%20from%20Yandex%20(recommendations%20for%20choosing%20packaging)/app.py) - файл приложения, обрабатывающего запрос,
- [requirements.txt](https://github.com/Olesia2288/Hackathons/blob/main/Hackathon%20from%20Yandex%20(recommendations%20for%20choosing%20packaging)/requirements.txt) - список пакетов и их версий, которые требуются для выполнения программного кода,
- [file_item.pcl](https://github.com/Olesia2288/Hackathons/blob/main/Hackathon%20from%20Yandex%20(recommendations%20for%20choosing%20packaging)/file_item.pcl) - модель,
- [encoder.pkl](https://github.com/Olesia2288/Hackathons/blob/main/Hackathon%20from%20Yandex%20(recommendations%20for%20choosing%20packaging)/encoder.pkl) - файл для кодировки(раскодировки) данных.
