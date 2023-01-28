# python-flask-docker
Итоговый проект курса "Машинное обучение в бизнесе"

Стек:

ML: sklearn, pandas, numpy
API: flask
Данные: с kaggle - https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data

Задача: предсказать по медицинским характеристикам вероятность смертельного случая.

Используемые признаки:

- age - numeric
- anaemia - numeric
- creatinine_phosphokinase - numeric
- diabetes - numeric
- ejection_fraction - numeric
- high_blood_pressure - numeric
- platelets - numeric
- serum_creatinine - numeric
- serum_sodium - numeric
- sex - numeric
- smoking - numeric
- time - numeric

Преобразования признаков: numeric

Модель: logreg

### Клонируем репозиторий и создаем образ
```
$ git clone https://github.com/Kubyshina/MLinBusiness.git
$ cd MLinBusiness
$ docker build -t kubyshina/gb_docker_home_work .
```

### Запускаем контейнер

Здесь Вам нужно создать каталог локально и сохранить туда предобученную модель (<your_local_path_to_pretrained_models> нужно заменить на полный путь к этому каталогу)
```
$ docker run -d -p 8180:8180 -p 8181:8181 -v <your_local_path_to_pretrained_models>:/app/app/models kubyshina/gb_docker_home_work
```

### Переходим на localhost:8181
