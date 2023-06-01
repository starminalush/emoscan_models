mfdp-2023
==============================

Emotional recognition for online schools

Используемые датасеты
---------------------
<p><a target="_blank" href="https://www.kaggle.com/datasets/noamsegal/affectnet-training-data">AffectNet</a></p>
<p><a target="_blank" href="https://www.kaggle.com/datasets/msambare/fer2013">Fer2013</a></p>

DVC pipeline
------------
![image](https://github.com/starminalush/mfdp-2023/assets/103132748/81fd0261-e8d3-4359-8c60-d0eb456b1d8a)

Эксперименты
---------------

| Модель | Датасет | git tag | F1 мера | Latency (batch=1) | Throughtput (batch=5) | Вывод |
| --- | --- | --- | --- | --- | --- | --- |
| pretrained ResNet18 | FER2013 |  v1.0 | 0.67 | 0.012 | 918.7 | Получен бейзлайн |
| pretrained ResNet18 | AffectNet-8 |  v1.1 | 0.70 | 0.011 | 916.6 | Улучшение качества и незначительное ухудшение пропускной способности|
| pretrained ResNet18 | AffectNet-7 |  v1.2 | 0.712 | 0.013 | 913.9 | Улучшение качества и  незначительное ухудшение пропускной способности|
| pretrained DAN(RafDB) | AffectNet-7 |  v1.3 | 0.718 | 0.018 | 524.9 | Улучшение качества и  незначительное ухудшение времени инференса. Сильное время ухудшения пропускной способности, но все еще приемлимое|
| pretrained mobilenet_v3_small | AffectNet-7 |  v1.4 | 0.412 | 0.02 | 1254.1| Качество стало хуже. Мердж в мастер отклонен|


Запуск обучения на своих данных
---------

1. Настроить конфиг .env
2. Создать конфиг по примеру experiments/configs/fer/train.yaml. Указать импорты для нужных модулей
3. Запустить контейнер train_models
