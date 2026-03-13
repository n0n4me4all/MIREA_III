# HW08-09 – PyTorch MLP: регуляризация и оптимизация обучения

## 1. Кратко: что сделано

- Датасет: EMNIST (balanced, буквы/цифры, 47 классов, изображения 28×28).
- Часть A (S08): сравнение базовой MLP vs Dropout vs BatchNorm + выбор лучшего и EarlyStopping.
- Часть B (S09): диагностика слишком большого/маленького learning rate на кривых + SGD+momentum + weight decay.

## 2. Среда и воспроизводимость

- Python: 3.13.0
- torch / torchvision: 2.10.0+cu128 / 0.25.0+cu128
- Устройство (CPU/GPU): cuda
- Seed: 42
- Среда: запуск из venv (используем интерпретатор: `c:\Users\AA\VSC Projects\new_4\.venv\Scripts\python.exe`)
- Как запустить: открыть `HW08-09.ipynb` и выполнить Run All.

## 3. Данные

- Датасет: EMNIST (balanced split)
- Количество классов: 47 (цифры и буквы латинского алфавита)
- Размер изображений: 28×28
- Разделение: train/val/test (val отделён от train: 80/20; test — стандартный torchvision test split)
- Transform: ToTensor()

## 4. Базовая модель и обучение

- Модель MLP (кратко): Flatten - Linear/ReLU блоки - logits(47)
- Loss: CrossEntropyLoss
- Hidden sizes: [512, 256, 128]
- Базовый optimizer (часть A): Adam (lr=0.001)
- Batch size: 1024
- Epochs (макс): 20 (для E4 максимум 50 с EarlyStopping)
- EarlyStopping: patience=5, mode=max (val_accuracy)

## 5. Часть A (S08): регуляризация (E1-E4)

Опишите, что меняли. Формулировки должны быть короткими и сопоставимыми.

- E1 (base): без Dropout/BatchNorm
- E2 (Dropout): как E1 + Dropout(p=0.3)
- E3 (BatchNorm): как E1 + BatchNorm
- E4 (EarlyStopping): выбран лучший из (E2/E3) по val_accuracy + EarlyStopping

## 6. Часть B (S09): LR, оптимизаторы, weight decay (O1-O3)

- O1: LR слишком большой (Adam, lr=1e-1)
- O2: LR слишком маленький (Adam, lr=1e-5)
- O3: SSGD+momentum (momentum=0.9) + weight_decay=1e-4 (lr=5e-3)

## 7. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель: `./artifacts/best_model.pt`
- Конфиг лучшей модели: `./artifacts/best_config.json`
- Кривые лучшего прогона: `./artifacts/figures/curves_best.png`
- Кривые “плохих LR”: `./artifacts/figures/curves_lr_extremes.png`

Короткая сводка (5-9 строк):

- Лучший эксперимент части A: (E2/E3/E4)
- Лучшая val_accuracy:
- Итоговая test_accuracy (для лучшей модели):
- Что видно на O1 (слишком большой LR):
- Что видно на O2 (слишком маленький LR):
- Как повёл себя O3 (SGD+momentum + weight decay) относительно Adam (по кривым/метрике):

## 8. Анализ

(8-15 предложений)

- Задача классификации EMNIST сложнее, чем KMNIST, из-за большего числа классов (47).
- Некоторые символы визуально похожи (например O/0, l/I), что усложняет обучение модели.
- В E1 обычно наблюдается более сильное переобучение: train_accuracy растёт быстрее, чем val_accuracy.
- Dropout (E2) уменьшает переобучение за счёт случайного отключения нейронов.
- BatchNorm (E3) стабилизирует распределение активаций и может ускорить обучение.
- EarlyStopping (E4) предотвращает дальнейшее переобучение после достижения плато по val_accuracy.
- При слишком большом learning rate (O1) обучение становится нестабильным.
- При слишком маленьком learning rate (O2) обучение практически не прогрессирует.
- SGD+momentum с weight decay (O3) даёт более гладкие кривые обучения и служит хорошей альтернативой Adam.

## 9. Итоговый вывод

(3-7 предложений)

Для датасета EMNIST (balanced) MLP из {BASE_HIDDEN} способен обучаться
и достигать разумной точности, однако задача сложнее, чем KMNIST,
из-за увеличенного числа классов.

Использование регуляризации (Dropout или BatchNorm) и EarlyStopping
помогает уменьшить переобучение.

Для дальнейшего улучшения результатов можно:
- увеличить размер модели,
- использовать scheduler для learning rate,
- попробовать сверточные сети (CNN), которые лучше подходят
  для обработки изображений.

## 10. Приложение (опционально)

