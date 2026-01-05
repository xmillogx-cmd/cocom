# COCOM v11: Результаты обучения ListOps

## Сводная таблица

| Эксперимент | Group Size | Seq Len | Batch Size | Режим обучения | Лучшая Acc | Params | Заметки |
|-------------|------------|---------|------------|----------------|------------|--------|---------|
| v11 tokens | 1 | 2048 | 16 | Staged (T→S→E→Full) | 40.55% (Arb) | 3.4M | Коллапс на full stage |
| v11 grouped-8 | 8 | 256 | 128 | Staged | ~40% | 3.5M | 10x быстрее |
| v11 grouped-32 | 32 | 64 | 1024 | Staged | 38.25% | 5.0M | Early stopping на full |
| v11 full-only | 8 | 256 | 256 | Full only (30 epochs) | 37.65% | 3.5M | Коллапс epoch 6-8, early stop 9 |
| **CIFAR-10 2x2** | 2x2 patch | 256 | 256 | Staged | **68.77%** | 2.96M | Strategic лучший (63.6%), Arb=69.2% |
| **LRA Text** | 16 | 256 | 128 | Staged | **69.36%** | 4.1M | Tactical лучший (68.7%), +5% vs Transformer |
| **Pathfinder-32** | patch=4 | 64 | 32 | Staged | **65.33%** | ~3M | Exploratory лучший (66.9%), Arb=67.4% |

## Детальные результаты

### Эксперимент 1: v11 tokens (group=1)
- **Конфиг:** input_type='tokens', seq_len=2048, batch=16
- **Результаты:**
  - Strategic: 37.05%
  - Tactical: 38.00%
  - Exploratory: 39.10% ← Лучший эксперт
  - Arbitrator: 40.55%
  - Final: 39.85%
- **Наблюдения:**
  - Коллапс на full stage (17.8% позже)
  - Доминирование классов 0 и 9 (80%, 78%)
  - Остальные классы: 10-20%

### Эксперимент 2: v11 grouped-8
- **Конфиг:** input_type='tokens_conv1d', kernel=8, seq_len=256, batch=128
- **Результаты (staged):**
  - Тактик stage best: 36.45%
  - Strategic stage best: 39.30%
  - Exploratory stage best: 39.80%
- **Наблюдения:**
  - Та же точность как tokens, 10x быстрее
  - Arb улучшается с экспертами (frozen но input меняется)

### Эксперимент 3: v11 grouped-32
- **Конфиг:** input_type='tokens_conv1d', kernel=32, seq_len=64, batch=1024
- **Результаты:**
  - Tactical: 37.55%
  - Strategic: 37.90%
  - Exploratory: 38.25% ← Лучший
  - Full: 37.25% (деградация)
- **Наблюдения:**
  - Early stopping на full (epoch 6)
  - Очень быстрое обучение (~40x vs tokens)
  - На 2% ниже чем group=8

## Ключевые находки

### 1. Инвариантность к токенизации
- Все размеры групп достигают ~40% точности
- Узкое место — архитектура, не токенизация
- Скорость обучения масштабируется с компрессией

### 2. Преимущества Staged Training
- Каждый эксперт находит локальный оптимум
- Арбитратор наследует улучшения экспертов
- Нет интерференции при раннем обучении

### 3. Проблемы Full Stage
- Склонность к переобучению/коллапсу
- Early stopping помогает
- Возможно нужен ниже LR или регуляризация

## Переменные для корреляционного анализа

| Переменная | Протестированные значения |
|------------|--------------------------|
| group_size | 1, 8, 32 |
| batch_size | 16, 128, 512, 1024 |
| training_mode | staged, full-only |
| seq_len | 64, 256, 2048 |
| lr_tactical | 0.003 |
| lr_full | 3e-4 |
| epochs_per_stage | 5-8 |
| warmup | True/False |

## Гипотезы для проверки

1. **H1:** accuracy ~ константа для разных group_sizes (подтверждено ~40%)
2. **H2:** training_time ~ 1/group_size² (attention O(N²))
3. **H3:** staged > full-only для финальной точности
4. **H4:** arbitrator_acc >= max(expert_acc)
5. **H5:** collapse_probability ~ epochs в full stage

## Следующие эксперименты

- [ ] Full-only с group=8, batch=512
- [ ] Transfer: train group=32 → adapt to group=8
- [ ] Multi-layer experts (2-4 слоя)
- [ ] Сравнение с S4/MEGA при том же param count
