# COCOM v11: История развития

Этот документ кратко описывает эволюцию архитектуры COCOM от версии 4 до версии 11.

---

## Хронология версий

| Версия | Ключевое нововведение | Ограничение |
|--------|----------------------|-------------|
| **v4** | Sequence-based обучение с temporal learning | Общие QKV веса между экспертами |
| **v5** | Majority voting + CRF память | CRF требует последовательности, не универсален |
| **v6** | Learnable Memory (class_keys) | Memory как отдельный компонент |
| **v7** | Memory как арбитр, градиент → только победителю | Gradient dilution для остальных |
| **v8** | ConfidenceArbitrator с entropy-based взвешиванием | Всё ещё gradient dilution |
| **v9** | EMA Tactical с FFT (O(N log N)) | EMA требует warmup |
| **v10** | Staged sequential training | Ручное переключение стадий |
| **v11** | RMSNorm, SwiGLU, RoPE, OneCycleLR | **Текущая версия** |

---

## Ключевые инсайты из разработки

### 1. Независимые эксперты критичны
Ранние версии (v3-v4) использовали общие QKV веса — эксперты были математически идентичны. v5+ использует полностью независимые веса.

### 2. Эволюция роли Memory/Arbitrator
- v5: CRF записывает только консенсус → учит надёжные паттерны
- v7: Memory арбитрирует при разногласии экспертов
- v8+: Confidence-weighted комбинация + собственный классификатор

### 3. EMA для Tactical эксперта
v9 заменил window attention на Multi-Scale EMA:
- O(N log N) через FFT convolution
- Лучше для последовательных паттернов
- Требует правильный warmup (MEGA-style)

### 4. Staged Training
v10 ввёл последовательное обучение:
1. Tactical + Embeddings (warmup критичен)
2. Strategic (замороженные embeddings)
3. Exploratory (замороженные embeddings)
4. Full model (всё обучаемо)

---

## Сравнение архитектур

```
v4-v6: Общие QKV → все эксперты похожи
v7+:   Независимые эксперты → истинное разнообразие
v9:    Tactical = EMA (не attention)
v11:   Современные слои (RMSNorm, SwiGLU, RoPE)
```

---

## Ссылки

Для подробной технической документации см.:
- [guide_rus.md](guide_rus.md) — Полное техническое руководство v11
- [cocom_math_foundation_rus.md](cocom_math_foundation_rus.md) — Теоретические основы
