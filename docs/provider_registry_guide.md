# Provider Registry Guide

`provider_registry.json` описывает, какие reasoning/thinking параметры нужно передавать OpenAI-compatible агрегаторам. Это нужно потому, что единого стандарта нет: один провайдер принимает `reasoning.effort`, другой `extra_body.reasoning.effort`, третий `reasoning_effort`, а некоторые возвращают `400`, если отправить неизвестное поле.

Текущий registry использует `schema_version: 1` и `data_version: 6`. В нём есть отдельные записи для NVIDIA NIM GPT-OSS, NVIDIA NIM DeepSeek V4, NVIDIA NIM thinking-моделей и Hugging Face/Qwen endpoint `hf_qwen38`; matching выполняется по hostname `OPENAI_BASE_URL`, а для reasoning-профилей дополнительно учитывается имя модели.

Registry применяется только к профилям с `provider: openai`. Gemini настраивается отдельно через Google SDK-поля `thinking_budget`, `thinking_level` и `include_thoughts`.

> Кастомные HTTP-заголовки для OpenAI-compatible запросов (эмуляция QwenCode и др.) настраиваются отдельно через `headers.json` — см. раздел «HTTP-заголовки» в [`CONFIGURATION.md`](./CONFIGURATION.md).

## Где находится файл

По умолчанию используется файл:

```text
provider_registry.json
```

Путь можно переопределить через `.env`:

```env
PROVIDER_REGISTRY_PATH=provider_registry.json
```

Runtime загружает registry в `create_llm()` перед созданием `ChatOpenAI`. Если `OPENAI_BASE_URL` не найден в registry или провайдер помечен как `supports_reasoning: false`, reasoning payload не добавляется.
По умолчанию `supports_reasoning: true` применяется на уровне всего provider entry. Для строгих API, где reasoning поддерживают не все модели, можно добавить `model_match`.

## Общая структура

```json
{
  "schema_version": 1,
  "data_version": 6,
  "providers": []
}
```

| Поле | Для чего |
|---|---|
| `schema_version` | Версия схемы. Меняется при несовместимых изменениях формата |
| `data_version` | Версия данных. Меняется при добавлении или правке провайдеров |
| `providers` | Список provider entries |

## Provider Entry

Минимальный пример агрегатора с `extra_body.reasoning.effort`:

```json
{
  "id": "example_gateway",
  "enabled": true,
  "priority": 80,
  "match": ["api.example.com"],
  "match_type": "suffix",
  "supports_reasoning": true,
  "validation": "map",
  "reasoning": {
    "path": "extra_body.reasoning.effort",
    "allowed_values": ["low", "medium", "high"],
    "value_map": {
      "minimal": "low",
      "xhigh": "high"
    }
  },
  "notes": "Reasoning object API"
}
```

| Поле | Обязательное | Для чего |
|---|---:|---|
| `id` | да | Уникальный идентификатор провайдера |
| `enabled` | да | `false` временно выключает entry без удаления |
| `priority` | да | Победитель при нескольких совпадениях; больше значит важнее |
| `match` | да | Hostname-паттерны без path, например `openrouter.ai` |
| `match_type` | да | `exact` или `suffix` |
| `supports_reasoning` | да | Если `false`, provider матчится, но reasoning-поля не добавляются |
| `model_match` | нет | Дополнительный фильтр моделей для строгих API; если отсутствует, reasoning включается для всего provider entry |
| `validation` | да | `strict`, `map` или `passthrough` |
| `reasoning` | если `supports_reasoning=true` | Настройка поля effort и дополнительных полей |
| `notes` | нет | Человеческая пометка, на runtime не влияет |

## Matching по `base_url`

Registry матчится по hostname из `OPENAI_BASE_URL`.

Примеры:

| `base_url` | Hostname | Может совпасть с |
|---|---|---|
| `https://openrouter.ai/api/v1` | `openrouter.ai` | `openrouter.ai` |
| `https://api.openrouter.ai/v1` | `api.openrouter.ai` | `openrouter.ai`, если `match_type: suffix` |
| `api.openai.com/v1?foo=bar` | `api.openai.com` | `api.openai.com` |
| `http://localhost:3002/v1` | `localhost` | не добавлять в registry |

`match_type`:

| Значение | Логика |
|---|---|
| `exact` | Hostname должен совпасть полностью |
| `suffix` | Совпадает сам hostname или его поддомен через точку |

Важно: `suffix` не является substring search. Паттерн `openrouter.ai` совпадёт с `api.openrouter.ai`, но не с `evil-openrouter.ai`.

## Reasoning Config

```json
{
  "path": "extra_body.reasoning.effort",
  "allowed_values": ["low", "medium", "high"],
  "value_map": {
    "minimal": "low",
    "xhigh": "high"
  },
  "extra_fields": {
    "extra_body.reasoning.summary": "auto"
  }
}
```

| Поле | Для чего |
|---|---|
| `path` | Dot-path, куда записать значение `MODEL_REASONING_EFFORT` |
| `allowed_values` | Допустимые значения после нормализации |
| `value_map` | Маппинг входного effort в значение, которое принимает провайдер; результат может быть строкой, boolean или числом |
| `enabled_value` | Опциональное typed-значение, которое отправляется при `reasoning.enabled: true` |
| `disabled_value` | Опциональное значение, которое нужно явно отправить при `reasoning.enabled: false` |
| `none_value` | Опциональное typed-значение для входного `effort: "none"`; если поле отсутствует, payload пропускается |
| `extra_fields` | Дополнительные постоянные поля, если они документированы провайдером |

`path` пишет в kwargs для `ChatOpenAI`, не напрямую в HTTP JSON. Например:

| `path` | Итоговый kwargs |
|---|---|
| `reasoning.effort` | `{"reasoning": {"effort": "high"}}` |
| `extra_body.reasoning.effort` | `{"extra_body": {"reasoning": {"effort": "high"}}}` |
| `reasoning_effort` | `{"reasoning_effort": "high"}` |

## Validation Modes

| Mode | Поведение |
|---|---|
| `strict` | Значение должно быть в `allowed_values`, иначе ошибка |
| `map` | Сначала применяет `value_map`, потом проверяет `allowed_values` |
| `passthrough` | Пишет значение как есть, без проверки |

Обычно нужен `map`: пользователь может поставить `MODEL_REASONING_EFFORT=xhigh`, а провайдер принимает только `high`.

Для бинарного provider-specific режима используются `enabled_value` и `disabled_value`; они передаются в payload как typed-значения без строкового effort. В UI такой entry отображается как `On`/`Off`.

Особый случай: `MODEL_REASONING_EFFORT=none` отправляет `none_value`, если provider entry его объявляет (например, GPT-5.6 и NVIDIA DeepSeek V4 явно передают `none` для Non-think режима); если `none_value` отсутствует, reasoning payload не добавляется. Явное `reasoning.enabled: false` отправляет `disabled_value`, только если provider entry его объявляет.

В UI профилей OpenAI-compatible пункт `Default` не является отдельным уровнем и не добавляется в меню. Пользователь выбирает только значения из `allowed_values`; профиль без сохранённого reasoning-выбора сохраняет глобальную runtime-настройку. Старые профили с `reasoning.enabled: false` продолжают отключать reasoning.

## Как выбрать `path`

Смотри документацию конкретного агрегатора.

Типовые варианты:

| Провайдерный формат | `path` | Примеры |
|---|---|---|
| Native OpenAI Responses API | `reasoning.effort` | OpenAI |
| OpenAI SDK `extra_body` | `extra_body.reasoning.effort` | OpenRouter |
| Top-level Chat Completions field | `reasoning_effort` | Ollama Cloud, Fireworks, Mistral |

> **NVIDIA NIM** в текущем registry разделён на два формата: GPT-OSS использует top-level `reasoning_effort`, остальные matching thinking-модели — `extra_body.chat_template_kwargs.enable_thinking`; подробности ниже.

Не угадывай поле по названию модели. Один и тот же model id через разные gateways может требовать разные параметры.

## Model Match

`model_match` нужен только там, где сам provider строгий и возвращает ошибку на reasoning-поля для неподдерживаемых моделей. Если `model_match` отсутствует, registry считает, что весь provider entry поддерживает выбранный reasoning format.

Пример для official OpenAI:

```json
{
  "model_match": {
    "prefix": ["gpt-5", "gpt-6", "o1", "o3", "o4"]
  }
}
```

Поддерживаемые правила:

| Поле | Логика |
|---|---|
| `exact` | Полное совпадение имени модели |
| `prefix` | Имя модели начинается с указанного значения |
| `contains` | Имя модели содержит указанное значение |

Для OpenAI-compatible агрегаторов предпочтительнее не добавлять `model_match`, если провайдер сам маршрутизирует reasoning-capable модели и безопасно игнорирует или валидирует параметры. Так новые модели не требуют правки Python-кода.

## Как добавить нового агрегатора

1. Найди в документации провайдера точный параметр reasoning/thinking.
2. Проверь, какие effort values поддерживаются: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` и т.д.
3. Добавь entry в `provider_registry.json`.
4. Если провайдер не документирует reasoning API, добавь entry с `supports_reasoning: false`.
5. Выбери `match_type: exact`, если должен матчиться только один hostname.
6. Выбери `match_type: suffix`, если у провайдера могут быть поддомены.
7. Не добавляй локальные endpoints (`localhost`, `127.0.0.1`) в registry.
8. Добавь `model_match` только если provider возвращает ошибки для части моделей.
9. Добавь или обнови тест в `tests/test_runtime_refactor.py` для expected kwargs.
10. Если меняешь validation/matching, добавь тест в `tests/test_provider_registry.py`.
11. Запусти тесты.

Команды:

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_provider_registry.py tests\test_runtime_refactor.py -p no:cacheprovider
.\venv\Scripts\python.exe -m pytest -p no:cacheprovider
```

## Примеры

### OpenAI

Official OpenAI API (`api.openai.com`, а также зеркала вроде `api.ashna.ai`) разбит на несколько записей с `model_match` и убывающими приоритетами:

```json
{
  "id": "openai_gpt6",
  "enabled": true,
  "priority": 103,
  "match": ["api.openai.com", "api.ashna.ai"],
  "match_type": "exact",
  "supports_reasoning": true,
  "model_match": {
    "prefix": ["gpt-6"]
  },
  "validation": "strict",
  "reasoning": {
    "path": "reasoning.effort",
    "allowed_values": ["low", "medium", "high", "xhigh", "max"]
  },
  "notes": "OpenAI Responses API. GPT-6 supports reasoning.effort=low|medium|high|xhigh|max."
}
```

Порядок выбора для `api.openai.com`: `openai_gpt6` (103, prefix `gpt-6`) → `openai_gpt56` (102, prefix `gpt-5.6`, strict, с `none_value` и `reasoning.summary: auto`) → `openai_gpt5_legacy` (100, prefix `gpt-5`, map) → `openai_o_series` (90, prefix `o1`/`o3`/`o4`). Модели вне этих семейств не получают reasoning-поля.

### OpenRouter

```json
{
  "id": "openrouter",
  "enabled": true,
  "priority": 90,
  "match": ["openrouter.ai"],
  "match_type": "suffix",
  "supports_reasoning": true,
  "validation": "map",
  "reasoning": {
    "path": "extra_body.reasoning.effort",
    "allowed_values": ["minimal", "low", "medium", "high", "xhigh"],
    "value_map": {}
  }
}
```

### Top-level `reasoning_effort`

```json
{
  "id": "example_reasoning_effort",
  "enabled": true,
  "priority": 80,
  "match": ["api.example.com"],
  "match_type": "exact",
  "supports_reasoning": true,
  "validation": "map",
  "reasoning": {
    "path": "reasoning_effort",
    "allowed_values": ["low", "medium", "high"],
    "value_map": {
      "minimal": "low",
      "xhigh": "high"
    }
  }
}
```

### Conservative Entry

Используй это, когда провайдер есть в профилях, но reasoning API не подтверждён:

```json
{
  "id": "unknown_gateway",
  "enabled": true,
  "priority": 60,
  "match": ["api.unknown.example"],
  "match_type": "exact",
  "supports_reasoning": false,
  "validation": "passthrough",
  "notes": "Conservative default: do not send undocumented reasoning fields."
}
```

## Частые ошибки

- Отправлять `reasoning_effort` провайдеру, который ждёт `reasoning.effort`.
- Класть OpenRouter reasoning не в `extra_body`, когда используется OpenAI SDK совместимый клиент.
- Добавлять `summary: auto` без подтверждения в документации провайдера.
- Добавлять `localhost` в registry. Локальные OpenAI-compatible серверы слишком разные; лучше не слать им provider-specific поля по умолчанию.
- Делать `suffix` слишком широким, например `ai` или `com`.
- Добавлять hardcoded model gate в Python-код. Provider-level поведение должно жить в `provider_registry.json`; для строгих API используй `model_match`.

## NVIDIA NIM: особенности reasoning

NVIDIA NIM hosted API (`integrate.api.nvidia.com`, `api.nvidia.com`) разделён в registry на три provider entry с разными payload:

1. **DeepSeek V4 (`nvidia_nim_deepseek_v4`, priority 72)** — применяется к моделям, содержащим `deepseek-v4`, включая `deepseek-ai/deepseek-v4-flash-0731`. Используется top-level `reasoning_effort`; NVIDIA документирует значения `none`, `high`, `max`. В отличие от общего правила для `none`, этот профиль явно отправляет `reasoning_effort: "none"`, чтобы выбрать режим Non-think.

2. **GPT-OSS (`nvidia_nim_gpt_oss`, priority 71)** — применяется только к моделям, содержащим `gpt-oss`. Используется top-level `reasoning_effort`; допустимые значения после mapping: `low`, `medium`, `high`. Входные `minimal` и `xhigh`/`max` преобразуются соответственно в `low` и `high`.

3. **Остальные документированные thinking-модели (`nvidia_nim`, priority 70)** — применяется к моделям, содержащим `deepseek-r1`, `deepseek-v3`, `deepseek-prover`, `glm-5`, `glm-4.7`, `kimi-k2`, `qwen3` или `gemma-4`. UI предоставляет бинарный выбор `On`/`Off`. Runtime передаёт настоящий JSON boolean через `extra_body.chat_template_kwargs.enable_thinking`: `true` для включения и `false` для явного выключения. Общий `clear_thinking` не отправляется, потому что NVIDIA не документирует его как универсальный параметр этих моделей.

Все три записи используют suffix matching для `integrate.api.nvidia.com` и `api.nvidia.com`. Приоритеты нужны, чтобы специализированная запись DeepSeek V4 или GPT-OSS побеждала общий NVIDIA entry для тех же hostname.

Reasoning в streaming может возвращаться через нестандартные top-level поля delta: `delta.reasoning`, `delta.reasoning_content`, `delta.thinking`. Если `reasoning_detected=false`, это не доказывает, что thinking не включён: провайдер может не возвращать reasoning tokens клиенту.

## Hugging Face / Qwen endpoint

Запись `hf_qwen38` с priority 65 точно сопоставляется с hostname `g9hnto0u7lvbu837.us-east-2.aws.endpoints.huggingface.cloud`. Она поддерживает `reasoning_effort` со значениями `low`, `medium`, `xhigh`; входные значения `minimal` и `low` → `low`, `medium` → `medium`, `high` и `xhigh` → `xhigh`.

Для этого endpoint OpenAI-compatible профиль должен использовать соответствующий `OPENAI_BASE_URL`. У записи нет `model_match`, поэтому поддержка reasoning применяется к любой модели, отправленной на этот конкретный endpoint. Registry не распространяет entry на другие Hugging Face endpoint hostname.
