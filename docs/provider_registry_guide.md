# Provider Registry Guide

`provider_registry.json` описывает, какие reasoning/thinking параметры нужно передавать OpenAI-compatible агрегаторам. Это нужно потому, что единого стандарта нет: один провайдер принимает `reasoning.effort`, другой `extra_body.reasoning.effort`, третий `reasoning_effort`, а некоторые возвращают `400`, если отправить неизвестное поле.

Текущий registry использует `schema_version: 2`. Matching выполняется по hostname из `OPENAI_BASE_URL`, а внутри провайдера конкретное правило дополнительно выбирается по имени модели.

Registry применяется только к профилям с `provider: openai`. Gemini настраивается отдельно через Google SDK-поля `thinking_budget`, `thinking_level` и `include_thoughts`, а Anthropic — через свои thinking-параметры.

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

Runtime загружает registry в `create_llm()` перед созданием `ChatOpenAI`. Если hostname из `OPENAI_BASE_URL` не найден в registry, либо у совпавшего провайдера нет правила под текущую модель, либо у провайдера вообще нет `rules`, reasoning payload не добавляется.

## Идея схемы

Один провайдер = один объект. Внутри — **упорядоченный список `rules`**: первое правило, у которого совпали host **и** модель, побеждает. Порядок в массиве — это и есть приоритет; отдельного числового `priority` нет. Специфичные правила ставятся выше общих fallback-правил.

```json
{
  "schema_version": 2,
  "data_version": 1,
  "providers": []
}
```

| Поле | Для чего |
|---|---|
| `schema_version` | Версия схемы. Меняется при несовместимых изменениях формата |
| `data_version` | Версия данных. Меняется при добавлении или правке провайдеров |
| `providers` | Список provider-объектов |

## Provider

```json
{
  "id": "openai",
  "hosts": ["api.openai.com", "api.ashna.ai"],
  "rules": [
    {
      "models": { "prefix": ["gpt-6"] },
      "param": "reasoning.effort",
      "values": { "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max" }
    },
    {
      "models": { "prefix": ["gpt-5"] },
      "param": "reasoning.effort",
      "values": { "none": "none", "minimal": "minimal", "low": "low", "medium": "medium", "high": "high" },
      "extra": { "reasoning.summary": "auto" }
    }
  ]
}
```

| Поле | Обязательное | По умолчанию | Для чего |
|---|---:|---|---|
| `id` | да | — | Уникальный идентификатор провайдера |
| `hosts` | да | — | Hostname-паттерны без path, например `openrouter.ai` |
| `match_type` | нет | `exact` | `exact` или `suffix` |
| `enabled` | нет | `true` | `false` временно выключает провайдера без удаления |
| `rules` | нет | `[]` | Упорядоченный список правил; пустой список = reasoning не отправляется |
| `notes` | нет | — | Человеческая пометка, на runtime не влияет |

Провайдер **без `rules`** (или с `"rules": []`) означает «провайдер известен, но reasoning API не подтверждён»: host матчится, но никакие reasoning-поля не добавляются.

```json
{ "id": "unknown_gateway", "hosts": ["api.unknown.example"],
  "notes": "Reasoning API not confirmed — do not send undocumented reasoning fields." }
```

## Rule

| Поле | Обязательное | По умолчанию | Для чего |
|---|---:|---|---|
| `models` | нет | совпадает с любой моделью | Фильтр моделей: `exact`, `prefix`, `contains` |
| `mode` | нет | `effort` | `effort` (градация) или `toggle` (вкл/выкл) |
| `param` | да | — | Dot-path, куда записать значение |
| `values` | для `mode: effort` | — | Словарь «входной effort → значение провайдера» |
| `toggle` | для `mode: toggle` | — | `{ "on": ..., "off": ... }` |
| `extra` | нет | `{}` | Дополнительные постоянные поля |
| `notes` | нет | — | Человеческая пометка |

### `param` и `extra`

`param` и ключи `extra` — это dot-path, который во время выполнения превращается во вложенные kwargs для `ChatOpenAI` (не в HTTP JSON напрямую):

| `param` | Итоговый kwargs |
|---|---|
| `reasoning.effort` | `{"reasoning": {"effort": "high"}}` |
| `extra_body.reasoning.effort` | `{"extra_body": {"reasoning": {"effort": "high"}}}` |
| `reasoning_effort` | `{"reasoning_effort": "high"}` |

`extra` добавляется вместе с основным `param`, когда правило срабатывает (для `toggle` — только при `on`). Пример: DeepSeek V4 дополнительно шлёт `extra_body.thinking.type = enabled`.

### `values` (effort-режим)

`values` — словарь «входной effort → значение для провайдера». Ключи — это уровни `MODEL_REASONING_EFFORT` (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`), значения — то, что реально уходит провайдеру. Значение может быть строкой, boolean или числом.

Если входной effort **не встречается** как ключ в `values` — reasoning-поле не отправляется (тихо пропускается). Это правило одинаково работает для любого уровня, включая `none`.

```json
{
  "param": "reasoning_effort",
  "values": { "minimal": "low", "low": "low", "medium": "high", "high": "high", "xhigh": "max", "max": "max" }
}
```

### `toggle` (toggle-режим)

Часть провайдеров не принимают градацию — у них бинарный переключатель thinking on/off. Для них `mode: toggle`, а значения берутся из `toggle.on`/`toggle.off` в зависимости от `reasoning.enabled`. В UI такой провайдер отображается как `On`/`Off`.

```json
{ "models": { "prefix": ["kimi-k2.6", "kimi-k2.5"] }, "mode": "toggle",
  "param": "extra_body.thinking.type",
  "toggle": { "on": "enabled", "off": "disabled" } }
```

## Matching по `base_url`

Registry матчится по hostname из `OPENAI_BASE_URL`.

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

Порядок resolve на runtime:

1. Найти **первый** провайдер, у которого совпал host. На нём и останавливаемся.
2. Внутри него пройти `rules` сверху вниз и вернуть **первое** правило, совпавшее по модели (правило без `models` совпадает с любой моделью).
3. Если host совпал, но ни одно правило не подошло по модели — reasoning не отправляется.

`models`:

| Поле | Логика |
|---|---|
| `exact` | Полное совпадение имени модели |
| `prefix` | Имя модели начинается с указанного значения |
| `contains` | Имя модели содержит указанное значение |

## Как значение effort превращается в payload

- **effort-режим:** входной `MODEL_REASONING_EFFORT` ищется среди ключей `values`. Если найден — пишется соответствующее значение по `param` плюс `extra`. Если не найден (включая `none`) — reasoning-поле не добавляется. При `reasoning.enabled: false` effort-правило не отправляет ничего.
- **toggle-режим:** при `reasoning.enabled: true` пишется `toggle.on`, при `false` — `toggle.off`.

Отдельного режима валидации нет: неизвестный effort не приводит к ошибке, он просто молча игнорируется. Если важно ловить неподдерживаемый effort как явную ошибку, это стоит делать не в registry, а отдельной проверкой в коде загрузки профиля.

В UI профилей OpenAI-compatible для effort-правил показываются различные значения провайдера (уникальные значения `values`, в порядке первого появления); для toggle-правил — `On`/`Off`. Пункт `Default` не является отдельным уровнем и в меню не добавляется. Профиль без сохранённого reasoning-выбора использует глобальную runtime-настройку; старые профили с `reasoning.enabled: false` продолжают отключать reasoning.

## Как добавить нового агрегатора

1. Найди в документации провайдера точный параметр reasoning/thinking и допустимые значения.
2. Есть ли уже provider-объект с таким `hosts`? Если да — добавь новое правило в его `rules` (специфичные `models`-правила выше общих). Если нет — создай новый provider-объект.
3. Определи `mode`: если провайдер принимает градацию (`low/medium/high/…`) — `effort` (по умолчанию, можно не писать); если только on/off — `toggle`.
4. Заполни `values` (или `toggle.on`/`toggle.off`) только для задокументированных значений. Незаполненное — не отправляется.
5. Если провайдер не документирует reasoning API — добавь provider-объект без `rules` с пояснением в `notes`.
6. Выбери `match_type: suffix` только там, где у провайдера бывают поддомены (по умолчанию `exact`).
7. Не добавляй локальные endpoints (`localhost`, `127.0.0.1`) в registry.
8. Обнови/добавь тест на итоговые kwargs.

Команды:

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_provider_registry.py tests\test_model_profiles.py -p no:cacheprovider
.\venv\Scripts\python.exe -m pytest -p no:cacheprovider
```

## Примеры

### OpenAI

Official OpenAI API (`api.openai.com`, а также зеркала вроде `api.ashna.ai`) — один provider-объект с несколькими правилами; специфичные (`gpt-6`, `gpt-5.6`) стоят выше общего fallback (`gpt-5`, o-серия):

```json
{
  "id": "openai",
  "hosts": ["api.openai.com", "api.ashna.ai"],
  "rules": [
    { "models": { "prefix": ["gpt-6"] }, "param": "reasoning.effort",
      "values": { "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max" } },
    { "models": { "prefix": ["gpt-5"] }, "param": "reasoning.effort",
      "values": { "none": "none", "minimal": "minimal", "low": "low", "medium": "medium", "high": "high" },
      "extra": { "reasoning.summary": "auto" } }
  ]
}
```

### OpenRouter

```json
{
  "id": "openrouter",
  "hosts": ["openrouter.ai"],
  "match_type": "suffix",
  "rules": [
    { "param": "extra_body.reasoning.effort",
      "values": { "none": "none", "minimal": "minimal", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh" } }
  ]
}
```

### Top-level `reasoning_effort`

```json
{
  "id": "example_reasoning_effort",
  "hosts": ["api.example.com"],
  "rules": [
    { "param": "reasoning_effort",
      "values": { "minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "high" } }
  ]
}
```

### Conservative Entry

Используй это, когда провайдер есть в профилях, но reasoning API не подтверждён:

```json
{
  "id": "unknown_gateway",
  "hosts": ["api.unknown.example"],
  "notes": "Conservative default: do not send undocumented reasoning fields."
}
```

## Частые ошибки

- Отправлять `reasoning_effort` провайдеру, который ждёт `reasoning.effort`.
- Класть OpenRouter reasoning не в `extra_body`, когда используется OpenAI SDK совместимый клиент.
- Добавлять `summary: auto` без подтверждения в документации провайдера.
- Добавлять `localhost` в registry. Локальные OpenAI-compatible серверы слишком разные; лучше не слать им provider-specific поля по умолчанию.
- Делать `suffix` слишком широким, например `ai` или `com`.
- Добавлять hardcoded model gate в Python-код. Provider-level поведение должно жить в `provider_registry.json`; для строгих API используй `models` внутри правила.

## NVIDIA NIM: особенности reasoning

NVIDIA NIM hosted API (`integrate.api.nvidia.com`, `api.nvidia.com`, `match_type: suffix`) — один provider-объект `nvidia_nim` с тремя правилами; специфичные стоят выше общего:

1. **DeepSeek V4** (`models.contains: ["deepseek-v4"]`) — top-level `reasoning_effort` со значениями `none`, `high`, `max`. В отличие от других правил, `none` явно отправляет `reasoning_effort: "none"`, чтобы выбрать режим Non-think.
2. **GPT-OSS** (`models.contains: ["gpt-oss"]`) — top-level `reasoning_effort`; входные значения нормализуются к `low`, `medium`, `high`.
3. **Остальные thinking-модели** (`deepseek-r1`, `deepseek-v3`, `deepseek-prover`, `glm-5`, `glm-4.7`, `kimi-k2`, `qwen3`, `gemma-4`) — `mode: toggle`, boolean через `extra_body.chat_template_kwargs.enable_thinking` (`true`/`false`). Универсальный `clear_thinking` не отправляется, потому что NVIDIA не документирует его.

Reasoning в streaming может возвращаться через нестандартные top-level поля delta (`delta.reasoning`, `delta.reasoning_content`, `delta.thinking`). Если `reasoning_detected=false`, это не доказывает, что thinking не включён: провайдер может не возвращать reasoning tokens клиенту.

## Hugging Face / Qwen endpoint

Provider `hf_qwen38` точно (`exact`) сопоставляется с hostname `g9hnto0u7lvbu837.us-east-2.aws.endpoints.huggingface.cloud` и поддерживает `reasoning_effort` (`values`: `minimal`/`low` → `low`, `medium` → `medium`, `high`/`xhigh` → `xhigh`). У правила нет `models`, поэтому reasoning применяется к любой модели на этом конкретном endpoint; на другие Hugging Face endpoint-хосты запись не распространяется.
