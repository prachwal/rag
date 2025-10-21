# OpenRouter Integration - Quick Start

## Overview

Moduł OpenRouter zapewnia zunifikowany dostęp do wielu modeli LLM (Large Language Models) od różnych dostawców poprzez jedno API. Obsługuje modele od OpenAI, Anthropic, Google, Meta i innych.

## Architektura

Moduł OpenRouter został zaprojektowany zgodnie z architekturą RAG Application:

```
Common/services/
├── llm_provider_base.py     # Abstrakcyjne interfejsy dla LLM providers
├── openrouter_service.py    # Implementacja OpenRouter
└── huggingface_service.py   # Implementacja HuggingFace

rag_cli/commands/
└── openrouter.py            # Komendy CLI dla OpenRouter

Common/tests/
└── test_openrouter_service.py  # Testy jednostkowe
```

### Wspólny Interfejs LLM Providers

Wszystkie serwisy LLM implementują abstrakcyjne interfejsy z `llm_provider_base.py`:

- **LLMProviderBase** - podstawowy interfejs dla generowania tekstu
- **ChatProviderBase** - interfejs dla konwersacji/czatu
- **EmbeddingProviderBase** - interfejs dla embeddingów

## Konfiguracja

### 1. Dodaj klucz API do `.env`

```bash
# OpenRouter API Settings
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_API_TIMEOUT=30
```

### 2. Zdobądź klucz API

1. Zarejestruj się na https://openrouter.ai
2. Przejdź do https://openrouter.ai/keys
3. Wygeneruj nowy klucz API
4. Skopiuj klucz do pliku `.env`

## Użycie w CLI

### Test połączenia

```bash
python3 rag_cli.py openrouter test
```

Wyjście:
```
✅ Connected to OpenRouter API (337 models available)
⏱️  Response time: 0.16s
🔐 Authenticated: True
📊 API version: v1
```

### Lista dostępnych modeli

```bash
# Wszystkie modele
python3 rag_cli.py openrouter models

# Pierwsze 10 modeli
python3 rag_cli.py openrouter models --limit 10

# Wyszukiwanie modeli
python3 rag_cli.py openrouter models --search "gpt"
python3 rag_cli.py openrouter models --search "claude"

# Wyjście JSON
python3 rag_cli.py openrouter models --json
```

### Generowanie tekstu

```bash
# Podstawowe użycie
python3 rag_cli.py openrouter generate "Write a haiku about AI"

# Wybór modelu
python3 rag_cli.py openrouter generate --model anthropic/claude-2 "Explain quantum computing"

# Zaawansowane parametry
python3 rag_cli.py openrouter generate \
  --model openai/gpt-4 \
  --max-tokens 500 \
  --temperature 0.9 \
  "Write a creative story"

# Wyjście JSON
python3 rag_cli.py openrouter generate --json "Test prompt"
```

### Interaktywny czat

```bash
# Podstawowy czat
python3 rag_cli.py openrouter chat

# Czat z wybranym modelem
python3 rag_cli.py openrouter chat --model anthropic/claude-2

# Czat z systemowym promptem
python3 rag_cli.py openrouter chat \
  --system "You are a helpful coding assistant" \
  --model openai/gpt-4

# Kontrola parametrów
python3 rag_cli.py openrouter chat \
  --model openai/gpt-3.5-turbo \
  --max-tokens 150 \
  --temperature 0.7
```

W trybie czatu:
- Wpisz wiadomość i naciśnij Enter
- Wpisz `quit`, `exit` lub `q` aby zakończyć
- Użyj Ctrl+C lub Ctrl+D aby przerwać

## Użycie w kodzie Python

### Import

```python
from Common.services.openrouter_service import openrouter_service
```

### Sprawdzanie dostępności

```python
if openrouter_service.is_available():
    print("OpenRouter jest skonfigurowany")
else:
    print("Ustaw OPENROUTER_API_KEY w .env")
```

### Test połączenia

```python
result = openrouter_service.test_connection()
print(result)
# {
#     "status": "success",
#     "message": "Connected to OpenRouter API (337 models available)",
#     "response_time": 0.16,
#     "authenticated": True,
#     "model_count": 337,
#     "api_version": "v1"
# }
```

### Lista modeli

```python
result = openrouter_service.get_available_models()
if result["status"] == "success":
    for model in result["models"]:
        print(f"{model['id']}: {model['name']}")
        print(f"  Context: {model['context_length']} tokens")
```

### Generowanie tekstu

```python
result = openrouter_service.generate_text(
    prompt="Write a haiku about AI",
    model="openai/gpt-3.5-turbo",
    max_tokens=250,
    temperature=0.7
)

if result["status"] == "success":
    print(result["generated_text"])
    print(f"Tokens used: {result['usage']['total_tokens']}")
```

### Czat (konwersacja)

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help you today?"},
    {"role": "user", "content": "What's the weather like?"}
]

result = openrouter_service.chat(
    messages=messages,
    model="openai/gpt-3.5-turbo",
    max_tokens=250,
    temperature=0.7
)

if result["status"] == "success":
    print(result["response"])
    print(f"Tokens: {result['usage']['total_tokens']}")
```

### Zaawansowane parametry

```python
result = openrouter_service.generate_text(
    prompt="Your prompt here",
    model="anthropic/claude-2",
    max_tokens=500,
    temperature=0.9,
    top_p=0.95,
    frequency_penalty=0.5,
    presence_penalty=0.5,
    stop=["\n\n", "END"]
)
```

## Dostępne modele

OpenRouter zapewnia dostęp do setek modeli, w tym:

### OpenAI
- `openai/gpt-4` - GPT-4
- `openai/gpt-3.5-turbo` - GPT-3.5 Turbo (domyślny)
- `openai/gpt-4-turbo` - GPT-4 Turbo

### Anthropic
- `anthropic/claude-2` - Claude 2
- `anthropic/claude-instant-1` - Claude Instant

### Google
- `google/palm-2-codechat-bison` - PaLM 2 Code Chat
- `google/gemini-pro` - Gemini Pro

### Meta
- `meta-llama/llama-2-70b-chat` - Llama 2 70B Chat
- `meta-llama/llama-2-13b-chat` - Llama 2 13B Chat

### Open Source
- `mistralai/mistral-7b-instruct` - Mistral 7B Instruct
- `nousresearch/nous-hermes-llama2-13b` - Nous Hermes Llama 2

**Pełna lista**: Użyj komendy `python3 rag_cli.py openrouter models`

## Obsługa błędów

Serwis automatycznie obsługuje:
- Błędy połączenia (retry z exponential backoff)
- Błędy HTTP (401, 429, 500, etc.)
- Brakująca konfiguracja
- Limity API

```python
result = openrouter_service.generate_text(prompt="Test")

if result["status"] == "error":
    print(f"Error: {result['message']}")
    print(f"Type: {result.get('error_type')}")
    if "status_code" in result:
        print(f"HTTP Status: {result['status_code']}")
```

## Ceny i limity

- Różne modele mają różne ceny
- Użyj `openrouter models` aby zobaczyć ceny
- OpenRouter pokazuje użycie w odpowiedziach:
  ```python
  result["usage"] = {
      "prompt_tokens": 10,
      "completion_tokens": 20,
      "total_tokens": 30
  }
  ```

## Testowanie

Uruchom testy:

```bash
# Wszystkie testy OpenRouter
pytest Common/tests/test_openrouter_service.py -v

# Z pokryciem kodu
pytest Common/tests/test_openrouter_service.py --cov=Common.services.openrouter_service -v
```

## Rozszerzanie

OpenRouter implementuje interfejsy z `llm_provider_base.py`. Możesz:

1. Dodać nowego providera implementując te same interfejsy
2. Użyć polimorfizmu do przełączania między providerami
3. Stworzyć strategię wyboru najlepszego providera

Przykład:

```python
from Common.services.llm_provider_base import LLMProviderBase

def generate_with_fallback(
    providers: list[LLMProviderBase],
    prompt: str
) -> Dict[str, Any]:
    """Try providers in order until one succeeds."""
    for provider in providers:
        if not provider.is_available():
            continue
        
        result = provider.generate_text(prompt=prompt)
        if result["status"] == "success":
            return result
    
    return {"status": "error", "message": "All providers failed"}
```

## Troubleshooting

### "OpenRouter API key not configured"

```bash
# Sprawdź plik .env
cat .env | grep OPENROUTER_API_KEY

# Ustaw klucz
echo "OPENROUTER_API_KEY=your_key_here" >> .env
```

### "Invalid API key"

- Sprawdź czy klucz jest poprawny na https://openrouter.ai/keys
- Upewnij się, że nie ma spacji przed/po kluczu

### "Rate limit exceeded"

- Poczekaj chwilę przed kolejnym requestem
- Rozważ upgrade planu na OpenRouter
- Użyj innego modelu (niektóre mają wyższe limity)

### Testy nie przechodzą

```bash
# Zresetuj cache pytest
rm -rf .pytest_cache __pycache__

# Sprawdź czy wszystkie zależności są zainstalowane
pip install -r requirements.txt

# Uruchom testy z verbose
pytest Common/tests/test_openrouter_service.py -vv
```

## Zasoby

- OpenRouter Docs: https://openrouter.ai/docs
- API Keys: https://openrouter.ai/keys
- Models: https://openrouter.ai/models
- Pricing: https://openrouter.ai/pricing
