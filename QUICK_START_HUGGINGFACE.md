# HuggingFace Service - Quick Start

## Instalacja zależności

```bash
# Podstawowe (tylko API)
pip install requests

# Lokalne embeddings (zalecane)
pip install sentence-transformers

# Lokalna generacja tekstu (opcjonalne, wymaga GPU dla większych modeli)
pip install transformers torch
```

## Konfiguracja (.env)

```bash
# Opcjonalne - dla dostępu do HuggingFace API
HUGGINGFACE_TOKEN=hf_your_token_here

# Konfiguracja lokalnego modelu embeddings
EMBEDDING_MODEL=Voicelab/sbert-large-cased-pl

# Timeout dla API
HUGGINGFACE_API_TIMEOUT=30
```

## Przykłady użycia

### 1. Test połączenia

```bash
# Sprawdź dostępność wszystkich backendy
python3 rag_cli.py huggingface test

# JSON output
python3 rag_cli.py huggingface test --json
```

### 2. Sprawdź capabilities

```bash
python3 rag_cli.py huggingface capabilities
```

### 3. Embeddings (preferuje lokalne modele)

```bash
# Pojedynczy tekst
python3 rag_cli.py huggingface embeddings "To jest przykładowy tekst"

# Wiele tekstów
python3 rag_cli.py huggingface embeddings "Tekst 1" "Tekst 2" "Tekst 3"

# Z pliku (stdin)
echo "Tekst do embedowania" | python3 rag_cli.py huggingface embeddings --stdin

# Wymuś API zamiast lokalnego modelu
python3 rag_cli.py huggingface embeddings "Tekst" --api

# Użyj konkretnego modelu
python3 rag_cli.py huggingface embeddings "Tekst" --model "all-MiniLM-L6-v2"

# JSON output
python3 rag_cli.py huggingface embeddings "Tekst" --json
```

### 4. Generacja tekstu

```bash
# Używając API (domyślnie)
python3 rag_cli.py huggingface generate "Hello, how are you?"

# Z parametrami
python3 rag_cli.py huggingface generate "Tell me a story" \
  --model "microsoft/DialoGPT-medium" \
  --max-new-tokens 100 \
  --temperature 0.8

# Wymuś lokalny model (wymaga transformers + torch)
python3 rag_cli.py huggingface generate "Hello" --local

# JSON output
python3 rag_cli.py huggingface generate "Hello" --json
```

### 5. Lista dostępnych modeli

```bash
# Podstawowa lista
python3 rag_cli.py huggingface models

# Wyszukiwanie
python3 rag_cli.py huggingface models --search "sentiment"

# Z filtrem po zadaniu
python3 rag_cli.py huggingface models --filter "text-generation" --limit 20

# JSON output
python3 rag_cli.py huggingface models --search "gpt" --json
```

## Użycie w kodzie Python

```python
from Common.services.huggingface_service import huggingface_service

# Test connection
result = huggingface_service.test_connection()
print(result)

# Embeddings (lokalnie)
texts = ["Tekst 1", "Tekst 2"]
result = huggingface_service.get_local_embeddings(texts)
embeddings = result["embeddings"]

# Embeddings (przez API)
result = huggingface_service.get_embeddings(texts, use_local=False)

# Generacja tekstu (API)
result = huggingface_service.generate_text(
    prompt="Hello, world!",
    model="microsoft/DialoGPT-medium",
    max_new_tokens=50
)
print(result["generated_text"])

# Generacja tekstu (lokalnie - wymaga dużo RAM/GPU)
result = huggingface_service.generate_text(
    prompt="Hello!",
    model="gpt2",
    use_local=True
)

# Lista modeli
result = huggingface_service.list_models(search="bert", limit=10)
models = result["models"]

# Sprawdź capabilities
caps = huggingface_service.get_capabilities()
print(f"Local embeddings: {caps['local_embeddings']}")
print(f"API access: {caps['api_access']}")
```

## Strategia fallback

Service automatycznie wybiera najlepszy backend:

### Embeddings
1. **Próba**: Lokalny model (sentence-transformers)
2. **Fallback**: HuggingFace API (jeśli token dostępny)
3. **Error**: Clear message z instrukcjami instalacji

### Text Generation
1. **use_local=True**: Tylko lokalny model
2. **use_local=False** (default): Tylko API
3. **Brak token + use_local=False**: Próba lokalnego modelu → Error jeśli nie zainstalowany

## Tips & Tricks

### Oszczędzanie pamięci RAM

```bash
# Używaj mniejszych modeli dla embeddings
export EMBEDDING_MODEL="all-MiniLM-L6-v2"  # ~80MB

# Zamiast
# export EMBEDDING_MODEL="all-mpnet-base-v2"  # ~420MB
```

### Pierwsze uruchomienie

Pierwsze wywołanie pobierze model z HuggingFace Hub (może trwać 30-120s):
```bash
# Model zostanie zapisany w cache (~/.cache/huggingface/)
python3 rag_cli.py huggingface embeddings "test"
```

### Praca offline

```bash
# Gdy models są już w cache, możesz pracować offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 rag_cli.py huggingface embeddings "tekst"
```

### Batch processing w kodzie

```python
# Dla dużych ilości tekstów (>100)
texts = [...1000 tekstów...]

# Przetwarzaj w batchach
batch_size = 32
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    result = huggingface_service.get_local_embeddings(batch)
    # Przetwórz wyniki...
```

## Troubleshooting

### "sentence-transformers not installed"
```bash
pip install sentence-transformers
```

### "transformers not installed"
```bash
pip install transformers torch
```

### "CUDA out of memory" podczas generacji
```python
# Użyj mniejszego modelu lub CPU
result = huggingface_service.generate_text(
    prompt="Hello",
    model="distilgpt2",  # Mniejszy model
    use_local=True
)
```

### Model pobiera się za długo
```bash
# Sprawdź czy model już istnieje
ls ~/.cache/huggingface/hub/

# Lub użyj API zamiast lokalnego
python3 rag_cli.py huggingface embeddings "text" --api
```

## Więcej informacji

- Pełna dokumentacja: `ANALYSIS_HUGGINGFACE_REDESIGN.md`
- Testy: `Common/tests/test_huggingface_service.py`
- Kod serwisu: `Common/services/huggingface_service.py`
- Komendy CLI: `rag_cli/commands/huggingface.py`
