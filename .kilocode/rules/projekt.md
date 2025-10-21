# RAG Application Project

## Opis projektu

Aplikacja RAG (Retrieval-Augmented Generation) to kompleksowe rozwiązanie do zarządzania konfiguracją i integracji z zewnętrznymi API, w szczególności YouTube Data API v3. Projekt został zbudowany z myślą o modułowej architekturze, łatwości testowania i profesjonalnym podejściu do zarządzania konfiguracją.

## Architektura

### Struktura katalogów
```
rag-application/
├── Common/                          # Wspólne moduły
│   ├── services/                    # Usługi aplikacji
│   │   ├── config_service.py        # Centralna konfiguracja
│   │   └── youtube_service.py       # Integracja z YouTube API
│   └── tests/                       # Testy jednostkowe
│       ├── test_config_service.py   # Testy konfiguracji
│       └── test_youtube_service.py  # Testy YouTube API
├── rag_cli.py                       # Interfejs wiersza poleceń
├── .env                             # Konfiguracja główna
├── .env.youtube                     # Konfiguracja YouTube API
├── pyproject.toml                   # Konfiguracja projektu Python
├── requirements.txt                 # Zależności
├── pytest.ini                       # Konfiguracja testów
└── .kilocode/rules/                 # Zasady projektu
    └── projekt.md                   # Ten plik
```

### Główne komponenty

#### 1. ConfigService (`Common/services/config_service.py`)
- **Singleton pattern** zapewnia pojedynczą instancję konfiguracji
- **Pydantic Settings** do walidacji i typowania konfiguracji
- **Automatyczne generowanie SECRET_KEY** gdy nie jest podany
- **Wsparcie dla .env plików** i zmiennych środowiskowych
- **Lazy initialization** dla optymalizacji wydajności

#### 2. YouTubeAPIService (`Common/services/youtube_service.py`)
- **Kompletna integracja z YouTube Data API v3**
- **Wyszukiwanie filmów** z pełną kontrolą parametrów
- **Pobieranie szczegółów filmów** (statystyki, tagi, czas trwania)
- **Informacje o kanałach** (subskrybenci, liczba filmów)
- **Wyszukiwanie najnowszych filmów** z filtrem czasowym
- **Robust error handling** (quota exceeded, network failures)
- **Retry strategy** z wykładniczym backoff

#### 3. CLI Interface (`rag_cli.py`)
- **Click-based framework** dla profesjonalnego CLI
- **Polecenia:**
  - `rag help` - pomoc i informacje o dostępnych komendach
  - `rag config` - wyświetlanie bieżącej konfiguracji
  - `rag youtube test` - testowanie łączności z YouTube API
- **Error handling** z przyjaznymi komunikatami

## Technologie

- **Python 3.10+** - język programowania
- **Pydantic v2** - walidacja danych i ustawień
- **Requests** - HTTP klient dla API
- **Click** - framework CLI
- **Pytest** - testowanie jednostkowe
- **Dotenv** - zarządzanie zmiennymi środowiskowymi

## Konfiguracja

### Wymagane zmienne środowiskowe
- `SECRET_KEY` - klucz bezpieczeństwa (auto-generowany jeśli nie podany)
- `YOUTUBE_API_KEY` - klucz API YouTube Data v3

### Pliki konfiguracyjne
- `.env` - główna konfiguracja aplikacji
- `.env.youtube` - dedykowana konfiguracja YouTube API

## Zasady rozwoju

### Guidelines
- **używaj python3** - projekt wymaga Python 3.10 lub nowszego
- **piszemy pytest** - wszystkie testy używają frameworka pytest
- **modular architecture** - kod podzielony na niezależne moduły
- **comprehensive testing** - pełne pokrycie testami jednostkowymi
- **type hints** - używanie type hints dla lepszej jakości kodu
- **error handling** - robust obsługa błędów i wyjątków
- **documentation** - kod udokumentowany zgodnie z docstring konwencjami

### Standardy kodowania
- **PEP 8** - styl kodowania Python
- **Google docstrings** - format dokumentacji
- **Black** - formatowanie kodu (zalecane)
- **MyPy** - sprawdzanie typów (zalecane)

### Testowanie
- **100% pokrycie** kluczowych funkcji
- **Mocking** dla zewnętrznych zależności
- **Integration tests** dla kompleksowych workflow
- **Error scenarios** testing

## Uruchamianie

### Wymagania wstępne
```bash
pip install -r requirements.txt
```

### Konfiguracja
```bash
# Skopiuj przykładowe pliki konfiguracyjne
cp .env.example .env
cp .env.youtube.example .env.youtube

# Edytuj klucze API
nano .env.youtube
```

### Uruchamianie CLI
```bash
# Wyświetl pomoc
python3 rag_cli.py help

# Pokaż konfigurację
python3 rag_cli.py config

# Testuj YouTube API
python3 rag_cli.py youtube test --query "python tutorial"
```

### Testowanie
```bash
# Uruchom wszystkie testy
python3 -m pytest

# Uruchom testy z pokryciem
python3 -m pytest --cov=Common
```

## API Reference

### ConfigService
```python
from Common.services.config_service import config_service

# Pobierz ustawienia
settings = config_service.settings

# Pobierz konkretne ustawienie
api_key = config_service.get_setting('api_key')

# Pobierz konfigurację YouTube
youtube_config = config_service.get_youtube_config()
```

### YouTubeAPIService
```python
from Common.services.youtube_service import search_youtube_videos

# Wyszukaj filmy
results = search_youtube_videos("python tutorial", max_results=10)

# Pobierz szczegóły filmów
details = get_youtube_video_details(['video_id_1', 'video_id_2'])
```

## Bezpieczeństwo

- **SECRET_KEY** automatycznie generowany jeśli nie podany
- **API keys** walidowane przed użyciem
- **Input validation** dla wszystkich parametrów
- **Error handling** bez ujawniania wrażliwych danych

## Rozwój

Projekt jest otwarty na rozszerzenia i nowe funkcjonalności. Przestrzeganie zasad opisanych w tym dokumencie zapewnia spójność i jakość kodu.
