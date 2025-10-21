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
├── rag_cli/                         # Interfejs wiersza poleceń
│   ├── cli.py                       # Główny moduł CLI
│   └── commands/                    # Polecenia CLI
│       ├── youtube.py               # Polecenia YouTube
│       ├── config.py                # Polecenia konfiguracji
│       └── help.py                  # Polecenia pomocy
├── rag_cli.py                       # Punkt wejścia aplikacji
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
#### 3. TranscriptionService (`Common/services/transcription_service.py`)
- **Inteligentna transkrypcja filmów YouTube** z strategią priorytetową:
  - **YouTube Transcript API** - pierwsza kolejność (szybki, darmowy, dokładny)
  - **OpenAI Whisper** - fallback dla filmów bez napisów
- **Automatyczne pobieranie audio** z yt-dlp (tylko gdy potrzebne)
- **Wsparcie dla wielu języków** z auto-detecją
- **Segmentacja transkrypcji** z timestampami
- **Lazy loading** modeli dla optymalizacji pamięci

#### 4. CLI Interface (`rag_cli/`)
- **Singleton pattern** zapewnia pojedynczą instancję konfiguracji
- **Pydantic Settings** do walidacji i typowania konfiguracji
- **Automatyczne generowanie SECRET_KEY** gdy nie jest podany
- **Wsparcie dla .env plików** i zmiennych środowiskowych
  - `rag youtube info <video>` - szczegółowe informacje o filmie
  - `rag youtube channel <channel>` - informacje o kanale
  - `rag youtube playlists <channel>` - lista playlist kanału
  - `rag youtube playlist <playlist>` - wszystkie filmy z playlisty
  - `rag youtube videos <channel>` - wszystkie filmy kanału
  - `rag youtube transcribe <video>` - transkrypcja filmu na tekst (YouTube API + Whisper fallback)
- **Lazy initialization** dla optymalizacji wydajności

#### 2. YouTubeAPIService (`Common/services/youtube_service.py`)
- **Kompletna integracja z YouTube Data API v3**
- **Wyszukiwanie filmów** z pełną kontrolą parametrów
- **Pobieranie szczegółów filmów** (statystyki, tagi, czas trwania)
- **Informacje o kanałach** (subskrybenci, liczba filmów)
- **Wyszukiwanie najnowszych filmów** z filtrem czasowym
- **Playlist management** - pobieranie playlist kanału i ich zawartości z paginacją
- **Channel videos** - pobieranie wszystkich filmów kanału z automatyczną paginacją
- **Video/Channel ID extraction** - ekstrakcja ID z różnych formatów URL YouTube
- **Robust error handling** (quota exceeded, network failures)
- **Retry strategy** z wykładniczym backoff

#### 3. CLI Interface (`rag_cli/`)
- **Click-based framework** dla profesjonalnego CLI
- **Polecenia:**
  - `rag help` - pomoc i informacje o dostępnych komendach
  - `rag config` - wyświetlanie bieżącej konfiguracji
  - `rag youtube test` - testowanie łączności z YouTube API
  - `rag youtube info <video>` - szczegółowe informacje o filmie
  - `rag youtube channel <channel>` - informacje o kanale
  - `rag youtube playlists <channel>` - lista playlist kanału
  - `rag youtube playlist <playlist>` - wszystkie filmy z playlisty
  - `rag youtube videos <channel>` - wszystkie filmy kanału
- **JSON output support** dla wszystkich poleceń YouTube
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
# Pobierz informacje o filmie
python3 rag_cli.py youtube info "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Pobierz informacje o kanale
python3 rag_cli.py youtube channel "UC1234567890abcdef"

# Lista playlist kanału
python3 rag_cli.py youtube playlists "UC1234567890abcdef"

# Wszystkie filmy z playlisty
python3 rag_cli.py youtube playlist "PL1234567890abcdef"

# Wszystkie filmy kanału
python3 rag_cli.py youtube videos "UC1234567890abcdef"

# Transkrypcja filmu (YouTube API lub Whisper)
python3 rag_cli.py youtube transcribe "VIDEO_ID" --language en

# Czysty tekst (jedna linia na segment)
python3 rag_cli.py youtube transcribe "VIDEO_ID" --text

# JSON output
python3 rag_cli.py youtube transcribe "VIDEO_ID" --json

# Wszystkie polecenia wspierają format JSON z opcją --json
python3 rag_cli.py youtube info "VIDEO_ID" --json

## Uruchamianie

### Wymagania wstępne
```bash
pip install -r requirements.txt
```

### Konfiguracja
```bash
### TranscriptionService
```python
from Common.services.transcription_service import transcribe_youtube_video

# Transkrybuj film YouTube (automatycznie wybiera najlepszą metodę)
result = transcribe_youtube_video("VIDEO_ID_OR_URL", language="en")

print(result['text'])  # Pełny tekst transkrypcji
print(result['language'])  # Wykryty język
print(result['source'])  # Źródło: 'youtube_transcript_api' lub 'whisper'
print(result['segments'])  # Segmenty z timestampami
```
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

# Pobierz informacje o filmie
python3 rag_cli.py youtube info "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Pobierz informacje o kanale
python3 rag_cli.py youtube channel "UC1234567890abcdef"

# Lista playlist kanału
python3 rag_cli.py youtube playlists "UC1234567890abcdef"

# Wszystkie filmy z playlisty
python3 rag_cli.py youtube playlist "PL1234567890abcdef"

# Wszystkie filmy kanału
python3 rag_cli.py youtube videos "UC1234567890abcdef"

# Wszystkie polecenia wspierają format JSON z opcją --json
python3 rag_cli.py youtube info "VIDEO_ID" --json
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
from Common.services.youtube_service import (
    search_youtube_videos,
    get_youtube_video_info,
    get_youtube_channel_info,
    get_youtube_channel_playlists,
    get_youtube_playlist_videos_full,
    get_youtube_channel_videos_full
)

# Wyszukaj filmy
results = search_youtube_videos("python tutorial", max_results=10)

# Pobierz informacje o pojedynczym filmie
video_info = get_youtube_video_info("VIDEO_ID_OR_URL")

# Pobierz informacje o kanale
channel_info = get_youtube_channel_info("CHANNEL_ID_OR_URL")

# Lista playlist kanału
playlists = get_youtube_channel_playlists("CHANNEL_ID", max_results=20)

# Wszystkie filmy z playlisty (z paginacją)
playlist_videos = get_youtube_playlist_videos_full("PLAYLIST_ID")

# Wszystkie filmy kanału (z paginacją)
channel_videos = get_youtube_channel_videos_full("CHANNEL_ID")
```

## Bezpieczeństwo

- **SECRET_KEY** automatycznie generowany jeśli nie podany
- **API keys** walidowane przed użyciem
- **Input validation** dla wszystkich parametrów
- **Error handling** bez ujawniania wrażliwych danych

## Rozwój

Projekt jest otwarty na rozszerzenia i nowe funkcjonalności. Przestrzeganie zasad opisanych w tym dokumencie zapewnia spójność i jakość kodu.