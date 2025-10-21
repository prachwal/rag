# Common - Wspólna logika projektu

Ten folder zawiera wspólną logikę biznesową i infrastrukturę używaną w całym projekcie.

## Struktura folderu

- **models/** - Modele danych i klasy Pydantic
- **services/** - Servisy biznesowe i interfejsy
- **utils/** - Funkcje pomocnicze i narzędzia
- **constants/** - Stałe i konfiguracje
- **exceptions/** - Niestandardowe wyjątki aplikacji

## Zasady rozwoju

Zgodnie z wytycznymi projektu:

- **Python 3.11+** - Używaj najnowszych funkcji językowych (type hints, async/await, structural pattern matching)
- **Asynchroniczność** - Wszystkie operacje I/O powinny być asynchroniczne z użyciem asyncio
- **Type Hints** - Używaj pełnych adnotacji typów dla lepszej czytelności i narzędzi deweloperskich
- **Bezpieczeństwo** - Waliduj dane wejściowe, używaj prepared statements w SQL
- **Logowanie** - Implementuj strukturalne logowanie z poziomami (DEBUG, INFO, WARNING, ERROR)
- **Testy** - Pisz testy jednostkowe z pytest dla logiki biznesowej
- **Dokumentacja** - Używaj docstrings dla wszystkich publicznych funkcji i klas

## Użycie

### Konfiguracja aplikacji

Moduł zawiera serwis do zarządzania konfiguracją aplikacji z plików `.env`:

```python
# Podstawowe użycie
from common.services import get_settings, get_config

# Pobierz ustawienia aplikacji
settings = get_settings()
print(f"App name: {settings.app_name}")
print(f"Debug mode: {settings.debug}")
print(f"Database URL: {settings.database_url}")

# Lub użyj serwisu konfiguracyjnego
config_service = get_config()
db_url = config_service.get_database_url()
api_config = config_service.get_api_config()
is_debug = config_service.is_debug_mode()
```

### Przykład użycia w serwisie

```python
# Przykład użycia w innym module
from common.services import get_settings
from common.exceptions import BusinessLogicError
import logging

class ExampleService:
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)

    async def do_work_async(self) -> None:
        """Przykładowa metoda asynchroniczna"""
        self.logger.info("Starting work")

        try:
            # Użyj konfiguracji w logice biznesowej
            if self.settings.debug:
                self.logger.info("Running in debug mode")

            # Logika biznesowa
            result = await self._process_data()
            self.logger.info("Work completed successfully")
        except Exception as e:
            self.logger.error(f"Work failed: {e}")
            raise BusinessLogicError(f"Processing failed: {e}")

    async def _process_data(self) -> dict:
        """Prywatna metoda przetwarzająca dane"""
        # Implementacja logiki biznesowej
        return {"status": "processed"}
```

### Plik konfiguracyjny (.env)

Utwórz plik `.env` w głównym katalogu projektu:

```env
# Application Settings
APP_NAME=RAG Application
APP_VERSION=1.0.0
DEBUG=false

# Server Settings
HOST=0.0.0.0
PORT=8000

# Database Settings
DATABASE_URL=postgresql://localhost:5432/rag_db

# API Settings
API_KEY=your_api_key_here
API_TIMEOUT=30

# Logging Settings
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Security Settings (REQUIRED)
SECRET_KEY=your_secret_key_minimum_32_characters_long_key_here
```

### Testowanie

Uruchom testy za pomocą pytest:

```bash
# Zainstaluj zależności
pip install -r requirements.txt

# Uruchom wszystkie testy
pytest Common/tests/

# Uruchom testy z pokryciem kodu
pytest --cov=Common Common/tests/

# Uruchom tylko testy konfiguracji
pytest Common/tests/test_config_service.py -v
```