# PDF Parser для русскоязычных документов

Точный парсер PDF-документов на русском языке с высокой точностью извлечения текста, таблиц и печатей.

![Версия](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)
![Лицензия](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Содержание

- [Возможности](#-возможности)
- [Требования](#-требования)
- [Установка](#-установка)
- [Использование](#️-использование)
  - [Командная строка](#командная-строка)
  - [Python API](#python-api)
- [Конфигурация](#️-конфигурация)
- [Примеры использования](#-примеры-использования)
- [Структура проекта](#-структура-проекта)
- [Устранение неполадок](#-устранение-неполадок)
- [Тестирование](#-тестирование)
- [Вклад в проект](#-вклад-в-проект)
- [Лицензия](#-лицензия)

## 🚀 Возможности

- **Высокая точность распознавания текста** для русскоязычных документов
- **Корректное извлечение и распознавание таблиц**, включая многостраничные таблицы
- **Обнаружение и распознавание печатей** в сканированных документах
- **Предварительная обработка изображений** для улучшения качества распознавания
- **Параллельная обработка** для повышения производительности
- **Пакетная обработка** для экономии памяти при работе с большими документами
- **Поддержка различных форматов вывода**: JSON, текст, Markdown
- **Гибкая настройка параметров распознавания**

## 📋 Требования

### Системные требования

- Python 3.8 или выше
- 4 ГБ ОЗУ (рекомендуется 8+ ГБ для больших документов)
- Tesseract OCR 5.x с поддержкой русского языка
- Unpaper (опционально, для улучшения качества распознавания сканов)

### Зависимости Python

- dedoc>=1.0.0
- pdf2image>=1.16.0
- pdfplumber>=0.7.0
- opencv-python>=4.5.0
- numpy>=1.20.0
- scikit-image>=0.18.0 (опционально, для улучшенной бинаризации)
- img2pdf>=0.4.0
- PyPDF2>=3.0.0

## 💻 Установка

### Установка системных зависимостей

#### Ubuntu/Debian

```
# Установка Tesseract OCR с поддержкой русского языка
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-rus

# Установка Unpaper (опционально)
sudo apt-get install unpaper

# Установка других зависимостей
sudo apt-get install build-essential libpoppler-cpp-dev pkg-config python3-dev
```

#### CentOS/RHEL

```
# Установка Tesseract OCR
sudo yum install tesseract tesseract-langpack-rus

# Установка Unpaper (опционально)
sudo yum install unpaper

# Установка других зависимостей
sudo yum install gcc gcc-c++ python3-devel poppler-cpp-devel pkgconfig
```

#### Windows

1. Скачайте и установите [Tesseract для Windows](https://github.com/UB-Mannheim/tesseract/wiki)
2. При установке обязательно выберите русский язык
3. Добавьте путь к Tesseract в переменную среды PATH
4. Опционально: установите [Unpaper для Windows](https://github.com/unpaper/unpaper/releases)

#### macOS

```
# С использованием Homebrew
brew install tesseract
brew install tesseract-lang  # Устанавливает дополнительные языки, включая русский
brew install unpaper  # Опционально
```

### Установка пакета

#### Из PyPI (рекомендуется)

```
pip install pdfparser
```

#### Из исходников

```
# Клонирование репозитория
git clone https://github.com/1kuzz/pdfparser.git
cd pdfparser

# Создание виртуального окружения (рекомендуется)
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate

# Установка зависимостей
pip install -r requirements.txt

# Установка пакета в режиме разработки
pip install -e .
```

## 🖥️ Использование

### Командная строка

Базовое использование:

```
pdfparser input.pdf -o output.json -f json
```

Расширенное использование:

```
pdfparser input.pdf -o output.json -f json -d 400 --document-type tz --detect-stamps --process-stamps
```

#### Основные параметры CLI

```
usage: pdfparser [-h] [-o OUTPUT] [-f {json,text,markdown}] [-d DPI]
                 [--document-type {tz,other,law,article}] [--no-preprocess]
                 [--detect-stamps] [--process-stamps] [--use-unpaper]
                 [--binarize-method {otsu,sauvola}] [--temp-dir TEMP_DIR]
                 [--no-clean-temp] [--no-parallel] [--batch-size BATCH_SIZE]
                 [--config CONFIG] [--debug]
                 input_file

Парсер PDF документов с высокой точностью для русского языка

позиционные аргументы:
  input_file            Путь к входному PDF файлу

опциональные аргументы:
  -h, --help            показать справку и выйти
  -o OUTPUT, --output OUTPUT
                        Путь для сохранения результата (по умолчанию: input_file_result.{format})
  -f {json,text,markdown}, --format {json,text,markdown}
                        Формат вывода (по умолчанию: json)
  -d DPI, --dpi DPI     DPI для конвертации PDF в изображения (по умолчанию: 300)
  --document-type {tz,other,law,article}
                        Тип документа для dedoc (по умолчанию: tz)
  --no-preprocess       Отключить предобработку изображений
  --detect-stamps       Включить обнаружение печатей
  --process-stamps      Обрабатывать области печатей отдельно
  --use-unpaper         Использовать unpaper для очистки сканов
  --binarize-method {otsu,sauvola}
                        Метод бинаризации (по умолчанию: sauvola)
  --temp-dir TEMP_DIR   Директория для временных файлов
  --no-clean-temp       Не удалять временные файлы после обработки
  --no-parallel         Отключить параллельную обработку
  --batch-size BATCH_SIZE
                        Размер пакета при обработке больших документов (по умолчанию: 10)
  --config CONFIG       Путь к файлу конфигурации JSON
  --debug               Включить отладочный режим
```

### Python API

```
from pdfparser import PDFParser

# Создаем парсер с настройками
parser = PDFParser({
    "dpi": 400,                         # DPI для конвертации PDF в изображения
    "document_type": "tz",              # Тип документа (tz, law, article, other)
    "preprocess_images": True,          # Предобработка изображений
    "detect_stamps": True,              # Обнаружение печатей
    "process_stamps_separately": True,  # Обработка областей печатей отдельно
    "binarize_method": "sauvola",       # Метод бинаризации (sauvola или otsu)
    "output_format": "json",            # Формат вывода (json, text, markdown)
    "use_parallel": True,               # Параллельная обработка
    "max_workers": 4,                   # Количество рабочих процессов
    "batch_size": 10,                   # Размер пакета страниц
})

# Обрабатываем PDF
result = parser.process("document.pdf")

# Работа с результатом
print(f"Извлечено {len(result['text'])} символов текста")
print(f"Найдено {len(result['tables'])} таблиц")

# Если были обнаружены печати
if "stamp_texts" in result:
    print(f"Найдено {len(result['stamp_texts'])} текстов печатей")
    for i, stamp_text in enumerate(result['stamp_texts'], 1):
        print(f"Печать {i}: {stamp_text[:50]}...")
```

## ⚙️ Конфигурация

### Файл конфигурации

Вы можете создать файл конфигурации в формате JSON для сохранения настроек:

```
{
    "dpi": 400,
    "document_type": "tz",
    "preprocess_images": true,
    "detect_stamps": true,
    "process_stamps_separately": true,
    "use_unpaper": false,
    "binarize_method": "sauvola",
    "window_size": 25,
    "use_parallel": true,
    "max_workers": 4,
    "batch_size": 10,
    "clean_temp": true,
    "stamp_detection_params": {
        "dp": 1.2,
        "minDist": 100,
        "param1": 50,
        "param2": 30,
        "minRadius": 30,
        "maxRadius": 150
    },
    "quality_weights": {
        "text_quality": 0.7,
        "table_count": 0.3
    }
}
```

Затем используйте файл конфигурации:

```
pdfparser input.pdf -o output.json --config config.json
```

## 📝 Примеры использования

### Обработка технического задания

```
pdfparser tz.pdf -o tz_parsed.json --document-type tz -d 400 --detect-stamps
```

### Обработка юридического документа

```
pdfparser contract.pdf -o contract_parsed.md -f markdown --document-type law -d 400
```

### Обработка сканированного документа с печатями

```
pdfparser scan.pdf -o scan_parsed.json --use-unpaper --detect-stamps --process-stamps
```

### Пакетная обработка большого документа

```
pdfparser big_document.pdf -o big_document_parsed.json --batch-size 5 --max-workers 8
```

### Программное использование для извлечения только таблиц

```python
from pdfparser import PDFParser

parser = PDFParser({
    "need_pdf_table_analysis": True,
    "output_format": "json"
})

result = parser.process("document_with_tables.pdf")

# Извлечение таблиц
tables = result.get("tables", [])

# Преобразование таблиц в Markdown
for i, table in enumerate(tables, 1):
    print(f"Таблица {i} (страница {table.page_number}):")
    print(table.to_markdown())
    print("\n")
```

## 📂 Структура проекта

```
pdfparser/
├── __init__.py
├── parser.py         # Основной класс PDFParser
├── models.py         # Класс Table и другие модели данных
├── utils.py          # Вспомогательные функции
├── image_processing.py # Обработка изображений
├── cli.py            # Интерфейс командной строки
├── config.py         # Конфигурация по умолчанию
├── tests/
│   ├── __init__.py
│   ├── test_image_processing.py
│   ├── test_table_processing.py
│   └── test_parser.py
├── requirements.txt
├── setup.py
└── README.md
```

## 🔧 Устранение неполадок

### Проблемы распознавания русского текста

1. Убедитесь, что Tesseract установлен с поддержкой русского языка:
   ```
   tesseract --list-langs
   ```
   В выводе должен быть 'rus'.

2. Проверьте версию Tesseract:
   ```
   tesseract --version
   ```
   Рекомендуется версия 5.0 или выше.

3. Попробуйте увеличить DPI и использовать предобработку изображений:
   ```
   pdfparser document.pdf -o output.json -d 600 --binarize-method sauvola
   ```

### Проблемы с памятью при обработке больших документов

1. Уменьшите размер пакета:
   ```
   pdfparser big_document.pdf -o output.json --batch-size 3
   ```

2. Отключите параллельную обработку:
   ```
   pdfparser big_document.pdf -o output.json --no-parallel
   ```

3. Укажите временную директорию с достаточным свободным местом:
   ```
   pdfparser big_document.pdf -o output.json --temp-dir /path/with/free/space
   ```

### Проблемы с таблицами

1. Убедитесь, что флаг анализа таблиц включен:
   ```
   pdfparser document.pdf -o output.json --config.need_pdf_table_analysis true
   ```

2. Увеличьте DPI до 400-600 для лучшего распознавания таблиц:
   ```
   pdfparser document.pdf -o output.json -d 500
   ```

### Включение режима отладки

Для получения подробной информации о процессе обработки:

```
pdfparser document.pdf -o output.json --debug
```

## 🧪 Тестирование

### Запуск тестов

```
# Установка зависимостей для тестирования
pip install pytest pytest-cov

# Запуск всех тестов
pytest

# Запуск тестов с отчетом о покрытии
pytest --cov=pdfparser
```

### Тестирование компонентов

```
# Тестирование модуля обработки изображений
pytest tests/test_image_processing.py

# Тестирование модуля обработки таблиц
pytest tests/test_table_processing.py

# Тестирование основного парсера
pytest tests/test_parser.py
```

## 🤝 Вклад в проект

Мы приветствуем вклад в развитие проекта! Если вы хотите помочь, пожалуйста:

1. Сделайте форк репозитория
2. Создайте ветку для вашей функции (`git checkout -b feature/amazing-feature`)
3. Зафиксируйте ваши изменения (`git commit -m 'Add some amazing feature'`)
4. Отправьте ветку (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

### Рекомендации по разработке

1. Следуйте стилю кодирования PEP 8
2. Добавляйте документацию к новым функциям
3. Пишите тесты для новой функциональности
4. Обновляйте README.md при необходимости

## 📄 Лицензия

Этот проект распространяется под лицензией MIT. См. файл `LICENSE` для получения дополнительной информации.

## 📞 Контакты

- **Разработчик**: Кузнецов Илья
- **GitHub**: [1kuzz](https://github.com/1kuzz/pdfparser)
- **Telegram**: [https://t.me/IlyaKuzz](https://t.me/IlyaKuzz)
- **Сообщить о проблеме**: [https://github.com/1kuzz/pdfparser/issues](https://github.com/1kuzz/pdfparser/issues)

---

## 💡 Примечания по использованию для различных типов документов

### Технические задания (tz)

Оптимальные настройки:
```
pdfparser document.pdf -o output.json --document-type tz --detect-stamps
```

### Юридические документы (law)

Оптимальные настройки:
```
pdfparser document.pdf -o output.json --document-type law --binarize-method sauvola
```

### Научные статьи (article)

Оптимальные настройки:
```
pdfparser document.pdf -o output.json --document-type article --dpi 400
```

### Сканированные документы

Оптимальные настройки:
```
pdfparser document.pdf -o output.json --use-unpaper --dpi 600 --detect-stamps --process-stamps
```

---

⭐ Если вам нравится проект, не забудьте поставить звезду на GitHub!