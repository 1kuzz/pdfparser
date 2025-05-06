"""
Конфигурация по умолчанию для парсера PDF
"""

DEFAULT_CONFIG = {
    # Параметры DPI для конвертации PDF в изображения
    "dpi": 300,
    
    # Настройки dedoc
    "document_type": "tz",  # тип документа - техническое задание
    "need_pdf_table_analysis": True,  # анализ таблиц
    "language": "rus+eng",  # язык OCR - русский + английский
    
    # Настройки предобработки изображений
    "preprocess_images": True,  # включить предобработку
    "deskew_enabled": True,  # выравнивание наклона
    "denoise_enabled": True,  # удаление шума
    "binarize_enabled": True,  # бинаризация
    "binarize_method": "sauvola",  # метод бинаризации: sauvola или otsu
    "window_size": 25,  # размер окна для метода Саувола
    
    # Использование unpaper для очистки сканов
    "use_unpaper": False,  # отключено по умолчанию, требует установки unpaper
    
    # Форматы вывода
    "output_format": "json",  # варианты: json, text, markdown
    
    # Путь для временных файлов
    "temp_dir": None,  # если None, будет использован tempfile.TemporaryDirectory()
    "clean_temp": True,  # удалять временные файлы после обработки
    
    # Параметры для распознавания печатей
    "detect_stamps": False,  # поиск печатей
    "process_stamps_separately": False,  # обрабатывать области печатей отдельно
    "stamp_detection_params": {  # параметры для HoughCircles
        "dp": 1.2,
        "minDist": 100,
        "param1": 50,
        "param2": 30,
        "minRadius": 30,
        "maxRadius": 150
    },
    
    # Параллельная обработка
    "use_parallel": True,  # использовать параллельную обработку
    "max_workers": 4,  # максимальное число параллельных процессов
    
    # Пакетная обработка для экономии памяти
    "batch_size": 10,  # число страниц в пакете при обработке больших документов
    
    # Веса для оценки качества
    "quality_weights": {
        "text_quality": 0.7,  # вес качества текста
        "table_count": 0.3,   # вес количества таблиц
    }
}