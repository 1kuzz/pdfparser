"""
Вспомогательные функции для парсинга PDF
"""

import os
import logging
import tempfile
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from functools import partial

import cv2
import numpy as np
from pdf2image import convert_from_path
from PyPDF2 import PdfReader, PdfWriter

from .config import DEFAULT_CONFIG


def get_logger(name: str) -> logging.Logger:
    """
    Создает и возвращает логгер с заданным именем.
    
    Args:
        name: Имя логгера
    
    Returns:
        logging.Logger: Настроенный логгер
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Устанавливаем уровень логирования по умолчанию
        logger.setLevel(logging.INFO)
    
    return logger


logger = get_logger(__name__)


def process_batch(pdf_path: str, page_range: Tuple[int, int], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Обрабатывает пакет страниц PDF документа.
    
    Args:
        pdf_path: Путь к PDF файлу
        page_range: Диапазон страниц (начало, конец)
        config: Конфигурация обработки
        
    Returns:
        Dict[str, Any]: Результат обработки пакета
    """
    start_page, end_page = page_range
    logger.info(f"Обработка пакета страниц {start_page}-{end_page} из {pdf_path}")
    
    # Создаем временный PDF с указанными страницами
    try:
        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        
        for i in range(start_page - 1, min(end_page, len(reader.pages))):
            writer.add_page(reader.pages[i])
        
        # Создаем временный файл
        temp_dir = config.get("temp_dir", tempfile.gettempdir())
        batch_pdf_path = os.path.join(temp_dir, f"batch_{start_page}_{end_page}_{os.path.basename(pdf_path)}")
        
        with open(batch_pdf_path, "wb") as out_file:
            writer.write(out_file)
        
        # Импортируем PDFParser здесь, чтобы избежать циклических импортов
        from .parser import PDFParser
        
        # Создаем экземпляр PDFParser
        parser = PDFParser(config)
        
        # Обрабатываем временный PDF
        result = parser.process_pdf(batch_pdf_path)
        
        # Корректируем номера страниц в таблицах
        if "tables" in result:
            for table in result["tables"]:
                if hasattr(table, "page_number"):
                    table.page_number += (start_page - 1)
        
        # Очищаем временный файл
        if config.get("clean_temp", True):
            Path(batch_pdf_path).unlink(missing_ok=True)
        
        return result
    except Exception as e:
        logger.error(f"Ошибка при создании временного PDF для пакета {page_range}: {e}")
        return {}


def merge_tables(tables: List['Table']) -> List['Table']:
    """
    Объединяет таблицы, которые могут быть продолжением друг друга.
    
    Args:
        tables: Список таблиц
        
    Returns:
        List[Table]: Объединенные таблицы
    """
    if not tables or len(tables) <= 1:
        return tables
    
    # Импортируем Table здесь, чтобы избежать циклических импортов
    from .models import Table
    
    # Сортируем таблицы по номеру страницы и позиции
    sorted_tables = sorted(tables, key=lambda t: (t.page_number, t.bbox[1] if t.bbox else 0))
    
    # Итеративное слияние таблиц
    merged = True
    while merged:
        merged = False
        merged_tables = []
        i = 0
        
        while i < len(sorted_tables):
            current_table = sorted_tables[i]
            
            # Проверяем, можно ли объединить текущую таблицу со следующей
            if i + 1 < len(sorted_tables) and current_table.can_merge_with(sorted_tables[i + 1]):
                # Объединяем таблицы
                merged_table = current_table.merge_with(sorted_tables[i + 1])
                merged_tables.append(merged_table)
                i += 2  # Пропускаем обе объединенные таблицы
                merged = True  # Отмечаем, что было объединение
                logger.debug(f"Объединены таблицы со страниц {current_table.page_number} и {sorted_tables[i-1].page_number}")
            else:
                # Добавляем текущую таблицу без изменений
                merged_tables.append(current_table)
                i += 1
        
        # Обновляем список таблиц для следующей итерации
        sorted_tables = merged_tables
    
    return sorted_tables


def evaluate_text_quality(text: str) -> float:
    """
    Оценивает качество распознанного текста по различным метрикам.
    
    Args:
        text: Текст для оценки
        
    Returns:
        float: Оценка качества от 0 до 1
    """
    import string
    
    if not text:
        return 0.0
    
    # Количество строк
    lines = text.split('\n')
    if not lines:
        return 0.0
    
    # Метрики качества
    metrics = {}
    
    # Средняя длина строки (слишком короткие строки могут быть мусором)
    line_lengths = [len(line.strip()) for line in lines]
    avg_line_length = sum(line_lengths) / len(lines) if line_lengths else 0
    metrics["avg_line_length"] = min(1.0, avg_line_length / 30.0)  # Нормализуем
    
    # Доля "хороших" символов (буквы, цифры, знаки пунктуации)
    good_chars = sum(1 for c in text if c.isalnum() or c in ".,;:!?()-—«»")
    metrics["good_chars_ratio"] = good_chars / len(text) if text else 0
    
    # Доля печатаемых символов
    printable_chars = sum(1 for c in text if c in string.printable)
    metrics["printable_ratio"] = printable_chars / len(text) if text else 0
    
    # Доля кириллических символов (для русского текста)
    cyrillic_chars = sum(1 for c in text if 'а' <= c.lower() <= 'я' or c == 'ё')
    metrics["cyrillic_ratio"] = cyrillic_chars / good_chars if good_chars else 0
    
    # Считаем отсутствие мусорных последовательностей
    garbage_patterns = ['~~~', '###', '***', '...', '___', '   ', '\\\\\\', '|||', '+++', '<<<', '>>>']
    garbage_count = sum(text.count(pattern) for pattern in garbage_patterns)
    metrics["no_garbage"] = 1.0 - min(1.0, garbage_count / len(lines))
    
    # Считаем наличие типичных для текста слов
    common_words = ['и', 'в', 'на', 'с', 'по', 'для', 'от', 'к', 'за', 'из', 'о', 'при']
    common_words_count = sum(text.count(' ' + word + ' ') for word in common_words)
    metrics["common_words"] = min(1.0, common_words_count / 10.0)  # Нормализуем
    
    # Вычисляем общую оценку как взвешенное среднее
    weights = {
        "avg_line_length": 0.15,
        "good_chars_ratio": 0.2,
        "printable_ratio": 0.2,
        "cyrillic_ratio": 0.25,
        "no_garbage": 0.1,
        "common_words": 0.1
    }
    
    overall_score = sum(metrics[key] * weights[key] for key in metrics)
    logger.debug(f"Оценка качества текста: {overall_score:.2f}, метрики: {metrics}")
    
    return overall_score


def evaluate_result_quality(result: Dict[str, Any], max_tables: int = 1) -> float:
    """
    Вычисляет общую оценку качества результата на основе текста и таблиц.
    
    Args:
        result: Результат парсинга
        max_tables: Максимальное известное количество таблиц (для нормализации)
        
    Returns:
        float: Общая оценка качества от 0 до 1
    """
    # Оценка качества текста
    text = result.get("text", "")
    text_quality = evaluate_text_quality(text)
    
    # Оценка количества таблиц
    tables = result.get("tables", [])
    table_count = len(tables)
    table_score = table_count / max(max_tables, 1)  # Нормализуем с защитой от деления на ноль
    
    # Весовые коэффициенты
    text_weight = DEFAULT_CONFIG["quality_weights"]["text_quality"]
    table_weight = DEFAULT_CONFIG["quality_weights"]["table_count"]
    
    # Общая оценка
    overall_score = text_quality * text_weight + table_score * table_weight
    
    logger.info(f"Общая оценка качества результата: {overall_score:.2f} " +
                f"(текст: {text_quality:.2f}, таблицы: {table_score:.2f})")
    
    return overall_score