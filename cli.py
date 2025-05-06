"""
Интерфейс командной строки для парсера PDF
"""

import os
import json
import argparse
import sys
from typing import Dict, Any

from .config import DEFAULT_CONFIG
from .parser import PDFParser
from .utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Разбор аргументов командной строки.
    
    Returns:
        argparse.Namespace: Объект с аргументами
    """
    parser = argparse.ArgumentParser(
        description="Парсер PDF документов с высокой точностью для русского языка"
    )
    
    parser.add_argument(
        "input_file", 
        help="Путь к входному PDF файлу"
    )
    
    parser.add_argument(
        "-o", "--output", 
        help="Путь для сохранения результата (по умолчанию: input_file_result.{format})",
        default=None
    )
    
    parser.add_argument(
        "-f", "--format", 
        help="Формат вывода (json, text, markdown)",
        choices=["json", "text", "markdown"],
        default="json"
    )
    
    parser.add_argument(
        "-d", "--dpi", 
        help="DPI для конвертации PDF в изображения",
        type=int,
        default=300
    )
    
    parser.add_argument(
        "--document-type", 
        help="Тип документа для dedoc",
        choices=["tz", "other", "law", "article"],
        default="tz"
    )
    
    parser.add_argument(
        "--no-preprocess", 
        help="Отключить предобработку изображений",
        action="store_true"
    )
    
    parser.add_argument(
        "--detect-stamps", 
        help="Включить обнаружение печатей",
        action="store_true"
    )
    
    parser.add_argument(
        "--process-stamps", 
        help="Обрабатывать области печатей отдельно",
        action="store_true"
    )
    
    parser.add_argument(
        "--use-unpaper", 
        help="Использовать unpaper для очистки сканов",
        action="store_true"
    )
    
    parser.add_argument(
        "--binarize-method", 
        help="Метод бинаризации",
        choices=["otsu", "sauvola"],
        default="sauvola"
    )
    
    parser.add_argument(
        "--temp-dir", 
        help="Директория для временных файлов",
        default=None
    )
    
    parser.add_argument(
        "--no-clean-temp", 
        help="Не удалять временные файлы после обработки",
        action="store_true"
    )
    
    parser.add_argument(
        "--no-parallel", 
        help="Отключить параллельную обработку",
        action="store_true"
    )
    
    parser.add_argument(
        "--batch-size", 
        help="Размер пакета при обработке больших документов",
        type=int,
        default=10
    )
    
    parser.add_argument(
        "--config", 
        help="Путь к файлу конфигурации JSON",
        default=None
    )
    
    parser.add_argument(
        "--debug", 
        help="Включить отладочный режим",
        action="store_true"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Загружает конфигурацию из JSON-файла.
    
    Args:
        config_path: Путь к файлу конфигурации
    
    Returns:
        Dict[str, Any]: Словарь с настройками
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Загружена конфигурация из файла: {config_path}")
        return config
    except Exception as e:
        logger.error(f"Ошибка при загрузке конфигурации из {config_path}: {e}")
        return {}


def main() -> int:
    """
    Основная функция для запуска парсера из командной строки.
    
    Returns:
        int: Код завершения (0 - успех, 1 - ошибка)
    """
    args = parse_args()
    
    # Настройка уровня логирования
    if args.debug:
        logger.setLevel("DEBUG")
    
    # Проверка существования входного файла
    if not os.path.isfile(args.input_file):
        logger.error(f"Входной файл не существует: {args.input_file}")
        return 1
    
    # Определение выходного файла
    if not args.output:
        input_base = os.path.splitext(args.input_file)[0]
        args.output = f"{input_base}_result.{args.format}"
    
    # Загрузка конфигурации из файла, если указан
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Обновление конфигурации аргументами командной строки
    config.update({
        "dpi": args.dpi,
        "document_type": args.document_type,
        "preprocess_images": not args.no_preprocess,
        "output_format": args.format,
        "temp_dir": args.temp_dir,
        "clean_temp": not args.no_clean_temp,
        "detect_stamps": args.detect_stamps,
        "process_stamps_separately": args.process_stamps,
        "use_unpaper": args.use_unpaper,
        "binarize_method": args.binarize_method,
        "use_parallel": not args.no_parallel,
        "batch_size": args.batch_size
    })
    
    try:
        # Создание и запуск парсера
        parser = PDFParser(config)
        result = parser.process(args.input_file)
        
        # Сохранение результата
        with open(args.output, "w", encoding="utf-8") as f:
            if args.format == "json":
                json.dump(result, f, ensure_ascii=False, indent=2)
            else:
                # Для других форматов результат уже в виде строки
                f.write(result)
        
        logger.info(f"Результат сохранен в {args.output}")
        
        # Явный вызов очистки временных файлов
        parser.cleanup_temp_files()
        
        return 0
    except Exception as e:
        logger.exception(f"Ошибка при обработке файла: {e}")
        
        # Пытаемся очистить временные файлы даже при ошибке
        if 'parser' in locals():
            parser.cleanup_temp_files()
            
        return 1


if __name__ == "__main__":
    sys.exit(main())