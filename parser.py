"""
Основной класс парсера PDF-документов
"""

import os
import json
import string
import logging
import tempfile
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from functools import partial

import cv2
import numpy as np
import img2pdf
from pdf2image import convert_from_path
import pdfplumber
from dedoc import DedocManager

from .config import DEFAULT_CONFIG
from .models import Table
from .utils import get_logger, process_batch, merge_tables, evaluate_text_quality, evaluate_result_quality
from .image_processing import (
    binarize_image, preprocess_image, process_with_unpaper,
    detect_stamps, clean_stamp_area
)


logger = get_logger(__name__)


class PDFParser:
    """
    Класс для парсинга PDF документов с высокой точностью.
    Поддерживает работу с текстовыми PDF и сканами.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Инициализация парсера с настраиваемыми параметрами.
        
        Args:
            config: Словарь с настройками парсера
        """
        self.dedoc = DedocManager()
        
        # Копируем настройки по умолчанию
        self.config = DEFAULT_CONFIG.copy()
        
        # Обновление конфигурации пользовательскими параметрами
        if config:
            self.config.update(config)
        
        # Создание временной директории, если не указана
        if not self.config["temp_dir"]:
            self.temp_dir_obj = tempfile.TemporaryDirectory()
            self.config["temp_dir"] = self.temp_dir_obj.name
            self._is_temp_dir_managed = True
        else:
            self.temp_dir_obj = None
            self._is_temp_dir_managed = False
            os.makedirs(self.config["temp_dir"], exist_ok=True)
        
        # Список временных файлов для очистки
        self._temp_files = set()
        
        logger.info(f"Инициализирован парсер с настройками: {self.config}")
    
    def __del__(self):
        """Очистка при удалении объекта"""
        self.cleanup_temp_files()
    
    def cleanup_temp_files(self):
        """Удаляет временные файлы, созданные в процессе работы"""
        # Проверяем флаг очистки
        if self.config["clean_temp"]:
            # Удаляем созданные временные файлы
            for file_path in self._temp_files:
                try:
                    Path(file_path).unlink(missing_ok=True)
                    logger.debug(f"Удален временный файл: {file_path}")
                except Exception as e:
                    logger.warning(f"Не удалось удалить временный файл {file_path}: {e}")
            
            # Очищаем список файлов
            self._temp_files.clear()
        
        # Удаляем временную директорию, если она управлялась нами
        if hasattr(self, 'temp_dir_obj') and self.temp_dir_obj and self._is_temp_dir_managed:
            self.temp_dir_obj.cleanup()
            self.temp_dir_obj = None
    
    def create_temp_file(self, prefix: str, suffix: str) -> str:
        """
        Создает временный файл и добавляет его в список для очистки.
        
        Args:
            prefix: Префикс имени файла
            suffix: Суффикс имени файла
            
        Returns:
            str: Путь к созданному временному файлу
        """
        fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=self.config["temp_dir"])
        os.close(fd)  # Закрываем дескриптор, файл будем открывать сами
        
        # Добавляем в список для последующей очистки
        self._temp_files.add(path)
        return path
    
    def has_text_layer(self, pdf_path: str) -> bool:
        """
        Проверяет, содержит ли PDF текстовый слой.
        
        Args:
            pdf_path: Путь к PDF файлу
            
        Returns:
            bool: True, если PDF содержит текстовый слой
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Проверяем первые несколько страниц на наличие текста
                pages_to_check = min(3, len(pdf.pages))
                for i in range(pages_to_check):
                    page = pdf.pages[i]
                    text = page.extract_text()
                    if text and text.strip():
                        return True
            return False
        except Exception as e:
            logger.error(f"Ошибка при проверке текстового слоя в {pdf_path}: {e}")
            return False
    
    def process_stamp_area(self, image: np.ndarray, stamp: Dict[str, Any]) -> str:
        """
        Обрабатывает отдельно область с печатью для улучшения распознавания текста на печати.
        
        Args:
            image: Изображение в формате numpy array
            stamp: Информация о печати (координаты центра и радиус)
            
        Returns:
            str: Распознанный текст с печати
        """
        if not self.config["process_stamps_separately"]:
            return ""
        
        try:
            # Вырезаем область с печатью
            x, y, r = stamp["x"], stamp["y"], stamp["radius"]
            # Расширяем область на 10% для захвата всей печати
            expanded_r = int(r * 1.1)
            
            # Определяем границы области с печатью
            left = max(0, x - expanded_r)
            top = max(0, y - expanded_r)
            right = min(image.shape[1], x + expanded_r)
            bottom = min(image.shape[0], y + expanded_r)
            
            # Вырезаем область
            stamp_image = image[top:bottom, left:right]
            
            if stamp_image.size == 0:
                logger.warning("Пустая область печати")
                return ""
            
            # Создаем временный файл для сохранения изображения печати
            stamp_image_path = self.create_temp_file("stamp_", ".png")
            cv2.imwrite(stamp_image_path, stamp_image)
            
            # Настройки dedoc для OCR печати
            stamp_params = {
                "pdf_with_text_layer": False,
                "document_type": "other",  # обычный текст, без структуры
                "need_pdf_table_analysis": False,  # таблиц на печати нет
                "language": self.config["language"]
            }
            
            # Выполняем распознавание текста на печати
            stamp_result = self.parse_with_dedoc(stamp_image_path, stamp_params)
            
            # Извлекаем текст
            stamp_text = self.extract_text_from_dedoc_result(stamp_result)
            
            logger.info(f"Распознан текст с печати: {stamp_text[:50]}...")
            return stamp_text
        except Exception as e:
            logger.error(f"Ошибка при обработке области печати: {e}")
            return ""
    
    def convert_pdf_to_images(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Конвертирует PDF в изображения с различными вариантами обработки.
        
        Args:
            pdf_path: Путь к PDF файлу
            
        Returns:
            List[Dict]: Список словарей с путями к изображениям и их метаданными
        """
        temp_dir = Path(self.config["temp_dir"])
        dpi = self.config["dpi"]
        
        logger.info(f"Конвертация PDF в изображения с DPI={dpi}")
        
        try:
            # Конвертируем PDF в изображения
            pages = convert_from_path(pdf_path, dpi=dpi)
            
            result = []
            for i, page in enumerate(pages, start=1):
                # Создаем базовое имя файла
                base_name = f"page_{i:03d}"
                original_path = str(temp_dir / f"{base_name}_original.png")
                
                # Сохраняем оригинальное изображение
                page.save(original_path, "PNG")
                self._temp_files.add(original_path)
                
                # Создаем словарь с информацией об изображении
                page_info = {
                    "page_number": i,
                    "original_path": original_path,
                    "processed_versions": {}
                }
                
                # Конвертируем PIL изображение в numpy массив для обработки
                img_np = np.array(page)
                
                # Обнаруживаем печати, если это включено
                if self.config["detect_stamps"]:
                    stamps = detect_stamps(img_np, self.config["stamp_detection_params"])
                    if stamps:
                        page_info["stamps"] = stamps
                        
                        # Если нужно, обрабатываем области печатей отдельно
                        if self.config["process_stamps_separately"]:
                            stamp_texts = []
                            for stamp in stamps:
                                stamp_text = self.process_stamp_area(img_np, stamp)
                                if stamp_text:
                                    stamp_texts.append(stamp_text)
                            
                            if stamp_texts:
                                page_info["stamp_texts"] = stamp_texts
                            
                            # Очищаем области печатей на копии изображения для основного OCR
                            cleaned_img = img_np.copy()
                            for stamp in stamps:
                                cleaned_img = clean_stamp_area(cleaned_img, stamp)
                            
                            # Сохраняем версию без печатей
                            stamps_cleaned_path = str(temp_dir / f"{base_name}_no_stamps.png")
                            cv2.imwrite(stamps_cleaned_path, cleaned_img)
                            page_info["processed_versions"]["no_stamps"] = stamps_cleaned_path
                            self._temp_files.add(stamps_cleaned_path)
                
                # Если предобработка включена, создаем различные версии
                if self.config["preprocess_images"]:
                    # Применяем общую функцию предобработки
                    clean_img = preprocess_image(img_np, self.config)
                    clean_path = str(temp_dir / f"{base_name}_clean.png")
                    cv2.imwrite(clean_path, clean_img)
                    page_info["processed_versions"]["clean"] = clean_path
                    self._temp_files.add(clean_path)
                    
                    # Если нужно, обрабатываем unpaper
                    if self.config["use_unpaper"]:
                        unpaper_path = str(temp_dir / f"{base_name}_unpaper.png")
                        if process_with_unpaper(original_path, unpaper_path):
                            page_info["processed_versions"]["unpaper"] = unpaper_path
                            self._temp_files.add(unpaper_path)
                
                result.append(page_info)
            
            logger.info(f"Создано {len(result)} изображений страниц с вариантами обработки")
            return result
        except Exception as e:
            logger.error(f"Ошибка при конвертации PDF в изображения: {e}")
            return []
    
    def parse_with_dedoc(self, file_path: str, params: Dict[str, Any]) -> Any:
        """
        Выполняет парсинг файла с помощью dedoc и указанными параметрами.
        
        Args:
            file_path: Путь к файлу для парсинга
            params: Параметры для dedoc.parse
            
        Returns:
            Any: Результат парсинга dedoc
        """
        try:
            logger.info(f"Парсинг файла {file_path} с параметрами: {params}")
            result = self.dedoc.parse(file_path, params)
            return result
        except Exception as e:
            logger.error(f"Ошибка при парсинге через dedoc: {e}")
            return None
    
    def extract_text_from_dedoc_result(self, result: Any) -> str:
        """
        Извлекает текст из результата dedoc.
        
        Args:
            result: Результат выполнения dedoc.parse()
            
        Returns:
            str: Извлеченный текст
        """
        if not result:
            return ""
        
        try:
            # Пробуем разные поля в зависимости от структуры результата
            if hasattr(result, 'plain_text'):
                logger.debug("Извлекаем текст из поля 'plain_text'")
                return result.plain_text
            elif hasattr(result, 'text'):
                logger.debug("Извлекаем текст из поля 'text'")
                return result.text
            elif hasattr(result, 'lines'):
                logger.debug("Извлекаем текст из списка 'lines'")
                return "\n".join([line.text for line in result.lines])
            else:
                logger.warning("Не удалось найти текст в результате dedoc")
                return ""
        except Exception as e:
            logger.error(f"Ошибка при извлечении текста из результата dedoc: {e}")
            return ""
    
    def extract_tables_from_dedoc_result(self, result: Any, page_number: int = 1) -> List[Table]:
        """
        Извлекает таблицы из результата dedoc.
        
        Args:
            result: Результат выполнения dedoc.parse()
            page_number: Номер страницы (если известен)
            
        Returns:
            List[Table]: Список таблиц
        """
        if not result:
            return []
        
        tables = []
        try:
            # Пробуем разные поля в зависимости от структуры результата
            tables_data = None
            
            if hasattr(result, 'parsed_tables'):
                tables_data = result.parsed_tables
                logger.debug(f"Извлекаем таблицы из поля 'parsed_tables', найдено: {len(tables_data) if tables_data else 0}")
            elif hasattr(result, 'tables'):
                tables_data = result.tables
                logger.debug(f"Извлекаем таблицы из поля 'tables', найдено: {len(tables_data) if tables_data else 0}")
            else:
                logger.debug("В результате dedoc не найдено полей с таблицами")
            
            if tables_data:
                for table_data in tables_data:
                    table = Table(table_data, page_number)
                    if table.rows > 0 and table.cols > 0:
                        tables.append(table)
            
            logger.info(f"Извлечено {len(tables)} таблиц")
        except Exception as e:
            logger.error(f"Ошибка при извлечении таблиц из результата dedoc: {e}")
        
        return tables
    
    def compare_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Сравнивает результаты разных подходов и выбирает лучший или объединяет их.
        
        Args:
            results: Список результатов парсинга
            
        Returns:
            Dict[str, Any]: Объединенный результат
        """
        if not results:
            return {}
        
        # Если только один результат, его и возвращаем
        if len(results) == 1:
            return results[0]
        
        logger.info(f"Сравнение и объединение {len(results)} результатов парсинга")
        
        # Находим максимальное количество таблиц среди всех результатов
        max_tables = max(len(result.get("tables", [])) for result in results)
        
        # Оцениваем качество каждого результата
        result_scores = []
        for i, result in enumerate(results):
            score = evaluate_result_quality(result, max_tables)
            result_scores.append((i, score))
        
        # Сортируем по оценке качества (по убыванию)
        result_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Выбираем результат с лучшей общей оценкой
        best_result_idx = result_scores[0][0]
        best_result = results[best_result_idx]
        
        # Если есть информация о печатях, добавляем ее
        stamp_texts = []
        for result in results:
            if "stamp_texts" in result:
                stamp_texts.extend(result["stamp_texts"])
        
        if stamp_texts and "stamp_texts" not in best_result:
            best_result["stamp_texts"] = stamp_texts
        
        return best_result
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Обрабатывает PDF документ с разными подходами и возвращает результат.
        
        Args:
            pdf_path: Путь к PDF файлу
            
        Returns:
            Dict[str, Any]: Результат обработки
        """
        # Проверяем наличие текстового слоя
        has_text = self.has_text_layer(pdf_path)
        logger.info(f"PDF {pdf_path} {'содержит' if has_text else 'не содержит'} текстовый слой")
        
        results = []
        
        # Вариант 1: Если есть текстовый слой, пробуем прямой парсинг
        if has_text:
            # Используем Tabby режим для текстового PDF
            params_text = {
                "pdf_with_text_layer": "auto_tabby",  # или "tabby"
                "document_type": self.config["document_type"],
                "need_pdf_table_analysis": self.config["need_pdf_table_analysis"],
                "language": self.config["language"],
                "fast_textual_layer_detection": False  # тщательная проверка текстового слоя
            }
            
            text_result = self.parse_with_dedoc(pdf_path, params_text)
            if text_result:
                # Извлекаем текст и таблицы
                text = self.extract_text_from_dedoc_result(text_result)
                tables = self.extract_tables_from_dedoc_result(text_result)
                
                results.append({
                    "text": text,
                    "tables": tables,
                    "source": "text_layer"
                })
        
        # Вариант 2: Принудительный OCR (даже если есть текстовый слой - для сравнения)
        params_ocr = {
            "pdf_with_text_layer": False,  # игнорировать текстовый слой
            "document_type": self.config["document_type"],
            "need_pdf_table_analysis": self.config["need_pdf_table_analysis"],
            "language": self.config["language"]
        }
        
        # Если у нас PDF без текста или нужно дополнительно проверить OCR
        if not has_text or self.config["preprocess_images"]:
            # Конвертируем PDF в изображения
            page_images = self.convert_pdf_to_images(pdf_path)
            
            # Если есть обработанные изображения, делаем OCR на них
            if page_images:
                # Для каждой версии обработки создаем временный PDF для OCR
                for version in ["clean", "unpaper", "no_stamps", "original"]:
                    # Собираем изображения данной версии обработки
                    version_images = []
                    for page in page_images:
                        if version == "original":
                            version_images.append(page["original_path"])
                        elif version in page["processed_versions"]:
                            version_images.append(page["processed_versions"][version])
                    
                    if not version_images:
                        continue
                    
                    # Создаем PDF из изображений
                    temp_pdf_path = self.create_temp_file(f"temp_{version}_", ".pdf")
                    
                    try:
                        # Используем img2pdf для создания PDF
                        with open(temp_pdf_path, "wb") as f:
                            f.write(img2pdf.convert(version_images))
                        
                        # Парсим через dedoc с OCR
                        version_result = self.parse_with_dedoc(temp_pdf_path, params_ocr)
                        if version_result:
                            # Извлекаем текст и таблицы
                            text = self.extract_text_from_dedoc_result(version_result)
                            tables = self.extract_tables_from_dedoc_result(version_result)
                            
                            # Добавляем информацию о печатях, если есть
                            stamp_texts = []
                            for page in page_images:
                                if "stamp_texts" in page:
                                    stamp_texts.extend(page["stamp_texts"])
                            
                            result_data = {
                                "text": text,
                                "tables": tables,
                                "source": f"ocr_{version}"
                            }
                            
                            if stamp_texts:
                                result_data["stamp_texts"] = stamp_texts
                            
                            results.append(result_data)
                    except Exception as e:
                        logger.error(f"Ошибка при обработке версии {version}: {e}")
        else:
            # Если не обрабатываем изображения, просто запускаем OCR на исходном PDF
            ocr_result = self.parse_with_dedoc(pdf_path, params_ocr)
            if ocr_result:
                # Извлекаем текст и таблицы
                text = self.extract_text_from_dedoc_result(ocr_result)
                tables = self.extract_tables_from_dedoc_result(ocr_result)
                
                results.append({
                    "text": text,
                    "tables": tables,
                    "source": "ocr_direct"
                })
        
        # Объединяем и анализируем результаты
        if not results:
            logger.error(f"Не удалось получить ни одного результата для {pdf_path}")
            return {}
        
        final_result = self.compare_results(results)
        
        # Если в результате есть таблицы, объединяем связанные
        if "tables" in final_result and final_result["tables"]:
            final_result["tables"] = merge_tables(final_result["tables"])
        
        return final_result
    
    def process(self, pdf_path: str) -> Dict[str, Any]:
        """
        Основной метод обработки PDF документа.
        
        Args:
            pdf_path: Путь к PDF файлу
            
        Returns:
            Dict[str, Any]: Результат обработки в выбранном формате
        """
        logger.info(f"Начало обработки PDF: {pdf_path}")
        
        # Определяем размер документа
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
        except Exception as e:
            logger.error(f"Ошибка при определении размера PDF: {e}")
            page_count = 0
        
        if page_count == 0:
            logger.error(f"PDF документ пуст или не может быть открыт: {pdf_path}")
            return {}
        
        logger.info(f"PDF содержит {page_count} страниц")
        
        # Если документ большой и включена пакетная обработка,
        # разбиваем на пакеты для экономии памяти
        if page_count > self.config["batch_size"]:
            batch_size = self.config["batch_size"]
            batch_results = []
            
            # Создаем пакеты страниц
            batches = [
                (i, min(i + batch_size - 1, page_count))
                for i in range(1, page_count + 1, batch_size)
            ]
            
            logger.info(f"Документ будет обработан в {len(batches)} пакетах")
            
            # Обрабатываем пакеты
            if self.config["use_parallel"] and len(batches) > 1:
                # Параллельная обработка пакетов
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=min(self.config["max_workers"], len(batches))
                ) as executor:
                    # Создаем частичную функцию с фиксированными параметрами
                    process_batch_fn = partial(process_batch, pdf_path=pdf_path, config=self.config)
                    
                    # Запускаем обработку пакетов
                    futures = [executor.submit(process_batch_fn, page_range=batch) for batch in batches]
                    
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            batch_result = future.result()
                            if batch_result:
                                batch_results.append(batch_result)
                        except Exception as e:
                            logger.error(f"Ошибка при параллельной обработке пакета: {e}")
            else:
                # Последовательная обработка пакетов
                for batch in batches:
                    try:
                        batch_result = process_batch(pdf_path=pdf_path, page_range=batch, config=self.config)
                        if batch_result:
                            batch_results.append(batch_result)
                    except Exception as e:
                        logger.error(f"Ошибка при обработке пакета {batch}: {e}")
            
            # Объединяем результаты пакетов
            if not batch_results:
                logger.error("Не удалось обработать ни один пакет")
                return {}
            
            # Объединяем тексты и таблицы
            all_text = ""
            all_tables = []
            all_stamp_texts = []
            
            for result in batch_results:
                all_text += result.get("text", "") + "\n"
                if "tables" in result:
                    all_tables.extend(result["tables"])
                if "stamp_texts" in result:
                    all_stamp_texts.extend(result["stamp_texts"])
            
            # Сортируем таблицы по номеру страницы
            all_tables.sort(key=lambda t: t.page_number)
            
            final_result = {
                "text": all_text.strip(),
                "tables": all_tables
            }
            
            if all_stamp_texts:
                final_result["stamp_texts"] = all_stamp_texts
        else:
            # Если документ небольшой, обрабатываем целиком
            final_result = self.process_pdf(pdf_path)
        
        # Форматируем результат в нужный формат
        formatted_result = self.format_result(final_result)
        
        # Очищаем временные файлы
        self.cleanup_temp_files()
        
        logger.info(f"PDF обработан успешно: {pdf_path}")
        return formatted_result
    
    def format_result(self, result: Dict[str, Any]) -> Union[Dict[str, Any], str]:
        """
        Форматирует результат парсинга в выбранный формат.
        
        Args:
            result: Результат парсинга
            
        Returns:
            Union[Dict[str, Any], str]: Отформатированный результат
        """
        if not result:
            if self.config["output_format"] == "json":
                return {}
            else:
                return ""
        
        # Преобразование в запрошенный формат
        output_format = self.config["output_format"]
        
        logger.info(f"Форматирование результата в {output_format}")
        
        if output_format == "json":
            # Преобразуем объекты Table в словари для JSON
            if "tables" in result:
                json_tables = [table.to_dict() for table in result["tables"]]
                result["tables"] = json_tables
            
            return result
        
        elif output_format == "text":
            # Преобразуем в плоский текст
            output = []
            
            # Основной текст
            if "text" in result and result["text"]:
                output.append(result["text"])
            
            # Таблицы (в виде простого текста)
            if "tables" in result and result["tables"]:
                output.append("\n\n=== ТАБЛИЦЫ ===\n")
                
                for table in result["tables"]:
                    output.append(f"\nТаблица (страница {table.page_number}):")
                    
                    for row in table.cells_content:
                        output.append("  ".join(str(cell) if cell else "" for cell in row))
            
            # Текст с печатей
            if "stamp_texts" in result and result["stamp_texts"]:
                output.append("\n\n=== ТЕКСТ ПЕЧАТЕЙ ===\n")
                
                for i, stamp_text in enumerate(result["stamp_texts"], 1):
                    output.append(f"Печать {i}:\n{stamp_text}\n")
            
            return "\n".join(output)
        
        elif output_format == "markdown":
            # Преобразуем в markdown
            output = []
            
            # Заголовок
            output.append("# Результат парсинга PDF\n")
            
            # Основной текст
            if "text" in result and result["text"]:
                output.append("## Текст документа\n")
                output.append(result["text"])
            
            # Таблицы (в формате markdown)
            if "tables" in result and result["tables"]:
                output.append("\n## Таблицы\n")
                
                for table in result["tables"]:
                    output.append(table.to_markdown())
            
            # Текст с печатей
            if "stamp_texts" in result and result["stamp_texts"]:
                output.append("\n## Текст печатей\n")
                
                for i, stamp_text in enumerate(result["stamp_texts"], 1):
                    output.append(f"### Печать {i}\n")
                    output.append(f"```\n{stamp_text}\n```\n")
            
            return "\n".join(output)
        
        # Если неизвестный формат, возвращаем как есть
        return result