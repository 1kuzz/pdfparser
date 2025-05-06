"""
Тесты для основного класса парсера
"""

import os
import unittest
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import cv2

from pdfparser.parser import PDFParser
from pdfparser.models import Table


class TestPDFParser(unittest.TestCase):
    """Тесты для класса PDFParser"""
    
    def setUp(self):
        """Настройка тестового окружения"""
        # Создаем временную директорию для тестов
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Создаем тестовый PDF-файл
        self.test_pdf_path = os.path.join(self.temp_dir.name, "test.pdf")
        self.create_test_pdf(self.test_pdf_path)
        
        # Создаем тестовый парсер
        self.parser = PDFParser({
            "temp_dir": self.temp_dir.name,
            "clean_temp": True
        })
    
    def tearDown(self):
        """Очистка тестового окружения"""
        # Удаляем временную директорию
        self.temp_dir.cleanup()
    
    def create_test_pdf(self, path):
        """Создает тестовый PDF-файл"""
        # В реальном тесте здесь нужно создать тестовый PDF
        # Для простоты тестов просто создаем пустой файл
        with open(path, "w") as f:
            f.write("Test PDF content")
    
    def test_init(self):
        """Тест инициализации парсера"""
        # Проверяем, что парсер создан корректно
        self.assertIsNotNone(self.parser, "Парсер должен быть создан")
        self.assertEqual(self.parser.config["temp_dir"], self.temp_dir.name, 
                         "Временная директория установлена неправильно")
        self.assertTrue(self.parser.config["clean_temp"], 
                        "Флаг очистки должен быть True")
    
    def test_create_temp_file(self):
        """Тест создания временного файла"""
        # Создаем временный файл
        temp_path = self.parser.create_temp_file("test_", ".txt")
        
        # Проверяем, что файл создан
        self.assertTrue(os.path.exists(temp_path), "Временный файл должен существовать")
        
        # Проверяем, что файл добавлен в список для очистки
        self.assertIn(temp_path, self.parser._temp_files, 
                     "Временный файл должен быть добавлен в список для очистки")
    
    def test_cleanup_temp_files(self):
        """Тест очистки временных файлов"""
        # Создаем временный файл
        temp_path = self.parser.create_temp_file("test_", ".txt")
        
        # Проверяем, что файл создан
        self.assertTrue(os.path.exists(temp_path), "Временный файл должен существовать")
        
        # Вызываем очистку
        self.parser.cleanup_temp_files()
        
        # Проверяем, что файл удален
        self.assertFalse(os.path.exists(temp_path), "Временный файл должен быть удален")
        
        # Проверяем, что список для очистки пуст
        self.assertEqual(len(self.parser._temp_files), 0, 
                        "Список для очистки должен быть пуст")
    
    @patch('pdfparser.parser.pdfplumber.open')
    def test_has_text_layer_true(self, mock_pdfplumber_open):
        """Тест проверки наличия текстового слоя (положительный случай)"""
        # Настраиваем мок
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Test text"
        mock_pdf.pages = [mock_page]
        mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf
        
        # Вызываем метод
        result = self.parser.has_text_layer(self.test_pdf_path)
        
        # Проверяем результат
        self.assertTrue(result, "Должно быть определено наличие текстового слоя")
    
    @patch('pdfparser.parser.pdfplumber.open')
    def test_has_text_layer_false(self, mock_pdfplumber_open):
        """Тест проверки наличия текстового слоя (отрицательный случай)"""
        # Настраиваем мок
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_pdf.pages = [mock_page]
        mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf
        
        # Вызываем метод
        result = self.parser.has_text_layer(self.test_pdf_path)
        
        # Проверяем результат
        self.assertFalse(result, "Должно быть определено отсутствие текстового слоя")
    
    @patch('pdfparser.parser.DedocManager')
    def test_parse_with_dedoc(self, mock_dedoc_manager):
        """Тест парсинга с помощью dedoc"""
        # Настраиваем мок
        mock_dedoc_instance = MagicMock()
        mock_dedoc_instance.parse.return_value = "Test dedoc result"
        mock_dedoc_manager.return_value = mock_dedoc_instance
        
        # Устанавливаем dedoc в парсере
        self.parser.dedoc = mock_dedoc_instance
        
        # Вызываем метод
        result = self.parser.parse_with_dedoc(self.test_pdf_path, {"param": "value"})
        
        # Проверяем результат
        self.assertEqual(result, "Test dedoc result", "Результат должен быть от dedoc")
        mock_dedoc_instance.parse.assert_called_once_with(self.test_pdf_path, {"param": "value"})
    
    def test_extract_text_from_dedoc_result(self):
        """Тест извлечения текста из результата dedoc"""
        # Создаем тестовые данные
        class MockResultWithPlainText:
            plain_text = "Plain text"
        
        class MockResultWithText:
            text = "Regular text"
        
        class MockLine:
            def __init__(self, text):
                self.text = text
        
        class MockResultWithLines:
            lines = [MockLine("Line 1"), MockLine("Line 2")]
        
        # Тестируем разные варианты
        self.assertEqual(
            self.parser.extract_text_from_dedoc_result(MockResultWithPlainText()),
            "Plain text",
            "Должен использоваться plain_text"
        )
        
        self.assertEqual(
            self.parser.extract_text_from_dedoc_result(MockResultWithText()),
            "Regular text",
            "Должен использоваться text"
        )
        
        self.assertEqual(
            self.parser.extract_text_from_dedoc_result(MockResultWithLines()),
            "Line 1\nLine 2",
            "Должны объединяться строки из lines"
        )
        
        self.assertEqual(
            self.parser.extract_text_from_dedoc_result(None),
            "",
            "Должна возвращаться пустая строка для None"
        )
    
    def test_format_result_json(self):
        """Тест форматирования результата в JSON"""
        # Создаем тестовые данные
        table = MagicMock(spec=Table)
        table.to_dict.return_value = {"page_number": 1, "rows": 3, "cols": 4}
        
        result = {
            "text": "Test text",
            "tables": [table],
            "stamp_texts": ["Stamp 1", "Stamp 2"]
        }
        
        # Устанавливаем формат вывода
        self.parser.config["output_format"] = "json"
        
        # Вызываем метод
        formatted = self.parser.format_result(result)
        
        # Проверяем результат
        self.assertIsInstance(formatted, dict, "Результат должен быть словарем")
        self.assertEqual(formatted["text"], "Test text", "Текст должен остаться без изменений")
        self.assertEqual(len(formatted["tables"]), 1, "Должна быть одна таблица")
        self.assertEqual(formatted["tables"][0], {"page_number": 1, "rows": 3, "cols": 4}, 
                         "Таблица должна быть преобразована в словарь")
    
    def test_format_result_text(self):
        """Тест форматирования результата в текст"""
        # Создаем тестовые данные
        table = MagicMock(spec=Table)
        table.page_number = 1
        table.cells_content = [["A1", "B1"], ["A2", "B2"]]
        
        result = {
            "text": "Test text",
            "tables": [table],
            "stamp_texts": ["Stamp 1", "Stamp 2"]
        }
        
        # Устанавливаем формат вывода
        self.parser.config["output_format"] = "text"
        
        # Вызываем метод
        formatted = self.parser.format_result(result)
        
        # Проверяем результат
        self.assertIsInstance(formatted, str, "Результат должен быть строкой")
        self.assertIn("Test text", formatted, "Текст должен остаться без изменений")
        self.assertIn("Таблица (страница 1)", formatted, "Должна быть информация о таблице")
        self.assertIn("Печать 1", formatted, "Должна быть информация о печати")
    
    def test_format_result_markdown(self):
        """Тест форматирования результата в markdown"""
        # Создаем тестовые данные
        table = MagicMock(spec=Table)
        table.to_markdown.return_value = "| A1 | B1 |\n| --- | --- |\n| A2 | B2 |"
        
        result = {
            "text": "Test text",
            "tables": [table],
            "stamp_texts": ["Stamp 1", "Stamp 2"]
        }
        
        # Устанавливаем формат вывода
        self.parser.config["output_format"] = "markdown"
        
        # Вызываем метод
        formatted = self.parser.format_result(result)
        
        # Проверяем результат
        self.assertIsInstance(formatted, str, "Результат должен быть строкой")
        self.assertIn("# Результат парсинга PDF", formatted, "Должен быть заголовок")
        self.assertIn("## Текст документа", formatted, "Должен быть раздел с текстом")
        self.assertIn("## Таблицы", formatted, "Должен быть раздел с таблицами")
        self.assertIn("| A1 | B1 |", formatted, "Должна быть таблица в формате markdown")
        self.assertIn("### Печать 1", formatted, "Должна быть информация о печати")


if __name__ == '__main__':
    unittest.main()