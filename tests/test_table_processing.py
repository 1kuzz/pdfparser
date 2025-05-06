"""
Тесты для модуля обработки таблиц
"""

import unittest
from unittest.mock import MagicMock

from pdfparser.models import Table
from pdfparser.utils import merge_tables


class TestTableProcessing(unittest.TestCase):
    """Тесты для функций работы с таблицами"""
    
    def create_mock_table(self, page_number, rows, cols, has_top_border=True, has_bottom_border=True, bbox=None):
        """Создает фиктивную таблицу для тестирования"""
        mock_data = MagicMock()
        mock_data.cells = []
        
        # Создаем тестовую таблицу
        table = Table(mock_data, page_number)
        table.rows = rows
        table.cols = cols
        table.has_top_border = has_top_border
        table.has_bottom_border = has_bottom_border
        table.cells_content = [[f"r{r}c{c}" for c in range(cols)] for r in range(rows)]
        
        # Устанавливаем bbox для таблицы
        if bbox:
            table.bbox = bbox
        else:
            table.bbox = [100, 100, 400, 200]
        
        return table
    
    def test_can_merge_with_sequential_pages(self):
        """Тест проверки возможности слияния таблиц на последовательных страницах"""
        # Создаем таблицы на последовательных страницах
        table1 = self.create_mock_table(page_number=1, rows=3, cols=4, 
                                        has_bottom_border=False)
        table2 = self.create_mock_table(page_number=2, rows=2, cols=4, 
                                        has_top_border=False)
        
        # Проверяем, что их можно объединить
        self.assertTrue(table1.can_merge_with(table2), 
                        "Таблицы должны быть помечены как возможные для слияния")
    
    def test_cannot_merge_with_non_sequential_pages(self):
        """Тест проверки невозможности слияния таблиц на непоследовательных страницах"""
        # Создаем таблицы на непоследовательных страницах
        table1 = self.create_mock_table(page_number=1, rows=3, cols=4, 
                                        has_bottom_border=False)
        table3 = self.create_mock_table(page_number=3, rows=2, cols=4, 
                                        has_top_border=False)
        
        # Проверяем, что их нельзя объединить
        self.assertFalse(table1.can_merge_with(table3), 
                         "Таблицы на непоследовательных страницах не должны объединяться")
    
    def test_cannot_merge_with_different_columns(self):
        """Тест проверки невозможности слияния таблиц с разным числом столбцов"""
        # Создаем таблицы с разным числом столбцов
        table1 = self.create_mock_table(page_number=1, rows=3, cols=4, 
                                        has_bottom_border=False)
        table2 = self.create_mock_table(page_number=2, rows=2, cols=5, 
                                        has_top_border=False)
        
        # Проверяем, что их нельзя объединить
        self.assertFalse(table1.can_merge_with(table2), 
                         "Таблицы с разным числом столбцов не должны объединяться")
    
    def test_cannot_merge_with_borders(self):
        """Тест проверки невозможности слияния таблиц с границами"""
        # Создаем таблицы с границами
        table1 = self.create_mock_table(page_number=1, rows=3, cols=4, 
                                        has_bottom_border=True)
        table2 = self.create_mock_table(page_number=2, rows=2, cols=4, 
                                        has_top_border=False)
        
        # Проверяем, что их нельзя объединить
        self.assertFalse(table1.can_merge_with(table2), 
                         "Таблицы с нижней границей не должны объединяться")
        
        table1.has_bottom_border = False
        table2.has_top_border = True
        
        # Проверяем, что их нельзя объединить
        self.assertFalse(table1.can_merge_with(table2), 
                         "Таблицы с верхней границей не должны объединяться")
    
    def test_merge_with(self):
        """Тест слияния двух таблиц"""
        # Создаем таблицы, которые можно объединить
        table1 = self.create_mock_table(page_number=1, rows=3, cols=4, 
                                        has_bottom_border=False)
        table2 = self.create_mock_table(page_number=2, rows=2, cols=4, 
                                        has_top_border=False)
        
        # Объединяем таблицы
        merged = table1.merge_with(table2)
        
        # Проверяем результат
        self.assertEqual(merged.rows, table1.rows + table2.rows, 
                         "Merged table should have combined rows")
        self.assertEqual(merged.cols, table1.cols, 
                         "Merged table should have the same number of columns")
        self.assertEqual(merged.page_number, table1.page_number, 
                         "Merged table should have the page number of the first table")
        
        # Проверяем содержимое объединенной таблицы
        self.assertEqual(len(merged.cells_content), table1.rows + table2.rows, 
                         "Merged table cells content should have combined rows")
        
        # Проверяем, что содержимое первой и второй таблицы сохранилось
        for r in range(table1.rows):
            for c in range(table1.cols):
                self.assertEqual(merged.cells_content[r][c], table1.cells_content[r][c], 
                                "Content from first table should be preserved")
        
        for r in range(table2.rows):
            for c in range(table2.cols):
                self.assertEqual(merged.cells_content[table1.rows + r][c], table2.cells_content[r][c], 
                                "Content from second table should be preserved")
    
  def test_merge_tables_function(self):
        """Тест функции merge_tables для слияния нескольких таблиц"""
        # Создаем таблицы, которые можно объединить последовательно
        table1 = self.create_mock_table(page_number=1, rows=3, cols=4, 
                                        has_bottom_border=False)
        table2 = self.create_mock_table(page_number=2, rows=2, cols=4, 
                                        has_top_border=False, has_bottom_border=False)
        table3 = self.create_mock_table(page_number=3, rows=2, cols=4, 
                                        has_top_border=False)
        
        # Создаем таблицу, которая не должна объединяться
        table4 = self.create_mock_table(page_number=5, rows=3, cols=4)
        
        # Объединяем таблицы
        tables = [table1, table2, table3, table4]
        merged_tables = merge_tables(tables)
        
        # Проверяем результат
        self.assertEqual(len(merged_tables), 2, 
                         "Должно получиться две таблицы: объединенная из трех и отдельная")
        
        # Первая таблица должна быть объединенной из трех
        self.assertEqual(merged_tables[0].rows, table1.rows + table2.rows + table3.rows, 
                         "Первая таблица должна иметь строки из трех таблиц")
        
        # Вторая таблица должна быть отдельной
        self.assertEqual(merged_tables[1].page_number, table4.page_number, 
                         "Вторая таблица должна быть с отдельной страницы")
    
    def test_merge_tables_empty_list(self):
        """Тест функции merge_tables с пустым списком"""
        # Вызываем функцию с пустым списком
        result = merge_tables([])
        
        # Проверяем, что результат также пустой список
        self.assertEqual(result, [], "Результат должен быть пустым списком")
    
    def test_merge_tables_single_table(self):
        """Тест функции merge_tables с одной таблицей"""
        # Создаем одну таблицу
        table = self.create_mock_table(page_number=1, rows=3, cols=4)
        
        # Вызываем функцию с одной таблицей
        result = merge_tables([table])
        
        # Проверяем, что результат содержит ту же таблицу
        self.assertEqual(len(result), 1, "Результат должен содержать одну таблицу")
        self.assertEqual(result[0].page_number, table.page_number, 
                         "Таблица должна быть той же самой")


if __name__ == '__main__':
    unittest.main()