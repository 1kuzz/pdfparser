"""
Модели данных для работы с PDF-документами
"""

from typing import Dict, List, Tuple, Optional, Any, Union


class Table:
    """
    Класс-обёртка для таблиц, обнаруженных в документе.
    Помогает с обработкой и слиянием таблиц, особенно многостраничных.
    """
    
    def __init__(self, table_data: Any, page_number: int):
        """
        Инициализация объекта таблицы на основе данных dedoc.
        
        Args:
            table_data: Объект таблицы из dedoc
            page_number: Номер страницы, на которой расположена таблица
        """
        self.original_data = table_data
        self.page_number = page_number
        
        # Извлекаем основные свойства таблицы
        # Примечание: структура может отличаться в зависимости от версии dedoc
        try:
            # Извлекаем кол-во строк и столбцов
            if hasattr(table_data, 'cells') and table_data.cells:
                # Определяем размерность таблицы по максимальным индексам
                max_row = max([cell.row_index for cell in table_data.cells], default=0)
                max_col = max([cell.col_index for cell in table_data.cells], default=0)
                self.rows = max_row + 1
                self.cols = max_col + 1
                
                # Собираем содержимое ячеек в матрицу
                self.cells_content = [[None for _ in range(self.cols)] for _ in range(self.rows)]
                for cell in table_data.cells:
                    row_idx, col_idx = cell.row_index, cell.col_index
                    if 0 <= row_idx < self.rows and 0 <= col_idx < self.cols:
                        self.cells_content[row_idx][col_idx] = cell.text
            else:
                # Запасной вариант, если структура отличается
                self.rows = getattr(table_data, 'rows_count', 0)
                self.cols = getattr(table_data, 'cols_count', 0)
                self.cells_content = getattr(table_data, 'cells_content', [])
                
            # Координаты таблицы (bbox)
            self.bbox = getattr(table_data, 'bbox', None)
            
            # Проверка наличия верхней и нижней границы (для объединения таблиц)
            # Если этих атрибутов нет, задаем значения по умолчанию
            self.has_top_border = getattr(table_data, 'has_top_border', True)
            self.has_bottom_border = getattr(table_data, 'has_bottom_border', True)
        
        except Exception as e:
            from .utils import get_logger
            logger = get_logger(__name__)
            logger.warning(f"Ошибка при создании объекта таблицы: {e}")
            self.rows = 0
            self.cols = 0
            self.cells_content = []
            self.bbox = None
            self.has_top_border = True
            self.has_bottom_border = True
    
    def can_merge_with(self, other: 'Table') -> bool:
        """
        Проверяет, может ли текущая таблица быть объединена с другой
        (вероятно, это продолжение одной таблицы на следующей странице).
        
        Args:
            other: Другая таблица для проверки возможности слияния
            
        Returns:
            bool: True, если таблицы могут быть объединены
        """
        # Основные критерии для объединения:
        # 1. Таблицы должны быть на последовательных страницах
        # 2. Количество столбцов должно совпадать
        # 3. Первая таблица не должна иметь нижней границы, вторая - верхней
        # 4. Ширина таблиц должна быть примерно одинаковой (если есть bbox)
        
        if self.page_number + 1 != other.page_number:
            return False
        
        if self.cols != other.cols:
            return False
        
        if self.has_bottom_border or other.has_top_border:
            return False
        
        # Если есть bbox, проверяем перекрытие по X
        if self.bbox and other.bbox:
            # Вычисляем перекрытие по X (в %)
            self_width = self.bbox[2] - self.bbox[0]
            other_width = other.bbox[2] - other.bbox[0]
            
            # Проверяем, насколько центры таблиц близки друг к другу
            self_center_x = (self.bbox[0] + self.bbox[2]) / 2
            other_center_x = (other.bbox[0] + other.bbox[2]) / 2
            
            # Допустимое отклонение - до 15% от ширины
            max_deviation = max(self_width, other_width) * 0.15
            
            if abs(self_center_x - other_center_x) > max_deviation:
                return False
        
        return True
    
    def merge_with(self, other: 'Table') -> 'Table':
        """
        Объединяет текущую таблицу с другой таблицей.
        
        Args:
            other: Таблица для объединения
            
        Returns:
            Table: Новая таблица, содержащая объединенные данные
        """
        # Создаем копию текущей таблицы
        merged = Table(self.original_data, self.page_number)
        
        # Объединяем содержимое ячеек
        merged.cells_content = self.cells_content + other.cells_content
        merged.rows = self.rows + other.rows
        merged.cols = max(self.cols, other.cols)
        
        # Оставляем нижнюю границу от второй таблицы
        merged.has_bottom_border = other.has_bottom_border
        
        # Если есть bbox, объединяем его
        if self.bbox and other.bbox:
            merged.bbox = [
                min(self.bbox[0], other.bbox[0]),
                self.bbox[1],
                max(self.bbox[2], other.bbox[2]),
                other.bbox[3]
            ]
        
        return merged
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразует таблицу в словарь для JSON-сериализации.
        
        Returns:
            Dict[str, Any]: Словарь с данными таблицы
        """
        return {
            "page_number": self.page_number,
            "rows": self.rows,
            "cols": self.cols,
            "cells": self.cells_content,
            "has_top_border": self.has_top_border,
            "has_bottom_border": self.has_bottom_border,
            "bbox": self.bbox
        }
    
    def to_markdown(self) -> str:
        """
        Преобразует таблицу в формат Markdown.
        
        Returns:
            str: Таблица в формате Markdown
        """
        if not self.cells_content or not self.rows or not self.cols:
            return ""
        
        result = []
        
        # Добавляем заголовок страницы
        result.append(f"\n### Таблица (страница {self.page_number})\n")
        
        # Формируем строки таблицы
        for row_idx, row in enumerate(self.cells_content):
            # Создаем строку с разделителями ячеек
            cells_text = []
            for col_idx in range(len(row)):
                cell_content = row[col_idx] if col_idx < len(row) and row[col_idx] else ""
                cell_content = str(cell_content).replace('|', '\\|')  # Экранируем символы |
                cells_text.append(cell_content)
            
            row_text = "| " + " | ".join(cells_text) + " |"
            result.append(row_text)
            
            # После первой строки добавляем разделитель заголовка
            if row_idx == 0:
                header_sep = "| " + " | ".join(["---" for _ in range(len(row))]) + " |"
                result.append(header_sep)
        
        # Объединяем в текст
        return "\n".join(result)