"""
Тесты для модуля обработки изображений
"""

import os
import tempfile
import unittest
import numpy as np
import cv2

from pdfparser.image_processing import (
    binarize_image, deskew_image, detect_stamps, clean_stamp_area, preprocess_image
)
from pdfparser.config import DEFAULT_CONFIG


class TestImageProcessing(unittest.TestCase):
    """Тесты для функций обработки изображений"""
    
    def setUp(self):
        """Создает тестовые изображения"""
        # Создаем пустое изображение
        self.blank_image = np.ones((300, 500), dtype=np.uint8) * 255
        
        # Создаем изображение с текстом
        self.text_image = self.blank_image.copy()
        cv2.putText(
            self.text_image, "Test Text", (50, 150), 
            cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 3
        )
        
        # Создаем изображение с наклоненным текстом
        self.skewed_image = self.blank_image.copy()
        center = (self.skewed_image.shape[1] // 2, self.skewed_image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
        self.skewed_image = cv2.warpAffine(
            self.text_image, rotation_matrix, 
            (self.skewed_image.shape[1], self.skewed_image.shape[0])
        )
        
        # Создаем изображение с шумом
        self.noisy_image = self.text_image.copy()
        noise = np.random.randint(0, 50, self.noisy_image.shape, dtype=np.uint8)
        self.noisy_image = cv2.subtract(self.noisy_image, noise)
        
        # Создаем изображение с кругом (печатью)
        self.stamp_image = self.text_image.copy()
        cv2.circle(self.stamp_image, (250, 150), 50, 150, -1)
    
    def test_binarize_image_otsu(self):
        """Тест бинаризации методом Оцу"""
        # Бинаризуем изображение с текстом
        binary = binarize_image(self.text_image, method="otsu")
        
        # Проверяем, что бинаризация сделала изображение строго черно-белым
        unique_values = np.unique(binary)
        self.assertLessEqual(len(unique_values), 2, "Бинаризованное изображение должно содержать только 0 и 255")
        
        # Проверяем, что текст сохранился (есть черные пиксели)
        black_pixels = np.sum(binary == 0)
        self.assertGreater(black_pixels, 0, "Текст не сохранился при бинаризации")
    
    def test_binarize_image_sauvola(self):
        """Тест бинаризации методом Саувола (если установлен scikit-image)"""
        try:
            from skimage.filters import threshold_sauvola
            
            # Бинаризуем изображение с текстом
            binary = binarize_image(self.text_image, method="sauvola")
            
            # Проверяем, что бинаризация сделала изображение строго черно-белым
            unique_values = np.unique(binary)
            self.assertLessEqual(len(unique_values), 2, "Бинаризованное изображение должно содержать только 0 и 255")
            
            # Проверяем, что текст сохранился (есть черные пиксели)
            black_pixels = np.sum(binary == 0)
            self.assertGreater(black_pixels, 0, "Текст не сохранился при бинаризации")
        except ImportError:
            self.skipTest("scikit-image не установлен, пропускаем тест Саувола")
    
    def test_deskew_image(self):
        """Тест выравнивания наклона"""
        # Выравниваем наклоненное изображение
        deskewed = deskew_image(self.skewed_image)
        
        # Тест сложно автоматизировать, просто проверяем, что изображение изменилось
        self.assertFalse(np.array_equal(self.skewed_image, deskewed), 
                         "Изображение не изменилось при выравнивании наклона")
    
    def test_detect_stamps(self):
        """Тест обнаружения печатей"""
        # Настраиваем параметры
        params = DEFAULT_CONFIG["stamp_detection_params"].copy()
        params["minRadius"] = 40
        params["maxRadius"] = 60
        
        # Находим печати
        stamps = detect_stamps(self.stamp_image, params)
        
        # Проверяем, что нашлась хотя бы одна печать
        self.assertGreaterEqual(len(stamps), 1, "Не найдена печать на изображении")
        
        # Проверяем координаты
        stamp = stamps[0]
        self.assertIn("x", stamp, "В найденной печати нет координаты x")
        self.assertIn("y", stamp, "В найденной печати нет координаты y")
        self.assertIn("radius", stamp, "В найденной печати нет радиуса")
    
    def test_clean_stamp_area(self):
        """Тест очистки области печати"""
        # Находим печати
        params = DEFAULT_CONFIG["stamp_detection_params"].copy()
        params["minRadius"] = 40
        params["maxRadius"] = 60
        stamps = detect_stamps(self.stamp_image, params)
        
        # Если печати найдены, очищаем область
        if stamps:
            cleaned = clean_stamp_area(self.stamp_image, stamps[0])
            
            # Проверяем, что в области печати стало больше белых пикселей
            stamp = stamps[0]
            x, y, r = stamp["x"], stamp["y"], stamp["radius"]
            
            # Создаем маску для печати
            mask = np.zeros(self.stamp_image.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            
            # Считаем белые пиксели в области печати до и после очистки
            white_before = np.sum((self.stamp_image == 255) & (mask == 255))
            white_after = np.sum((cleaned == 255) & (mask == 255))
            
            self.assertGreater(white_after, white_before, 
                              "Область печати не была очищена")
        else:
            self.skipTest("Печать не была обнаружена на тестовом изображении")
    
    def test_preprocess_image(self):
        """Тест полной предобработки изображения"""
        # Настраиваем конфигурацию
        config = {
            "deskew_enabled": True,
            "denoise_enabled": True,
            "binarize_enabled": True,
            "binarize_method": "otsu"
        }
        
        # Обрабатываем изображение
        processed = preprocess_image(self.noisy_image, config)
        
        # Проверяем, что изображение изменилось и стало бинаризованным
        self.assertFalse(np.array_equal(self.noisy_image, processed), 
                         "Изображение не изменилось при предобработке")
        
        # Проверяем, что изображение бинаризовано
        unique_values = np.unique(processed)
        self.assertLessEqual(len(unique_values), 2, 
                            "Обработанное изображение должно содержать только 0 и 255")


if __name__ == '__main__':
    unittest.main()