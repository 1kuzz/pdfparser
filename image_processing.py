"""
Функции для обработки и предобработки изображений
"""

import os
import subprocess
from typing import Dict, List, Tuple, Optional, Any, Union

import cv2
import numpy as np

from .utils import get_logger

logger = get_logger(__name__)


def binarize_image(image: np.ndarray, method: str = "otsu", window_size: int = 25) -> np.ndarray:
    """
    Бинаризует изображение с использованием указанного метода.
    
    Args:
        image: Входное изображение
        method: Метод бинаризации ("otsu" или "sauvola")
        window_size: Размер окна для метода Саувола
        
    Returns:
        np.ndarray: Бинаризованное изображение
    """
    # Преобразуем в оттенки серого, если изображение цветное
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Применяем выбранный метод бинаризации
    if method.lower() == "sauvola":
        try:
            # Импортируем и используем Саувола из scikit-image
            from skimage.filters import threshold_sauvola
            logger.debug(f"Применяем бинаризацию методом Саувола с размером окна {window_size}")
            
            thresh_sauvola = threshold_sauvola(gray, window_size=window_size)
            binary = (gray > thresh_sauvola).astype(np.uint8) * 255
        except ImportError:
            logger.warning("scikit-image не установлен, используем метод Оцу")
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:  # otsu
        logger.debug("Применяем бинаризацию методом Оцу")
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Удаляем мелкий шум
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary


def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Выравнивает наклон текста на изображении.
    
    Args:
        image: Входное изображение
        
    Returns:
        np.ndarray: Выровненное изображение
    """
    # Преобразуем в оттенки серого, если изображение цветное
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    try:
        # Обнаружение линий для определения угла наклона
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is not None and len(lines) > 0:
            # Вычисление среднего угла
            angles = []
            for line in lines:
                rho, theta = line[0]
                if theta < np.pi/4 or theta > 3*np.pi/4:  # Вертикальные линии
                    angles.append(theta)
            
            if angles:
                avg_angle = np.mean(angles)
                angle_degrees = np.degrees(avg_angle - np.pi/2)
                
                # Применяем вращение, если угол значителен
                if abs(angle_degrees) > 0.5:
                    (h, w) = gray.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
                    result = cv2.warpAffine(
                        image if len(image.shape) == 3 else gray, 
                        M, (w, h), 
                        flags=cv2.INTER_CUBIC, 
                        borderMode=cv2.BORDER_REPLICATE
                    )
                    logger.info(f"Изображение выровнено на {angle_degrees:.2f} градусов")
                    return result
    except Exception as e:
        logger.warning(f"Ошибка при выравнивании наклона: {e}")
    
    return image


def process_with_unpaper(image_path: str, output_path: str) -> bool:
    """
    Обрабатывает изображение с помощью unpaper для улучшения качества.
    
    Args:
        image_path: Путь к исходному изображению
        output_path: Путь для сохранения обработанного изображения
        
    Returns:
        bool: True, если обработка успешна
    """
    try:
        # Формируем команду unpaper с нужными флагами
        cmd = [
            "unpaper",
            "--layout", "single",
            "--deskew",
            "--no-noisefilter",  # Отключаем шумоподавление, чтобы не потерять печати
            image_path,
            output_path
        ]
        
        # Выполняем команду
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Ошибка при обработке unpaper: {result.stderr}")
            return False
        
        logger.info(f"Изображение успешно обработано с помощью unpaper: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при запуске unpaper: {e}")
        return False


def detect_stamps(image: np.ndarray, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Обнаруживает круглые печати на изображении.
    
    Args:
        image: Изображение в формате numpy array
        params: Параметры обнаружения кругов для cv2.HoughCircles
        
    Returns:
        List[Dict]: Список с информацией о найденных печатях (координаты, радиус)
    """
    stamps = []
    try:
        # Конвертируем в оттенки серого
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Применяем размытие для улучшения обнаружения кругов
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Используем преобразование Хафа для обнаружения кругов
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=params["dp"], 
            minDist=params["minDist"],
            param1=params["param1"], 
            param2=params["param2"], 
            minRadius=params["minRadius"], 
            maxRadius=params["maxRadius"]
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            for (x, y, r) in circles:
                stamps.append({
                    "x": int(x),
                    "y": int(y),
                    "radius": int(r)
                })
            
            logger.info(f"Обнаружено {len(stamps)} печатей на изображении")
    except Exception as e:
        logger.error(f"Ошибка при поиске печатей: {e}")
    
    return stamps


def clean_stamp_area(image: np.ndarray, stamp: Dict[str, Any]) -> np.ndarray:
    """
    Очищает область печати на изображении для улучшения распознавания основного текста.
    
    Args:
        image: Изображение в формате numpy array
        stamp: Информация о печати (координаты центра и радиус)
        
    Returns:
        np.ndarray: Изображение с очищенной областью печати
    """
    # Создаем копию изображения
    cleaned_image = image.copy()
    
    # Вырезаем область с печатью
    x, y, r = stamp["x"], stamp["y"], stamp["radius"]
    
    # Определяем границы области с печатью
    left = max(0, x - r)
    top = max(0, y - r)
    right = min(image.shape[1], x + r)
    bottom = min(image.shape[0], y + r)
    
    # Создаем маску для печати (круг)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    
    # Ограничиваем маску областью изображения
    mask = mask[top:bottom, left:right]
    
    # Среднее значение фона (обычно белый для документов)
    background_color = 255
    
    # Заменяем пиксели в области печати на фоновый цвет
    if len(cleaned_image.shape) == 3:  # Цветное изображение
        for c in range(3):  # Для каждого канала
            cleaned_image[top:bottom, left:right, c] = np.where(
                mask > 0, background_color, cleaned_image[top:bottom, left:right, c]
            )
    else:  # Оттенки серого
        cleaned_image[top:bottom, left:right] = np.where(
            mask > 0, background_color, cleaned_image[top:bottom, left:right]
        )
    
    return cleaned_image


def preprocess_image(image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    Предварительная обработка изображения для улучшения OCR.
    
    Args:
        image: Изображение в формате numpy array
        config: Конфигурация обработки
        
    Returns:
        np.ndarray: Обработанное изображение
    """
    # Преобразование в оттенки серого, если изображение цветное
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Выравнивание наклона
    if config.get("deskew_enabled", True):
        try:
            gray = deskew_image(gray)
        except Exception as e:
            logger.warning(f"Ошибка при выравнивании наклона: {e}")
    
    # Удаление шума
    if config.get("denoise_enabled", True):
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Бинаризация
    if config.get("binarize_enabled", True):
        # Используем общую функцию бинаризации
        binary = binarize_image(
            gray, 
            method=config.get("binarize_method", "sauvola"),
            window_size=config.get("window_size", 25)
        )
        return binary
    
    return gray