from __future__ import annotations
import argparse
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import signal

# Константы
PLOT_DPI = 150
PLOT_SIZE = (16, 10)

# Данные из полученного графика
data = {
    "2022-06-01": 4131696, "2022-07-01": 3621281, "2022-08-01": 4326653, "2022-09-01": 22790347,
    "2022-10-01": 13165326, "2022-11-01": 10442499, "2022-12-01": 39679196, "2023-01-01": 7587545,
    "2023-02-01": 25812367, "2023-03-01": 5261109, "2023-04-01": 30812610, "2023-05-01": 11285886,
    "2023-06-01": 21821692, "2023-07-01": 13902859, "2023-08-01": 5213703, "2023-09-01": 47615030,
    "2023-10-01": 124697679, "2023-11-01": 49212537, "2023-12-01": 22717109, "2024-01-01": 15265177,
    "2024-02-01": 17657155, "2024-03-01": 16600631, "2024-04-01": 17607736, "2024-05-01": 17192626,
    "2024-06-01": 15147653, "2024-07-01": 36026482, "2024-08-01": 17848075, "2024-09-01": 53067301,
    "2024-10-01": 14169602, "2024-11-01": 28968626, "2024-12-01": 47490555, "2025-01-01": 31875031,
    "2025-02-01": 36919810, "2025-03-01": 23661236, "2025-04-01": 38672023, "2025-05-01": 12049933,
    "2025-06-01": 15715526, "2025-07-01": 56257920, "2025-08-01": 36175074, "2025-09-01": 19745073,
    "2025-10-01": 19661037, "2025-11-01": 2200173
}


def setup_russian_labels():
    import matplotlib
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False


def prepare_data() -> pd.DataFrame:
    """Подготовка данных из предоставленного словаря"""
    dates = []
    downloads = []

    for date_str, download_count in data.items():
        dates.append(pd.to_datetime(date_str))
        downloads.append(download_count)

    df = pd.DataFrame({
        'period': dates,
        'downloads': downloads
    })
    df = df.set_index('period').sort_index()
    return df


def rogers_diffusion_model(historical_mean: float, target_5_years: float = 60000000) -> dict:
    """
    Модель диффузии инноваций Роджерса с настройкой под целевую цифру 60 млн
    """
    # Параметры фаз по Роджерсу с адаптацией под ИИ-рынок
    diffusion_phases = {
        'innovators': {
            'duration': 12,  # 1 год - инноваторы
            'percentage': 0.025,  # 2.5% от общего потенциала
            'target_multiplier': 1.3,  # Рост на 30% от текущего уровня
            'volatility': 0.4,
            'target_value': 35000000  # ~35M
        },
        'early_adopters': {
            'duration': 18,  # 1.5 года - ранние последователи
            'percentage': 0.135,  # 13.5% от общего потенциала
            'target_multiplier': 1.57,  # Рост до ~55M
            'volatility': 0.3,
            'target_value': 55000000  # ~55M
        },
        'early_majority': {
            'duration': 24,  # 2 года - раннее большинство
            'percentage': 0.34,  # 34% от общего потенциала
            'target_multiplier': 1.18,  # Рост до ~65M
            'volatility': 0.25,
            'target_value': 65000000  # ~65M
        },
        'late_majority': {
            'duration': 6,  # 6 месяцев - позднее большинство (только до 2030-08)
            'percentage': 0.34,  # 34% от общего потенциала
            'target_multiplier': 0.92,  # Стабилизация ~60M
            'volatility': 0.2,
            'target_value': 60000000  # ~60M
        }
    }

    print(f"=== МОДЕЛЬ РОДЖЕРСА ===")
    print(f"Целевое значение через 5 лет: {target_5_years / 1e6:.1f}M")

    for phase_name, params in diffusion_phases.items():
        print(f"{phase_name}: {params['target_value'] / 1e6:.1f}M")

    return diffusion_phases


def calculate_smoothed_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Вычисляет сглаженный тренд"""
    trend_df = df.copy()
    trend_df['ma_6m'] = trend_df['downloads'].rolling(window=6, min_periods=1, center=True).mean()

    if len(trend_df) >= 7:
        window_length = min(7, len(trend_df) - 2)
        if window_length >= 3:
            trend_df['savgol'] = signal.savgol_filter(
                trend_df['downloads'], window_length=window_length, polyorder=2
            )
        else:
            trend_df['savgol'] = trend_df['downloads']
    else:
        trend_df['savgol'] = trend_df['downloads']

    trend_df['smoothed_trend'] = trend_df[['ma_6m', 'savgol']].mean(axis=1)
    trend_df['smoothed_trend'] = trend_df['smoothed_trend'].fillna(trend_df['downloads'])

    return trend_df[['downloads', 'smoothed_trend']]


def create_rogers_diffusion_forecast(df: pd.DataFrame, target_5_years: float = 60000000) -> pd.DataFrame:
    """ Создание прогноза на основе модели диффузии инноваций Роджерса """
    # Анализ исторических данных
    historical_stats = df['downloads'].describe()
    historical_mean = historical_stats['mean']
    last_value = df['downloads'].iloc[-1]

    print("=== ИСТОРИЧЕСКИЙ АНАЛИЗ ===")
    print(f"Среднее значение (2022-2025): {historical_mean / 1e6:.1f}M")
    print(f"Последнее значение (2025-11): {last_value / 1e6:.1f}M")

    # Получаем параметры модели Роджерса
    rogers_phases = rogers_diffusion_model(historical_mean, target_5_years)

    # Создаем прогноз
    last_date = df.index.max()
    forecast_data = []
    current_value = last_value

    total_months = sum(phase['duration'] for phase in rogers_phases.values())

    # Для воспроизводимости
    np.random.seed(42)

    for month in range(1, total_months + 1):
        # Определяем текущую фазу
        current_month = 0
        current_phase = None
        current_phase_name = None

        for phase_name, phase_params in rogers_phases.items():
            if month <= current_month + phase_params['duration']:
                current_phase = phase_params
                current_phase_name = phase_name
                phase_progress = (month - current_month) / phase_params['duration']
                break
            current_month += phase_params['duration']

        # Целевое значение для фазы
        target_value = current_phase['target_value']

        # Плавный переход к целевому значению
        transition_strength = 0.6 + 0.3 * phase_progress  # Усиливается к концу фазы
        base_value = current_value * (1 - transition_strength) + target_value * transition_strength

        # Сезонность
        current_date = last_date + pd.DateOffset(months=month)
        season_month = current_date.month - 1
        seasonal_factor = 1 + 0.25 * np.sin(2 * np.pi * (season_month - 2) / 12)

        # Тренд на основе прогресса фазы
        if current_phase_name in ['innovators', 'early_adopters']:
            trend_power = 0.1  # Сильный тренд в начале
        else:
            trend_power = 0.05  # Умеренный тренд позже

        trend_factor = 1 + trend_power * np.sin(phase_progress * np.pi / 2)

        # Случайная компонента
        random_factor = 1 + current_phase['volatility'] * np.random.normal(0, 0.4)

        # Расчет прогнозного значения
        forecast_value = base_value * trend_factor * seasonal_factor * random_factor

        # Сглаживание
        smoothing = 0.8
        forecast_value = current_value * smoothing + forecast_value * (1 - smoothing)

        # Ограничения
        forecast_value = max(forecast_value, 5000000)
        forecast_value = min(forecast_value, 100000000)

        current_value = forecast_value

        forecast_data.append({
            'period': current_date,
            'downloads': forecast_value,
            'is_forecast': True,
            'phase': current_phase_name
        })

    # Создаем DataFrame с прогнозом
    forecast_df = pd.DataFrame(forecast_data).set_index('period')

    # Добавляем исторические данные
    historical_df = df.copy()
    historical_df['is_forecast'] = False
    historical_df['phase'] = 'historical'

    # Объединяем
    full_df = pd.concat([historical_df, forecast_df])

    return full_df, rogers_phases


def plot_rogers_diffusion_forecast(df: pd.DataFrame, rogers_phases: dict, out_path: str):
    """Построение графика с моделью диффузии Роджерса"""
    plt.figure(figsize=PLOT_SIZE, dpi=PLOT_DPI)
    ax = plt.gca()

    # Разделяем данные
    historical_data = df[~df['is_forecast']]
    forecast_data = df[df['is_forecast']]

    # Вычисляем сглаженный тренд
    historical_with_trend = calculate_smoothed_trend(historical_data)

    # Цвета для фаз
    phase_colors = {
        'historical': '#2E86AB',  # Синий для исторических данных
        'smoothed_trend': 'darkred',  # Темно-красный для тренда
        'innovators': '#FF6B35',  # Оранжевый - инноваторы
        'early_adopters': '#4ECDC4',  # Бирюзовый - ранние последователи
        'early_majority': '#45B7D1',  # Синий - раннее большинство
        'late_majority': '#96CEB4'  # Зеленый - позднее большинство
    }

    # Создаем единую легенду
    legend_elements = []

    # Исторические данные
    line_historical, = ax.plot(historical_data.index, historical_data['downloads'],
                               color=phase_colors['historical'], linewidth=2.5,
                               marker='o', markersize=3, alpha=0.7)
    legend_elements.append(line_historical)

    # Сглаженный тренд
    line_trend, = ax.plot(historical_with_trend.index, historical_with_trend['smoothed_trend'],
                          color=phase_colors['smoothed_trend'], linewidth=3, linestyle='--', alpha=0.8)
    legend_elements.append(line_trend)

    # Прогноз по фазам Роджерса
    if not forecast_data.empty:
        for phase in ['innovators', 'early_adopters', 'early_majority', 'late_majority']:
            if phase in forecast_data['phase'].unique():
                phase_data = forecast_data[forecast_data['phase'] == phase]
                line, = ax.plot(phase_data.index, phase_data['downloads'],
                                color=phase_colors[phase], linewidth=2.5, linestyle='--', alpha=0.9)
                legend_elements.append(line)

    # Разделительная линия
    split_date = historical_data.index.max()
    ax.axvline(x=split_date, color='green', linestyle=':', alpha=0.7, linewidth=2)
    ax.text(split_date, ax.get_ylim()[1] * 0.8, 'Начало прогноза',
            rotation=90, va='top', ha='right', color='green', fontweight='bold')

    # Настройка графика
    ax.set_title("Прогноз скачиваний моделей на основе Теории диффузии инноваций Роджерса",
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Период", fontsize=12)
    ax.set_ylabel("Количество скачиваний", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Форматирование осей
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x / 1e6:.0f}M'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha='right')

    # Единая легенда
    legend_labels = [
        'Фактические данные',
        '2022-2025 Сглаженные данные',
        '2025-2026 Инноваторы (2.5%) - рост до ~35M',
        '2026-2027 Ранние последователи (13.5%) - рост до ~55M',
        '2027-2029 Раннее большинство (34%) - рост до ~65M',
        '2029-2030 Позднее большинство (34%) - стабилизация ~60M'
    ]

    ax.legend(legend_elements, legend_labels, loc='upper left', fontsize=9)

    # Информация о модели


    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=PLOT_DPI)
    plt.close()


def main(out_png: str, out_csv: Optional[str] = None):
    setup_russian_labels()

    df = prepare_data()

    if df.empty:
        print("Нет данных.")
        return

    print("Построение прогноза по модели Роджерса...")
    target_5_years = 60000000  # Целевые 60 млн
    forecast_df, rogers_phases = create_rogers_diffusion_forecast(df, target_5_years)

    if out_csv:
        forecast_df.to_csv(out_csv, index_label="period")
        print(f"Данные сохранены в: {out_csv}")

    plot_rogers_diffusion_forecast(forecast_df, rogers_phases, out_png)
    print(f"График сохранен в: {out_png}")

    # Детальный анализ результатов
    historical_avg = df['downloads'].mean()
    forecast_data = forecast_df[forecast_df['is_forecast']]
    forecast_avg = forecast_data['downloads'].mean()
    final_value = forecast_data['downloads'].iloc[-1]

    print(f"\n=== РЕЗУЛЬТАТЫ ПРОГНОЗА ===")
    print(f"Историческое среднее (2022-2025): {historical_avg / 1e6:.1f}M")
    print(f"Прогнозное среднее (2026-2030): {forecast_avg / 1e6:.1f}M")
    print(f"Конечное значение (2030-08): {final_value / 1e6:.1f}M")

    total_growth = ((forecast_avg - historical_avg) / historical_avg * 100)
    print(f"Средний рост: {total_growth:+.1f}%")

    # Анализ по фазам
    print(f"\n=== АНАЛИЗ ПО ФАЗАМ РОДЖЕРСА ===")
    for phase_name in ['innovators', 'early_adopters', 'early_majority', 'late_majority']:
        if phase_name in forecast_data['phase'].unique():
            phase_data = forecast_data[forecast_data['phase'] == phase_name]
            phase_avg = phase_data['downloads'].mean()
            phase_target = rogers_phases[phase_name]['target_value']
            print(f"{phase_name}: {phase_avg / 1e6:.1f}M (цель: {phase_target / 1e6:.1f}M)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Прогноз по модели диффузии инноваций Роджерса")
    parser.add_argument("--out", required=True, help="Путь к PNG")
    parser.add_argument("--out-csv", default=None, help="Путь к CSV (опционально)")
    args = parser.parse_args()
    main(args.out, args.out_csv)