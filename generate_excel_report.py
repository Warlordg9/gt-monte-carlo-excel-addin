import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from engine.simulator import MonteCarloEngine

def create_report(config_path: str, output_path: str = None):
    # Загрузка конфига
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Запуск симуляции
    engine = MonteCarloEngine(config)
    results = engine.run()
    stats = engine.get_statistics()
    sensitivity = engine.get_sensitivity()
    input_samples = engine.input_samples_df
    output_samples = results

    # Путь для сохранения
    if output_path is None:
        output_path = Path(config_path).stem + "_report.xlsx"

    # Создаём Excel-файл через pandas ExcelWriter
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        # Лист 1: Статистики
        stats_df = pd.DataFrame(stats).T
        stats_df.to_excel(writer, sheet_name='Statistics')

        # Лист 2: Чувствительность (для первого выхода)
        first_out = list(sensitivity.keys())[0]
        sens_df = sensitivity[first_out]
        sens_df.to_excel(writer, sheet_name=f'Sensitivity_{first_out}')

        # Лист 3: Пример первых 1000 итераций входов
        input_samples.head(1000).to_excel(writer, sheet_name='Input_samples_head')

        # Лист 4: Пример первых 1000 итераций выходов
        output_samples.head(1000).to_excel(writer, sheet_name='Output_samples_head')

        # Лист 5: Гистограмма для GT Gross Power (пример)
        fig, ax = plt.subplots()
        ax.hist(output_samples['GT_Gross_Power_MW'], bins=50, alpha=0.7)
        ax.set_title('Distribution of GT Gross Power')
        ax.set_xlabel('MW')
        # Сохраняем рисунок во временный буфер и вставляем
        fig.savefig('temp_hist.png')
        plt.close(fig)
        # Вставка картинки через xlsxwriter требует отдельного кода, проще сохранить как PNG и приложить к папке
        # Для простоты просто сохраним рисунок рядом
        plt.figure()
        output_samples['GT_Gross_Power_MW'].hist(bins=50)
        plt.savefig(Path(output_path).stem + '_hist.png')

    print(f"Report saved to {output_path}")
    print(f"Histogram saved as {Path(output_path).stem}_hist.png")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        create_report(sys.argv[1])
    else:
        create_report("config/gt36_s5_config.json")
