import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from engine.simulator import MonteCarloEngine

def plot_tornado(sensitivity_df, output_name, save_path):
    """Строит tornado plot для первых 10 входов по абсолютной корреляции."""
    top10 = sensitivity_df.head(10).copy()
    top10['abs_corr'] = top10['spearman_corr'].abs()
    top10 = top10.sort_values('abs_corr')
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top10['input'], top10['spearman_corr'], color='steelblue')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Spearman rank correlation')
    ax.set_title(f'Sensitivity tornado: {output_name}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_simulation.py <config.json>")
        sys.exit(1)
    
    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"Config file not found: {config_path.absolute()}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Running Monte Carlo with {config['n_iterations']} iterations...")
    engine = MonteCarloEngine(config)
    results = engine.run()
    stats = engine.get_statistics()
    sensitivity = engine.get_sensitivity()
    
    # Сохраняем Excel
    output_path = config_path.stem + "_report.xlsx"
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        stats_df = pd.DataFrame(stats).T
        stats_df.to_excel(writer, sheet_name='Statistics')
        first_out = list(sensitivity.keys())[0]
        sens_df = sensitivity[first_out]
        sens_df.to_excel(writer, sheet_name=f'Sensitivity_{first_out}')
        results.head(1000).to_excel(writer, sheet_name='Outputs_head')
    
    # Гистограмма
    plt.figure(figsize=(10,6))
    results['GT_Gross_Power_MW'].hist(bins=50, alpha=0.7)
    plt.title('GT Gross Power Distribution')
    plt.xlabel('MW')
    plt.savefig(config_path.stem + "_hist.png", dpi=150)
    plt.close()
    
    # Tornado plot для GT Gross Power
    if 'GT_Gross_Power_MW' in sensitivity:
        plot_tornado(sensitivity['GT_Gross_Power_MW'], 'GT Gross Power', config_path.stem + "_tornado.png")
    
    print(f"Done! Results saved to {output_path}, histogram and tornado PNGs.")

if __name__ == "__main__":
    main()
