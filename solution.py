"""
КОНТРОЛЬНАЯ РАБОТА №2 – ВАРИАНТ 16 (thvarprj)
X=X₇ (премии) → Y=X₁ (производительность труда)
Читает данные из enterprises_data.xlsx
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import os

# 📖 ЧТЕНИЕ ДАННЫХ ИЗ EXCEL + НАГЛЯДНЫЙ ВЫВОД
print("📖 thvarprj: Загрузка enterprises_data.xlsx...")
df = pd.read_excel('enterprises_data.xlsx')
X = df['X7_Премии'].values  # Столбец C
Y = df['X1_Производительность'].values  # Столбец B
n = len(X)

print(f"✅ thvarprj: {n} предприятий загружено")
print(f"   X₇ (премии): {X.min():.2f}...{X.max():.2f}")
print(f"   X₁: {Y.min():.2f}...{Y.max():.2f}")

# 🔥 НОВЫЙ БЛОК: ВЫВОД ИСХОДНЫХ ДАННЫХ
print("\n📋 ИСХОДНЫЕ ДАННЫЕ (первые 10 + последние 5 строк):")
print("="*50)
print(df[['Предприятие', 'X7_Премии', 'X1_Производительность']].head(10).to_string(index=False))
print("         ... (показаны 10 из 50) ...")
print(df[['Предприятие', 'X7_Премии', 'X1_Производительность']].tail(5).to_string(index=False))

def task1_descriptive_stats(X, Y):
    """📊 Задание 1: Описательные статистики"""
    stats_X = {
        'n': len(X), 'Средняя': np.mean(X), 'СКО': np.std(X, ddof=1),
        'Медиана': np.median(X), 'Мода': float(pd.Series(X).mode().iloc[0]),
        'Период': f"({X.min():.2f}; {X.max():.2f})", 'Размах': X.max() - X.min()
    }
    stats_Y = stats_X.copy()
    stats_Y.update({
        'Средняя': np.mean(Y), 'СКО': np.std(Y, ddof=1),
        'Медиана': np.median(Y), 'Мода': float(pd.Series(Y).mode().iloc[0]),
        'Период': f"({Y.min():.2f}; {Y.max():.2f})", 'Размах': Y.max() - Y.min()
    })
    
    df_stats = pd.DataFrame([stats_X, stats_Y], index=['X₇', 'X₁'])
    return df_stats

def task2_regression(X, Y):
    """📈 Задание 2: Регрессия + проверки преподавателя"""
    slope, intercept, r, p, stderr = stats.linregress(X, Y)
    R2 = r**2
    med_X = np.median(X)
    Y_pred = intercept + slope * med_X
    m = len(X) - 1
    check1 = (len(X)*np.mean(Y) - Y[0]) / m  # (n*ȳ-Y₁)/m
    check2 = 1 + 3.322 * np.log(m)           # 1+3.322*ln(m)
    
    return {
        'Уравнение': f'Y = {intercept:.4f} + {slope:.4f}X',
        'r': round(r, 4), 'R²': round(R2, 4), 'p-value': f'{p:.4f}',
        'Ŷ(медиана_X={med_X:.2f})': round(Y_pred, 4),
        'Проверка_преподавателя': f'({check1:.3f}); {check2:.2f}'
    }

def task3_ci(X):
    """🎯 Задание 3: Доверительный интервал μ_X₇"""
    n, mean, std = len(X), np.mean(X), np.std(X, ddof=1)
    t_crit = stats.t.ppf(0.975, n-1)
    margin = t_crit * std / np.sqrt(n)
    return {'μ̂': round(mean, 4), 'ДИ_95%': [round(mean-margin, 4), round(mean+margin, 4)]}

# 🚀 ОСНОВНОЙ ЗАПУСК thvarprj
os.makedirs('results', exist_ok=True)

print("\n" + "="*60)
print("🎓 thvarprj – КОНТРОЛЬНАЯ РАБОТА №2, ВАРИАНТ 16")
print("="*60)

print("\n📊 ЗАДАНИЕ 1: ОПИСАТЕЛЬНЫЕ СТАТИСТИКИ")
print("-" * 40)
stats_df = task1_descriptive_stats(X, Y)
print(stats_df.round(4))

print("\n📈 ЗАДАНИЕ 2: ЛИНЕЙНАЯ РЕГРЕССИЯ")
print("-" * 40)
reg = task2_regression(X, Y)
for k, v in reg.items():
    print(f"{k:25}: {v}")

print("\n🎯 ЗАДАНИЕ 3: ДОВЕРИТЕЛЬНЫЙ ИНТЕРВАЛ")
print("-" * 40)
ci = task3_ci(X)
print(f"Точечная: μ_X₇ = {ci['μ̂']}")
print(f"ДИ 95%: [{ci['ДИ_95%'][0]}; {ci['ДИ_95%'][1]}]")

# 📊 ГРАФИКИ thvarprj
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

slope, intercept = stats.linregress(X, Y)[:2]
axes[0].scatter(X, Y, alpha=0.7, s=60, color='steelblue')
axes[0].plot(X, intercept + slope*X, 'r-', linewidth=3, 
             label=f'{reg["Уравнение"]}\nR²={reg["R²"]}')
axes[0].set_xlabel('X₇: Премии (тыс.руб./чел.)', fontsize=11)
axes[0].set_ylabel('X₁: Производительность труда', fontsize=11)
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].set_title('Регрессия Y на X (Задание 2)', fontsize=12)  # Без эмодзи

axes[1].hist(X, bins=15, alpha=0.7, edgecolor='black', color='lightcoral')
axes[1].axvline(np.mean(X), color='r', lw=3, ls='--', 
                label=f'μ̂={ci["μ̂"]}')
axes[1].axvline(ci['ДИ_95%'][0], color='orange', lw=2, ls=':', 
                label=f'ДИ95%: [{ci["ДИ_95%"][0]}; {ci["ДИ_95%"][1]}]')
axes[1].set_xlabel('X₇: Премии', fontsize=11)
axes[1].set_ylabel('Частота', fontsize=11)
axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].set_title('Доверительный интервал (Задание 3)', fontsize=12)  # Без эмодзи

plt.tight_layout()
plt.savefig('results/thvarprj_full_report.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ thvarprj: Сохранено results/thvarprj_full_report.png")
print("🎓 КОНТРОЛЬНАЯ РАБОТА ВЫПОЛНЕНА!")
