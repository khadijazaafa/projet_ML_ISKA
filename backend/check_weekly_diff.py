"""
check_weekly_diff.py
====================
Affiche la comparaison entre :
- La dernière semaine connue (historique)
- La semaine actuelle (fetchée depuis USGS)
- La semaine prédite (prochaine)
+ Détecte si la semaine actuelle est une anomalie
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR    = Path(__file__).parent.parent
HISTORY_CSV = BASE_DIR / "data" / "weekly_history.csv"
FORECAST    = BASE_DIR / "data" / "output_forecast.json"

def run():
    # ── 1. CHARGER HISTORIQUE ─────────────────────────────────
    weekly = pd.read_csv(HISTORY_CSV, parse_dates=["date"])
    weekly = weekly.sort_values("date").reset_index(drop=True)

    # ── 2. CHARGER PRÉDICTION ─────────────────────────────────
    with open(FORECAST, "r") as f:
        forecast = json.load(f)

    pred_central = forecast["prediction"]["central"]
    pred_low     = forecast["prediction"]["low"]
    pred_high    = forecast["prediction"]["high"]
    next_week    = forecast["next_week_start"]

    # ── 3. EXTRAIRE LES DERNIÈRES SEMAINES ────────────────────
    last_8 = weekly.tail(8).reset_index(drop=True)

    print("\n" + "═"*65)
    print("   ANALYSE HEBDOMADAIRE — COMPARAISON & TENDANCE")
    print("═"*65)

    # ── 4. TABLEAU DES 8 DERNIÈRES SEMAINES ───────────────────
    print("\n   Historique des 8 dernières semaines :\n")
    print(f"  {'Semaine':<14} {'Séismes':>9} {'Variation':>10} {'Tendance':>10}")
    print(f"  {'─'*14} {'─'*9} {'─'*10} {'─'*10}")

    for i, row in last_8.iterrows():
        nb  = int(row["nb_seismes"])
        date_str = str(row["date"].date())

        if i == 0:
            variation = "—"
            tendance  = "—"
        else:
            prev = int(last_8.loc[i-1, "nb_seismes"])
            diff = nb - prev
            pct  = (diff / prev * 100) if prev != 0 else 0
            sign = "+" if diff >= 0 else ""
            variation = f"{sign}{diff} ({sign}{pct:.1f}%)"
            if diff > 50:
                tendance = "🔺 Forte hausse"
            elif diff > 10:
                tendance = "↗  Hausse"
            elif diff < -50:
                tendance = "🔻 Forte baisse"
            elif diff < -10:
                tendance = "↘  Baisse"
            else:
                tendance = "➡  Stable"

        # Marquer la semaine actuelle
        marker = " ◀ ACTUELLE" if i == len(last_8) - 1 else ""
        print(f"  {date_str:<14} {nb:>9,} {variation:>10}   {tendance}{marker}")

    # ── 5. FOCUS : DERNIÈRE vs AVANT-DERNIÈRE ─────────────────
    s_curr = last_8.iloc[-1]
    s_prev = last_8.iloc[-2]

    nb_curr = int(s_curr["nb_seismes"])
    nb_prev = int(s_prev["nb_seismes"])
    diff_abs = nb_curr - nb_prev
    diff_pct = (diff_abs / nb_prev * 100) if nb_prev != 0 else 0

    print(f"\n  {'─'*65}")
    print(f"\n  🔍 FOCUS : {str(s_prev['date'].date())} → {str(s_curr['date'].date())}")
    print(f"\n     Semaine précédente : {nb_prev:,} séismes")
    print(f"     Semaine actuelle   : {nb_curr:,} séismes")
    print(f"     Différence         : {'+' if diff_abs>=0 else ''}{diff_abs:,} ({'+' if diff_pct>=0 else ''}{diff_pct:.1f}%)")

    # Évaluation de l'anomalie
    mean_8 = last_8["nb_seismes"].mean()
    std_8  = last_8["nb_seismes"].std()
    z_score = (nb_curr - mean_8) / std_8 if std_8 > 0 else 0

    print(f"\n     Moyenne 8 semaines : {mean_8:.0f} séismes")
    print(f"     Écart-type 8 sem.  : {std_8:.0f}")
    print(f"     Z-score semaine    : {z_score:.2f}")

    if abs(z_score) > 2:
        print(f"\n     ⚠️  ANOMALIE DÉTECTÉE (z={z_score:.2f}) — Activité inhabituelle!")
        print(f"        Le modèle a vu cette anomalie → prédiction élevée normale")
    elif abs(z_score) > 1:
        print(f"\n     ⚡ Activité légèrement inhabituelle (z={z_score:.2f})")
    else:
        print(f"\n     ✅ Activité dans la norme (z={z_score:.2f})")

    # ── 6. PRÉDICTION SEMAINE PROCHAINE ───────────────────────
    print(f"\n  {'─'*65}")
    print(f"\n  🔮 PRÉDICTION — Semaine du {next_week}")
    print(f"\n     Actuelle → Prédite  : {nb_curr:,} → {pred_central:,} séismes")

    diff_pred = pred_central - nb_curr
    diff_pred_pct = (diff_pred / nb_curr * 100) if nb_curr != 0 else 0
    print(f"     Variation attendue  : {'+' if diff_pred>=0 else ''}{diff_pred:,} ({'+' if diff_pred_pct>=0 else ''}{diff_pred_pct:.1f}%)")
    print(f"     Intervalle [10-90%] : [{pred_low:,} – {pred_high:,}]")
    print(f"     Largeur intervalle  : {pred_high - pred_low:,} séismes")

    if pred_central > nb_curr * 1.2:
        print(f"\n     📈 Tendance : HAUSSE prévue (+20% min)")
    elif pred_central < nb_curr * 0.8:
        print(f"\n     📉 Tendance : BAISSE prévue (-20% min)")
    else:
        print(f"\n     ➡️  Tendance : STABLE (±20%)")

    # ── 7. MINI GRAPHIQUE ASCII ───────────────────────────────
    print(f"\n  {'─'*65}")
    print(f"\n  📈 Graphique ASCII (8 sem. + prédiction) :\n")

    all_vals = list(last_8["nb_seismes"]) + [pred_central]
    all_dates = [str(r["date"].date()) for _, r in last_8.iterrows()] + [f"→{next_week}"]
    max_val = max(all_vals)
    min_val = min(all_vals)
    bar_max = 40

    for date_s, val in zip(all_dates, all_vals):
        bar_len = int((val - min_val) / (max_val - min_val + 1) * bar_max)
        bar = "█" * bar_len
        # Couleur ASCII : marquer prédiction
        marker = " ◀ PRÉDIT" if "→" in date_s else ""
        label = date_s[:10]
        print(f"  {label}  {bar:<{bar_max}} {val:,}{marker}")

    print(f"\n  {'─'*65}")
    print(f"  Généré le : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    print("═"*65 + "\n")


if __name__ == "__main__":
    run()