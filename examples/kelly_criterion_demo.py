"""
Kelly Criterion Position Sizing Demonstration
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.trading.position_sizing import (
    compare_position_sizing_methods,
    print_comparison_table
)


def main():
    print("=" * 80)
    print("Kelly Criterion Position Sizing Demo")
    print("=" * 80)

    # Example scenario
    capital = 100000.0
    price = 150.0
    win_rate = 0.55  # 55% win rate
    avg_win = 0.05   # 5% average win
    avg_loss = 0.03  # 3% average loss
    confidence = 0.75  # 75% confidence from model
    volatility = 0.30  # 30% annualized volatility

    print(f"\nScenario:")
    print(f"  Capital: ${capital:,.2f}")
    print(f"  Stock Price: ${price:.2f}")
    print(f"  Win Rate: {win_rate*100:.0f}%")
    print(f"  Avg Win: {avg_win*100:.1f}%")
    print(f"  Avg Loss: {avg_loss*100:.1f}%")
    print(f"  Model Confidence: {confidence*100:.0f}%")
    print(f"  Stock Volatility: {volatility*100:.0f}%")

    # Compare methods
    results = compare_position_sizing_methods(
        current_capital=capital,
        current_price=price,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        confidence=confidence,
        stock_volatility=volatility
    )

    # Print results
    print_comparison_table(results)

    # Recommendation
    print("\n" + "=" * 80)
    print("Analysis & Recommendations")
    print("=" * 80)
    print("\n1. Fixed Risk (2%):")
    print("   - Most conservative approach")
    print("   - Easy to implement and understand")
    print("   - Doesn't adapt to edge or confidence")
    print("\n2. Full Kelly:")
    print("   - Maximizes growth rate")
    print("   - Very aggressive - can lead to large drawdowns")
    print("   - NOT recommended (too risky)")
    print("\n3. Half Kelly (RECOMMENDED):")
    print("   - Good balance of growth and risk")
    print("   - Reduces variance compared to Full Kelly")
    print("   - Industry standard for Kelly-based sizing")
    print("\n4. Half Kelly + Confidence (BEST):")
    print("   - Adapts to model uncertainty")
    print("   - Reduces position when model is uncertain")
    print("   - Recommended for ML-based trading")
    print("\n5. Quarter Kelly:")
    print("   - Very conservative Kelly variant")
    print("   - Good for risk-averse traders")
    print("   - Lower growth but much lower variance")


if __name__ == "__main__":
    main()
