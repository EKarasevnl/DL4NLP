#!/usr/bin/env python3
"""
Test script for language direction analysis functionality.
Demonstrates the new direction comparison functions without running the full plotting pipeline.
"""

import pandas as pd
import sys
import os

# Add the tables_plotting package to the path
sys.path.append('/Users/louis/Downloads/DL4NLP-tables_plotting/tables_plotting')

# Import the new function
from plotting import categorize_language_direction

def test_direction_categorization():
    """Test the language direction categorization function."""
    print("Testing language direction categorization...")
    
    # Create sample data
    sample_data = {
        'source_lang': ['English', 'English', 'German', 'French', 'Russian', 'English', 'German'],
        'target_lang': ['German', 'French', 'English', 'Russian', 'German', 'Russian', 'Polish'],
        'benchmark': ['FloRes'] * 7,
        'quant_level': ['Baseline'] * 7,
        'bleu': [25.4, 28.1, 32.7, 24.8, 18.9, 26.3, 22.1]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df[['source_lang', 'target_lang']].to_string())
    
    # Apply direction categorization
    df_with_directions = categorize_language_direction(df)
    print("\nWith direction types:")
    print(df_with_directions[['source_lang', 'target_lang', 'direction_type']].to_string())
    
    # Count by direction type
    direction_counts = df_with_directions['direction_type'].value_counts()
    print("\nDirection type counts:")
    print(direction_counts.to_string())
    
    return df_with_directions

def demonstrate_direction_analysis():
    """Demonstrate how the direction analysis would work."""
    print("\n" + "="*60)
    print("LANGUAGE DIRECTION ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Test categorization
    df = test_direction_categorization()
    
    print("\n" + "-"*40)
    print("ANALYSIS BREAKDOWN:")
    print("-"*40)
    
    print("\n1. English → Foreign:")
    print("   - Translation from English to non-English languages")
    print("   - Examples: English→German, English→French, English→Russian")
    print("   - Use case: Evaluating how well the model translates FROM English")
    
    print("\n2. Foreign → English:")
    print("   - Translation from non-English languages to English")
    print("   - Examples: German→English")
    print("   - Use case: Evaluating how well the model translates TO English")
    
    print("\n3. Foreign → Foreign:")
    print("   - Translation between two non-English languages")
    print("   - Examples: German→Polish, French→Russian")
    print("   - Use case: Evaluating multilingual capabilities without English")
    
    print("\n" + "-"*40)
    print("POTENTIAL INSIGHTS:")
    print("-"*40)
    
    print("\n• Performance differences by direction:")
    print("  - Does quantization affect English→Foreign more than Foreign→English?")
    print("  - Are Foreign→Foreign translations more degraded by pruning?")
    
    print("\n• Resource level interactions:")
    print("  - Do high-resource → low-resource translations suffer more?")
    print("  - How does the direction interact with language resource levels?")
    
    print("\n• Model bias analysis:")
    print("  - Is the model English-centric in its representations?")
    print("  - Do compression techniques preserve multilingual capabilities?")
    
    print("\n" + "-"*40)
    print("PLOTTING FUNCTIONS CREATED:")
    print("-"*40)
    
    print("\n1. prepare_direction_grouped_data():")
    print("   - Groups quantization data by direction type")
    print("   - Computes means, std, and counts for each direction")
    
    print("\n2. prepare_direction_grouped_pruning_data():")
    print("   - Same as above but for pruning experiments")
    
    print("\n3. plot_direction_comparison_chart():")
    print("   - Line charts showing trends across quant/sparsity levels")
    print("   - Separate lines for each direction type")
    print("   - Includes sample counts and error bars")
    
    print("\n4. create_direction_comparison_bar_chart():")
    print("   - Side-by-side bar charts for specific levels")
    print("   - Shows absolute values and percent change vs baseline")
    print("   - Useful for comparing specific conditions")
    
    print("\n5. make_all_direction_comparisons():")
    print("   - Orchestrates all quantization direction analysis")
    
    print("\n6. make_all_pruning_direction_comparisons():")
    print("   - Orchestrates all pruning direction analysis")

if __name__ == "__main__":
    demonstrate_direction_analysis()
    
    print("\n" + "="*60)
    print("To run the full analysis (when NumPy issues are resolved):")
    print("python3 plotting.py")
    print("\nThis will generate plots in:")
    print("./plots/direction-analysis/{metric}/")
    print("="*60)