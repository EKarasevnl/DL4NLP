#!/usr/bin/env python3
"""
Test script to verify the additional metrics (COMET-22, kiwi-23, XL) work correctly.
"""

import sys
import os

# Add current directory to path to import the evaluation modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        from comet import download_model, load_from_checkpoint
        print("✓ COMET available")
        comet_available = True
    except ImportError as e:
        print(f"✗ COMET not available: {e}")
        comet_available = False
    
    try:
        import evaluate
        print("✓ evaluate library available")
        evaluate_available = True
    except ImportError as e:
        print(f"✗ evaluate library not available: {e}")
        evaluate_available = False
    
    return comet_available, evaluate_available

def test_metrics():
    """Test the metric computation functions with dummy data."""
    print("\nTesting metric computation functions...")
    
    # Import the metric computation function
    try:
        from flores_eval import _compute_advanced_metrics
        print("✓ Successfully imported _compute_advanced_metrics from flores_eval")
    except ImportError as e:
        print(f"✗ Failed to import from flores_eval: {e}")
        return
    
    # Test with dummy data
    sources = ["Hello world.", "This is a test."]
    hypotheses = ["Hola mundo.", "Esta es una prueba."]
    references = ["Hola mundo.", "Esto es una prueba."]
    
    try:
        metrics = _compute_advanced_metrics(sources, hypotheses, references)
        print(f"✓ Metric computation completed: {metrics}")
        
        # Check that all expected keys are present
        expected_keys = ["comet22", "kiwi23", "xl"]
        for key in expected_keys:
            if key in metrics:
                value = metrics[key]
                if value is not None:
                    print(f"✓ {key}: {value}")
                else:
                    print(f"⚠ {key}: None (metric not available or failed)")
            else:
                print(f"✗ Missing metric: {key}")
                
    except Exception as e:
        print(f"✗ Metric computation failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("=" * 50)
    print("Testing Additional Metrics Implementation")
    print("=" * 50)
    
    comet_available, evaluate_available = test_imports()
    
    if comet_available or evaluate_available:
        test_metrics()
    else:
        print("\nSkipping metric tests - no required packages available")
        print("To install required packages, run:")
        print("pip install -r requirements_additional_metrics.txt")
    
    print("\n" + "=" * 50)
    print("Test completed")

if __name__ == "__main__":
    main()