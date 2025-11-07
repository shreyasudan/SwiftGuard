"""Test AgenticGuard on the harmful-prompts dataset."""

import sys
import os
import json
import time
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
from datasets import load_dataset
from collections import defaultdict
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.orchestration import run_agentic_guard
from src.utils.llm_factory import get_provider_info

# Load environment variables
load_dotenv()

def test_agenticguard_on_dataset(sample_size: int = 50, mode: str = "classic"):
    """Test AgenticGuard on the harmful prompts dataset.

    Args:
        sample_size: Number of prompts to test
        mode: "classic" or "precision" mode
    """
    print(f"\n{'='*80}")
    print(f"Testing AgenticGuard on harmful-prompts dataset")
    print(f"Mode: {mode}, Sample size: {sample_size}")
    print(f"{'='*80}\n")

    # Load dataset
    print("Loading dataset...")
    ds = load_dataset("onepaneai/harmful-prompts")
    dataset = ds['train']  # The dataset only has train split
    print(f"Total prompts in dataset: {len(dataset)}")

    # Get unique types
    types = set(dataset['type'])
    print(f"Prompt types: {types}")
    print()

    # Statistics
    stats = defaultdict(lambda: {'total': 0, 'detected': 0, 'errors': 0})
    failures = []
    processing_times = []

    # Take a sample
    if sample_size > len(dataset):
        sample_size = len(dataset)

    samples = dataset.select(range(sample_size))

    print(f"Testing {sample_size} prompts...")
    print("-" * 80)

    for idx, row in enumerate(samples):
        prompt_type = row['type']
        prompt_text = row['prompt']

        # Show progress
        if (idx + 1) % 10 == 0:
            print(f"Progress: {idx + 1}/{sample_size}")

        try:
            start_time = time.time()
            result = run_agentic_guard(prompt_text, mode=mode)
            elapsed = time.time() - start_time
            processing_times.append(elapsed)

            # Update statistics
            stats[prompt_type]['total'] += 1
            stats['overall']['total'] += 1

            # Check if threat was detected
            if result.get('threat_detected', False):
                stats[prompt_type]['detected'] += 1
                stats['overall']['detected'] += 1
            else:
                # This is a failure - we should detect harmful prompts
                failures.append({
                    'index': idx,
                    'type': prompt_type,
                    'prompt': prompt_text[:100] + '...' if len(prompt_text) > 100 else prompt_text,
                    'result': result
                })

        except Exception as e:
            print(f"Error on prompt {idx}: {str(e)}")
            stats[prompt_type]['errors'] += 1
            stats['overall']['errors'] += 1
            failures.append({
                'index': idx,
                'type': prompt_type,
                'prompt': prompt_text[:100] + '...' if len(prompt_text) > 100 else prompt_text,
                'error': str(e),
                'traceback': traceback.format_exc()
            })

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print("\nDetection Statistics by Type:")
    print("-" * 60)
    print(f"{'Type':<20} {'Total':<10} {'Detected':<10} {'Detection %':<15} {'Errors':<10}")
    print("-" * 60)

    for prompt_type in sorted(stats.keys()):
        if prompt_type == 'overall':
            continue
        data = stats[prompt_type]
        if data['total'] > 0:
            detection_rate = (data['detected'] / data['total']) * 100
            print(f"{prompt_type:<20} {data['total']:<10} {data['detected']:<10} {detection_rate:<15.1f} {data['errors']:<10}")

    print("-" * 60)
    overall = stats['overall']
    if overall['total'] > 0:
        overall_detection = (overall['detected'] / overall['total']) * 100
        print(f"{'OVERALL':<20} {overall['total']:<10} {overall['detected']:<10} {overall_detection:<15.1f} {overall['errors']:<10}")

    # Processing time statistics
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)

        print(f"\nProcessing Time Statistics:")
        print(f"  Average: {avg_time:.3f}s")
        print(f"  Min: {min_time:.3f}s")
        print(f"  Max: {max_time:.3f}s")

    # Show failures
    print(f"\nFailure Analysis:")
    print(f"Total failures (false negatives): {len(failures)}")

    if failures:
        print("\nFirst 10 failures:")
        print("-" * 80)
        for failure in failures[:10]:
            print(f"\nIndex: {failure['index']}")
            print(f"Type: {failure['type']}")
            print(f"Prompt: {failure['prompt']}")
            if 'error' in failure:
                print(f"Error: {failure['error']}")
            else:
                result = failure['result']
                print(f"Threat Level: {result.get('threat_level', 'N/A')}")
                print(f"Confidence: {result.get('confidence_score', 0):.2%}")
                print(f"Action: {result.get('recommended_action', 'N/A')}")

    # Analyze failure patterns
    print("\n" + "="*80)
    print("FAILURE PATTERN ANALYSIS")
    print("="*80)

    # Group failures by type
    failure_by_type = defaultdict(list)
    for failure in failures:
        if 'error' not in failure:  # Only count detection failures, not errors
            failure_by_type[failure['type']].append(failure)

    print("\nFailures by prompt type:")
    for prompt_type, type_failures in failure_by_type.items():
        print(f"  {prompt_type}: {len(type_failures)} failures")

    # Analyze common patterns in failed prompts
    if failure_by_type:
        print("\nCommon patterns in failed detections:")

        # Sample failed prompts for each type
        for prompt_type, type_failures in failure_by_type.items():
            if type_failures:
                print(f"\n{prompt_type} failures (sample):")
                for failure in type_failures[:3]:  # Show first 3 examples
                    print(f"  - {failure['prompt'][:100]}...")

    return stats, failures

def main():
    """Main function to run tests."""
    # Check provider configuration
    provider_info = get_provider_info()
    print("\nProvider Configuration:")
    print(f"  LLM Provider: {provider_info['llm_provider']}")
    print(f"  Embedding Provider: {provider_info['embedding_provider']}")

    # Check API keys
    api_status = provider_info['api_key_set']
    provider = provider_info['llm_provider']

    if provider == "openai" and not api_status['openai']:
        print("Error: OPENAI_API_KEY not found")
        return
    elif provider == "together" and not api_status['together']:
        print("Error: TOGETHER_API_KEY not found")
        return
    elif provider == "anthropic" and not api_status['anthropic']:
        print("Error: ANTHROPIC_API_KEY not found")
        return

    # Run tests
    try:
        # Test in classic mode (faster)
        print("\n" + "="*80)
        print("TESTING CLASSIC MODE")
        print("="*80)
        classic_stats, classic_failures = test_agenticguard_on_dataset(
            sample_size=30,  # Smaller sample for testing
            mode="classic"
        )

        # Optional: Test precision mode (slower but more accurate)
        # print("\n" + "="*80)
        # print("TESTING PRECISION MODE")
        # print("="*80)
        # precision_stats, precision_failures = test_agenticguard_on_dataset(
        #     sample_size=20,
        #     mode="precision"
        # )

    except Exception as e:
        print(f"Test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()