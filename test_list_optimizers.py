#!/usr/bin/env python
"""Test script for list_optimizers function."""

from autotimm.utils import list_optimizers

def test_list_optimizers():
    """Test the list_optimizers function."""
    print("Testing list_optimizers()...\n")
    
    # Test with timm
    opts = list_optimizers(include_timm=True)
    
    print("=" * 60)
    print("PyTorch Optimizers:")
    print("=" * 60)
    for opt in opts['torch']:
        print(f"  - {opt}")
    print(f"\nTotal PyTorch optimizers: {len(opts['torch'])}")
    
    print("\n" + "=" * 60)
    print("timm Optimizers:")
    print("=" * 60)
    timm_opts = opts.get('timm', [])
    if timm_opts:
        for opt in timm_opts:
            print(f"  - {opt}")
        print(f"\nTotal timm optimizers: {len(timm_opts)}")
    else:
        print("  (timm not available or no optimizers found)")
    
    # Test without timm
    print("\n" + "=" * 60)
    print("Without timm (include_timm=False):")
    print("=" * 60)
    opts_no_timm = list_optimizers(include_timm=False)
    print(f"Has 'timm' key: {'timm' in opts_no_timm}")
    print(f"PyTorch optimizers: {len(opts_no_timm['torch'])}")
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_list_optimizers()
