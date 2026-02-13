#!/usr/bin/env python
"""Quick test of loss registry functionality."""

import sys

print("Testing Loss Registry Features...")

try:
    # Test imports
    from autotimm.losses import (
        get_loss_registry,
        list_available_losses,
        register_custom_loss,
    )
    print("✓ Import successful")
    
    # Test listing
    losses = list_available_losses()
    print(f"✓ Found {len(losses)} losses")
    
    # Test registry
    registry = get_loss_registry()
    dice = registry.get_loss("dice", num_classes=10)
    print(f"✓ Created loss: {type(dice)}")
    
    print("\n✅ ALL TESTS PASSED")
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
