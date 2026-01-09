#!/bin/bash
# Test runner script for VaultIQ

echo "========================================"
echo "Running VaultIQ Test Suite"
echo "========================================"
echo ""

# Run unit tests
echo "Running unit tests..."
pytest test/unit/ -v --tb=short

echo ""
echo "========================================"
echo "Test run complete!"
echo "========================================"
