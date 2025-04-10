#!/bin/bash

# AudioBloomAI Build Script for Xcode 16 and macOS 15 SDK
# This script ensures proper build configuration for Swift 6 and macOS 15+

set -e

# Print header
echo "=== AudioBloomAI Build Script for Xcode 16 and macOS 15 SDK ==="

# Check for SDK
SDK_PATH="/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX15.4.sdk"
if [ ! -d "$SDK_PATH" ]; then
    echo "Error: macOS 15.4 SDK not found at $SDK_PATH"
    echo "Please ensure Xcode 16+ is installed with macOS 15.4 SDK"
    exit 1
fi

echo "Using SDK at: $SDK_PATH"

# Run Swift 6 modernization script
echo "Running Swift 6 modernization script..."
if [ -f "Scripts/modernize_swift.swift" ]; then
    swift Scripts/modernize_swift.swift
else
    echo "Modernization script not found. Skipping..."
fi

# Build with the proper SDK and Swift 6
echo "Building project with Swift 6 and macOS 15+ configuration..."
swift build \
    -c release \
    --sdk "$SDK_PATH" \
    -Xswiftc "-swift-version" -Xswiftc "6" \
    -Xswiftc "-enable-actor-data-race-checks" \
    -Xswiftc "-enable-bare-slash-regex" \
    -Xswiftc "-enable-upcoming-feature" -Xswiftc "BareSlashRegexLiterals"

# Run tests
echo "Running tests..."
swift test \
    --sdk "$SDK_PATH" \
    -Xswiftc "-swift-version" -Xswiftc "6" \
    -Xswiftc "-enable-actor-data-race-checks"

echo "Build completed successfully!"

