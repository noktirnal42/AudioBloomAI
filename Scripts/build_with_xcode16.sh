#!/bin/bash

# Script to build AudioBloomAI with Xcode 16 and macOS 15 SDK
# Ensures proper Swift 6 compatibility

set -e  # Exit on error

# Set the path to Xcode 16
XCODE_PATH="/Applications/Xcode.app/Contents/Developer"
SDK_PATH="$XCODE_PATH/Platforms/MacOSX.platform/Developer/SDKs/MacOSX15.4.sdk"

# Store current directory
CURRENT_DIR="$(pwd)"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== AudioBloomAI Build Script for Xcode 16 and macOS 15 SDK ===${NC}"
echo -e "Using SDK at: ${YELLOW}$SDK_PATH${NC}"

# Check if Xcode 16 is available
if [ ! -d "$XCODE_PATH" ]; then
    echo -e "${RED}Error: Xcode 16 not found at $XCODE_PATH${NC}"
    exit 1
fi

# Check if macOS 15 SDK is available
if [ ! -d "$SDK_PATH" ]; then
    echo -e "${RED}Error: macOS 15 SDK not found at $SDK_PATH${NC}"
    exit 1
fi

# Run modernization script first
echo -e "${GREEN}Running Swift 6 modernization script...${NC}"
swift Scripts/modernize_swift.swift

# Clean build directory
echo -e "${GREEN}Cleaning build directory...${NC}"
rm -rf .build

# Build with Swift 6
echo -e "${GREEN}Building with Swift 6 and macOS 15 SDK...${NC}"
xcrun --sdk macosx swiftc \
    -swift-version 6 \
    -target arm64-apple-macosx15.0 \
    -sdk "$SDK_PATH" \
    -O -whole-module-optimization \
    -package-description-version 6.0 \
    -emit-package \
    Package.swift

# Build the project
echo -e "${GREEN}Building project...${NC}"
swift build \
    -c release \
    -Xswiftc "-swift-version" -Xswiftc "6" \
    -Xswiftc "-target" -Xswiftc "arm64-apple-macosx15.0" \
    -Xswiftc "-sdk" -Xswiftc "$SDK_PATH" \
    -Xswiftc "-O" -Xswiftc "-whole-module-optimization"

# Run tests
echo -e "${GREEN}Running tests...${NC}"
swift test \
    -Xswiftc "-swift-version" -Xswiftc "6" \
    -Xswiftc "-target" -Xswiftc "arm64-apple-macosx15.0" \
    -Xswiftc "-sdk" -Xswiftc "$SDK_PATH"

echo -e "${GREEN}Build completed successfully!${NC}"

