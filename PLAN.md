# AudioBloomAI Project Plan

## Project Overview
AudioBloomAI is a next-generation audio visualizer leveraging Apple Silicon features, focusing on Metal, Core ML, and the Neural Engine integration.

## Architecture
The project is structured into several key modules:

1. AudioBloomApp (Main Executable)
   - SwiftUI-based user interface
   - Main application coordination

2. AudioBloomCore
   - Core shared functionality
   - Common utilities and interfaces

3. AudioProcessor
   - Audio signal processing
   - Real-time audio analysis

4. Visualizer
   - Metal-based visualization engine
   - Shader management and rendering
   - Located in Sources/Visualizer
   - Includes Metal shader resources

5. MLEngine
   - Neural Engine integration
   - Machine learning model management
   - Integration with Gemini API

## Development Requirements
- macOS 15.0+ deployment target
- Xcode 16+ with Metal support
- SDK Path: /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX15.4.sdk
- Swift Package Manager for dependency management

## External Dependencies
- swift-algorithms (v1.0.0+)
- swift-numerics (v1.0.0+)
- Google Gemini API (for ML features)

## Project Structure
```
Sources/
  ├── AudioBloomApp/
  ├── AudioBloomCore/
  ├── AudioProcessor/
  ├── Visualizer/
  │   └── Resources/
  │       └── Shaders/
  └── MLEngine/

Tests/
  ├── AudioBloomTests/
  ├── AudioProcessorTests/
  ├── VisualizerTests/
  └── MLEngineTests/
```

## Development Workflow
1. All development work branches from 'develop'
2. Feature branches format: feature/description
3. Pull requests merge into develop
4. Release branches created from develop when ready
5. Main branch contains only stable releases
