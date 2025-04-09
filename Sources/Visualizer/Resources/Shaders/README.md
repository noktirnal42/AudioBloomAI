# AudioBloomAI Shader System

## Overview

The AudioBloomAI shader system provides real-time audio visualization through Metal shaders that respond to audio input. These shaders transform audio analysis data into dynamic visual elements using Apple's Metal framework.

## Shader Files

- **AudioVisualizerShader.metal**: Core shader that handles audio visualization rendering
  - Contains parameters for audio analysis, visual theming, and animation
  - Implements multiple visualization modes (Spectrum, Waveform, Particles, Neural)

## Parameter Structure

The `AudioVisualizerParameters` structure contains all necessary parameters for the visualization:

### Audio Analysis Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `bassLevel` | float | Low frequency intensity (typically 20-250Hz) |
| `midLevel` | float | Mid frequency intensity (typically 250-2000Hz) |
| `trebleLevel` | float | High frequency intensity (typically 2000-20000Hz) |
| `leftLevel` | float | Left channel volume level (0.0-1.0) |
| `rightLevel` | float | Right channel volume level (0.0-1.0) |

### Theme Colors

| Parameter | Type | Description |
|-----------|------|-------------|
| `primaryColor` | float4 | Primary theme color (RGBA) |
| `secondaryColor` | float4 | Secondary theme color (RGBA) |
| `backgroundColor` | float4 | Background color (RGBA) |
| `accentColor` | float4 | Accent color for highlights (RGBA) |

### Animation Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `time` | float | Current time in seconds (for animations) |
| `sensitivity` | float | Audio sensitivity multiplier (0.0-1.0) |
| `motionIntensity` | float | Motion intensity multiplier (0.0-1.0) |
| `themeIndex` | float | Current theme index (0-7 for different themes) |

### Visualization Settings

| Parameter | Type | Description |
|-----------|------|-------------|
| `visualizationMode` | float | Current visualization mode:<br>0: Spectrum<br>1: Waveform<br>2: Particles<br>3: Neural |
| `previousMode` | float | Previous visualization mode (for transitions) |
| `transitionProgress` | float | Progress between mode transitions (0.0-1.0) |
| `colorIntensity` | float | Color intensity parameter (0.0-1.0) |

## Usage Guidelines

### Integration with Swift

To use these shaders in your Swift code:

```swift
// Initialize Metal pipeline
let device = MTLCreateSystemDefaultDevice()!
let library = device.makeDefaultLibrary()!
let function = library.makeFunction(name: "audioVisualizerFragment")!

// Create shader parameters
var params = AudioVisualizerParameters()
params.bassLevel = 0.5
params.midLevel = 0.3
params.trebleLevel = 0.2
// ... set other parameters

// Create a buffer from the parameters
let paramBuffer = device.makeBuffer(
    bytes: &params,
    length: MemoryLayout<AudioVisualizerParameters>.stride,
    options: .storageModeShared
)

// Set the buffer in your render command encoder
encoder.setFragmentBuffer(paramBuffer, offset: 0, index: 0)
```

### Modifying Visualization Modes

Each visualization mode uses different techniques to represent audio data:

1. **Spectrum Mode (0)**: Displays frequency data as vertical bars or a continuous curve
2. **Waveform Mode (1)**: Shows the audio waveform directly
3. **Particles Mode (2)**: Uses particle systems affected by audio data
4. **Neural Mode (3)**: Neural network-enhanced visualization patterns

When implementing a new visualization mode:
- Add a new constant for the mode
- Implement the visualization in the fragment shader
- Add transition logic between the new mode and existing modes

### Performance Considerations

- Use `packed_float4` for color values to optimize memory usage
- Consider using half-precision floats for parameters when full precision isn't needed
- Implement level-of-detail adjustments based on device performance

## Development Workflow

1. Make changes to shader parameters in the `AudioVisualizerParameters` structure
2. Test with various audio inputs to ensure responsive visualization
3. Document any new parameters or visualization modes
4. Ensure backward compatibility with existing parameter structures

## Future Improvements

- Support for dynamic texture inputs
- Audio beat detection parameters
- User customizable color palettes
- Per-mode specific parameters
- Hardware-accelerated FFT processing for audio analysis

