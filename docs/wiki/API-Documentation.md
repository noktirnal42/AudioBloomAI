# API Documentation

## AudioBloomCore

### AudioBloomSettings

```swift
public protocol AudioBloomSettings {
    var audioConfiguration: AudioConfiguration { get set }
    var visualizationPreferences: VisualizationPreferences { get set }
    var performanceSettings: PerformanceSettings { get set }
}
```

#### Key Components
- Audio configuration management
- Visualization preferences
- Performance tuning
- State persistence

### AudioDataPublisher

```swift
public protocol AudioDataPublisher {
    func publishAudioData(_ data: AudioData)
    func subscribe(_ subscriber: AudioDataSubscriber)
    func unsubscribe(_ subscriber: AudioDataSubscriber)
}
```

#### Features
- Real-time audio data distribution
- Subscriber management
- Data transformation
- Error handling

## AudioProcessor

### AudioFeatureExtractor

```swift
public protocol AudioFeatureExtractor {
    func extractFeatures(from buffer: AudioBuffer) -> AudioFeatures
    var configuration: FeatureExtractionConfig { get set }
}
```

#### Capabilities
- Frequency analysis
- Amplitude detection
- Pattern recognition
- Real-time processing

### AudioBufferManager

```swift
public protocol AudioBufferManager {
    func prepareBuffer(size: Int) -> AudioBuffer
    func processBuffer(_ buffer: AudioBuffer)
    func releaseBuffer(_ buffer: AudioBuffer)
}
```

#### Features
- Buffer allocation
- Memory management
- Processing queue
- Resource optimization

## Visualizer

### VisualizationRenderer

```swift
public protocol VisualizationRenderer {
    func render(audioFeatures: AudioFeatures)
    func updateRenderState(_ state: RenderState)
    var configuration: RenderConfiguration { get set }
}
```

#### Capabilities
- Metal shader management
- Frame rendering
- State handling
- Performance optimization

### VisualizationController

```swift
public protocol VisualizationController {
    func updateVisualization(with features: AudioFeatures)
    func switchEffect(to effect: VisualizationEffect)
    func configureEffect(_ effect: VisualizationEffect, parameters: EffectParameters)
}
```

#### Features
- Effect management
- Transition handling
- Parameter control
- State coordination

## Data Types

### AudioData

```swift
public struct AudioData {
    let timestamp: TimeInterval
    let samples: [Float]
    let format: AudioFormat
}
```

### AudioFeatures

```swift
public struct AudioFeatures {
    let frequencies: [Float]
    let amplitudes: [Float]
    let tempo: Float
    let timestamp: TimeInterval
}
```

### VisualizationEffect

```swift
public struct VisualizationEffect {
    let type: EffectType
    var parameters: EffectParameters
    var state: EffectState
}
```

## Error Handling

### AudioProcessingError

```swift
public enum AudioProcessingError: Error {
    case bufferOverflow
    case processingFailed
    case invalidConfiguration
    case resourceUnavailable
}
```

### VisualizationError

```swift
public enum VisualizationError: Error {
    case shaderCompilationFailed
    case renderingFailed
    case invalidEffectState
    case resourceLimit
}
```

## Performance Guidelines

### Audio Processing
- Buffer size recommendations
- Processing thread management
- Memory usage optimization
- Error recovery strategies

### Visualization
- Frame rate targets
- Metal resource management
- State transition handling
- Memory constraints

## Best Practices

### Implementation Guidelines
- Error handling patterns
- Resource management
- State synchronization
- Performance optimization

### Threading Considerations
- Main thread usage
- Background processing
- Synchronization methods
- Thread safety

## Version Compatibility

### Current Version
- Swift 6 compatibility
- macOS 15+ support
- Metal API requirements
- Minimum system specifications

### Future Considerations
- API evolution
- Backward compatibility
- Deprecation policies
- Migration strategies
