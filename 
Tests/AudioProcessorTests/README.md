# Audio Processor Tests

This directory contains test suites for the AudioProcessor module.

## Test Organization

- AudioPipeline Tests:
  - Basic processing and configuration
  - Channel handling and mapping
  - Performance and resource management
  - Error handling and recovery

- FFT Processing Tests:
  - Basic FFT operations
  - Window functions
  - Frequency analysis
  - Band processing

- Metal Compute Tests:
  - Core functionality
  - Buffer management
  - Shader execution
  - Performance

## Running Tests

```bash
swift test --filter "AudioProcessor"
```

## Test Files Structure

### Audio Pipeline Tests

- AudioPipelineTestHelpers.swift - Shared utilities for testing
- AudioPipelineBaseTests.swift - Base class with common setup
- AudioPipelineChannelTests.swift - Tests for stereo and multi-channel processing
- AudioPipelineCoreSetupTests.swift - Core initialization and setup
- AudioPipelineCoreBufferTests.swift - Buffer management
- AudioPipelineCoreProcessingTests.swift - Audio processing
- AudioPipelineCoreErrorTests.swift - Error handling and recovery
- AudioPipelineCoreStateTests.swift - Pipeline state management
- AudioPipelinePerformanceTests.swift - Performance tests

### FFT Processing Tests

- FFTProcessorBaseTests.swift - Base class with common setup
- FFTProcessingInitTests.swift - Initialization tests
- FFTProcessingWindowTests.swift - Window function tests
- FFTProcessingFrequencyTests.swift - Frequency analysis
- FFTProcessingBandTests.swift - Frequency band tests
- FFTProcessingMemoryTests.swift - Memory management tests
- FFTProcessorPerformanceTests.swift - Performance tests

### Metal Compute Tests

- MetalComputeBaseTests.swift - Base class with common setup
- MetalComputeInitTests.swift - Initialization and setup
- MetalComputeBufferTests.swift - Buffer management
- MetalComputeShaderTests.swift - Shader execution
- MetalComputeSyncTests.swift - Synchronization tests
- MetalComputePerformanceTests.swift - Performance tests
## Test Coverage Guidelines

### Required Test Coverage

- Unit Tests: 90% or higher
- Integration Tests: 80% or higher
- Performance Tests: Key operations only

### Test Categories

1. Functional Tests
   - Input validation
   - Output verification
   - Edge cases
   - Error conditions

2. Integration Tests
   - Component interaction
   - Pipeline flow
   - Resource management
   - State transitions

3. Performance Tests
   - Processing latency
   - Memory usage
   - Resource cleanup
   - Load handling

### Writing New Tests

1. Test Setup
   - Use AudioPipelineTestHelpers for common operations
   - Initialize components with known states
   - Clean up resources in tearDown

2. Test Structure
   - Clear test names describing functionality
   - Comprehensive verification points
   - Error case coverage
   - Resource cleanup

3. Documentation
   - Document test purpose
   - List verification points
   - Note any special requirements
   - Include example usage where helpful
