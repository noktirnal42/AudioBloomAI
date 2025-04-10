// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency

import XCTest
import Combine
import AVFoundation
@testable import AudioProcessor
@testable import AudioBloomCore

/// Shared test utilities for audio processor testing
@available(macOS 15.0, *)
enum AudioProcessorTestUtils {
    
    // MARK: - Mock Audio Components
    
    /// Creates a mock audio source with optional buffer data
    /// - Parameter buffer: Sample data to include in the source output
    /// - Returns: Configured mock audio source
    static func createMockSource(buffer: [Float]? = nil) -> MockAudioSource {
        let source = MockAudioSource()
        if let buffer = buffer {
            source.nextBuffer = buffer
        }
        return source
    }
    
    /// Creates a mock audio sink
    /// - Returns: Configured mock audio sink
    static func createMockSink() -> MockAudioSink {
        return MockAudioSink()
    }
    
    // MARK: - Test Data Generators
    
    /// Creates a test audio buffer with the specified values
    /// - Parameter samples: The sample values to include in the buffer
    /// - Returns: Audio buffer containing the samples
    static func createTestBuffer(samples: [Float]) -> [Float] {
        return samples
    }
    
    /// Generates a sine wave test buffer
    /// - Parameters:
    ///   - frequency: Frequency of the sine wave in Hz
    ///   - sampleRate: Sample rate in Hz
    ///   - duration: Duration in seconds
    /// - Returns: Buffer containing the sine wave
    static func generateSineWaveBuffer(frequency: Float, sampleRate: Float, duration: Float) -> [Float] {
        let sampleCount = Int(duration * sampleRate)
        var buffer = [Float](repeating: 0, count: sampleCount)
        
        for sampleIndex in 0..<sampleCount {
            let phase = 2.0 * Float.pi * frequency * Float(sampleIndex) / sampleRate
            buffer[sampleIndex] = sin(phase)
        }
        
        return buffer
    }
    
    /// Generates a white noise test buffer
    /// - Parameters:
    ///   - amplitude: Amplitude of the noise
    ///   - sampleCount: Number of samples to generate
    /// - Returns: Buffer containing white noise
    static func generateNoiseBuffer(amplitude: Float, sampleCount: Int) -> [Float] {
        var buffer = [Float](repeating: 0, count: sampleCount)
        
        for sampleIndex in 0..<sampleCount {
            let randomValue = Float.random(in: -1.0...1.0)
            buffer[sampleIndex] = randomValue * amplitude
        }
        
        return buffer
    }
    
    // MARK: - Helper Methods
    
    /// Verifies that two audio buffers are approximately equal
    /// - Parameters:
    ///   - buffer1: First buffer to compare
    ///   - buffer2: Second buffer to compare
    ///   - accuracy: Tolerance for floating point comparison
    /// - Returns: Whether the buffers are approximately equal
    static func areBuffersEqual(_ buffer1: [Float], _ buffer2: [Float], accuracy: Float = 0.001) -> Bool {
        guard buffer1.count == buffer2.count else {
            return false
        }
        
        for sampleIndex in 0..<buffer1.count {
            if abs(buffer1[sampleIndex] - buffer2[sampleIndex]) > accuracy {
                return false
            }
        }
        
        return true
    }
}

// MARK: - Mock Audio Components

/// Mock audio source for testing
@available(macOS 15.0, *)
class MockAudioSource: AudioProcessingNode {
    let id = UUID()
    var name = "MockSource"
    var isEnabled = true
    
    /// Buffer to be returned in the next process call
    var nextBuffer: [Float] = []
    
    var inputRequirements: AudioNodeIORequirements {
        let standardFormat = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!
        return AudioNodeIORequirements(
            supportedFormats: [standardFormat],
            channels: .oneOrMore,
            bufferSize: .oneOrMore,
            sampleRates: [48000]
        )
    }
    
    var outputCapabilities: AudioNodeIORequirements {
        let standardFormat = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!
        return AudioNodeIORequirements(
            supportedFormats: [standardFormat],
            channels: .oneOrMore,
            bufferSize: .oneOrMore,
            sampleRates: [48000]
        )
    }
    
    func configure(parameters: [String: Any]) throws {}
    
    func process(inputBuffers: [AudioBufferID], outputBuffers: [AudioBufferID], context: AudioProcessingContext) async throws -> Bool {
        // If there's an output buffer, populate it with the next buffer
        if let outputBufferID = outputBuffers.first {
            try context.updateBuffer(id: outputBufferID, data: &nextBuffer, size: nextBuffer.count * MemoryLayout<Float>.size)
        }
        return true
    }
    
    func reset() {}
}

/// Mock audio sink for testing
@available(macOS 15.0, *)
class MockAudioSink: AudioProcessingNode {
    let id = UUID()
    var name = "MockSink"
    var isEnabled = true
    
    /// Buffers received during processing
    var receivedBuffers: [[Float]] = []
    
    var inputRequirements: AudioNodeIORequirements {
        let standardFormat = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!
        return AudioNodeIORequirements(
            supportedFormats: [standardFormat],
            channels: .oneOrMore,
            bufferSize: .oneOrMore,
            sampleRates: [48000]
        )
    }
    
    var outputCapabilities: AudioNodeIORequirements {
        let standardFormat = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!
        return AudioNodeIORequirements(
            supportedFormats: [standardFormat],
            channels: .oneOrMore,
            bufferSize: .oneOrMore,
            sampleRates: [48000]
        )
    }
    
    func configure(parameters: [String: Any]) throws {}
    
    func process(inputBuffers: [AudioBufferID], outputBuffers: [AudioBufferID], context: AudioProcessingContext) async throws -> Bool {
        // Store any input buffers received
        for inputBufferID in inputBuffers {
            if let buffer = try? context.getBuffer(id: inputBufferID) {
                let bufferPtr = buffer.cpuBuffer?.assumingMemoryBound(to: Float.self)
                if let ptr = bufferPtr {
                    let bufferContents = Array(UnsafeBufferPointer(start: ptr, count: buffer.size / MemoryLayout<Float>.size))
                    receivedBuffers.append(bufferContents)
                }
            }
        }
        return true
    }
    
    func reset() {
        receivedBuffers = []
    }
}

