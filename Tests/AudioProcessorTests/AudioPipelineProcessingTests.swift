import XCTest
@testable import AudioProcessor
@testable import AudioBloomCore

/// Tests for AudioPipelineCore processing capabilities and signal flow
final class AudioPipelineProcessingTests: AudioPipelineBaseTests {

    // MARK: - Setup

    override func setUp() async throws {
        try await super.setUp()

        // Add a basic passthrough node for most tests
        try audioPipeline.registerNode(PassthroughNode())
    }

    // MARK: - Basic Audio Processing Tests

    /// Tests that the pipeline can process audio buffers correctly
    func testBasicBufferProcessing() throws {
        // Prepare a test buffer
        let testBuffer = generateSineWave(frequency: 440.0, duration: 0.1)
        let bufferSize = testBuffer.count

        // Start the pipeline
        try audioPipeline.start()

        // Process the buffer
        let processedBuffer = try audioPipeline.processAudio(testBuffer)

        // Stop the pipeline
        try audioPipeline.stop()

        // Verify the processed buffer
        XCTAssertEqual(processedBuffer.count, bufferSize, "Processed buffer size should match input")

        // With a passthrough node, the buffer should be unchanged
        assertAudioBuffersEqual(testBuffer, processedBuffer)
    }

    /// Tests that signal flows through multiple nodes correctly
    func testSignalFlow() throws {
        // Remove the default passthrough node
        try audioPipeline.reset()

        // Create a chain of processing nodes
        let gainNode = GainNode(gain: 0.5) // Reduce amplitude by half
        let invertNode = InvertNode() // Invert the signal phase

        // Register nodes in the pipeline
        try audioPipeline.registerNode(gainNode)
        try audioPipeline.registerNode(invertNode)

        // Create a test sine wave
        let testBuffer = generateSineWave(frequency: 440.0, duration: 0.1, amplitude: 1.0)

        // Start the pipeline
        try audioPipeline.start()

        // Process the buffer
        let processedBuffer = try audioPipeline.processAudio(testBuffer)

        // Stop the pipeline
        try audioPipeline.stop()

        // Verify that the signal went through both nodes
        // First it should have been reduced in amplitude, then inverted
        for i in 0..<min(testBuffer.count, processedBuffer.count) {
            let expectedSample = -0.5 * testBuffer[i] // Half amplitude and inverted
            XCTAssertEqual(processedBuffer[i], expectedSample, accuracy: 0.001)
        }
    }

    /// Tests that the pipeline handles multi-channel audio correctly
    func testChannelHandling() throws {
        // Create a stereo test buffer
        let monoBuffer = generateSineWave(frequency: 440.0, duration: 0.1)
        let stereoBuffer = generateTestBuffer(data: monoBuffer, channelCount: 2)

        // Register a channel aware node
        try audioPipeline.reset()
        try audioPipeline.registerNode(ChannelAwareNode())

        // Start the pipeline
        try audioPipeline.start()

        // Process the buffer
        let processedBuffer = try audioPipeline.processAudio(stereoBuffer)

        // Stop the pipeline
        try audioPipeline.stop()

        // Verify channel handling - ChannelAwareNode applies different gain to each channel
        // Left channel (even indices) should be amplified by 2.0
        // Right channel (odd indices) should be attenuated by 0.5
        for i in 0..<processedBuffer.count/2 {
            let leftIndex = i * 2
            let rightIndex = i * 2 + 1

            XCTAssertEqual(processedBuffer[leftIndex], stereoBuffer[leftIndex] * 2.0, accuracy: 0.001)
            XCTAssertEqual(processedBuffer[rightIndex], stereoBuffer[rightIndex] * 0.5, accuracy: 0.001)
        }
    }

    // MARK: - Signal Processing Tests

    /// Tests that gain adjustment is applied correctly
    func testGainAdjustment() throws {
        // Configure the pipeline with a gain node
        try audioPipeline.reset()
        let gainNode = GainNode(gain: 2.0) // Double the amplitude
        try audioPipeline.registerNode(gainNode)

        // Create a test signal
        let testBuffer = generateSineWave(frequency: 440.0, duration: 0.1, amplitude: 0.5)

        // Start the pipeline
        try audioPipeline.start()

        // Process the buffer
        let processedBuffer = try audioPipeline.processAudio(testBuffer)

        // Stop the pipeline
        try audioPipeline.stop()

        // Calculate RMS level of both buffers
        let inputRMS = calculateRMSLevel(testBuffer)
        let outputRMS = calculateRMSLevel(processedBuffer)

        // Verify that gain was applied correctly (RMS should be doubled)
        XCTAssertEqual(outputRMS, inputRMS * 2.0, accuracy: 0.01)

        // Verify individual samples
        for i in 0..<min(testBuffer.count, processedBuffer.count) {
            XCTAssertEqual(processedBuffer[i], testBuffer[i] * 2.0, accuracy: 0.001)
        }
    }

    /// Tests that sample rate conversion is handled correctly
    func testSampleRateConversion() throws {
        // Create a sample rate conversion node
        try audioPipeline.reset()
        let sampleRateNode = SampleRateConverterNode(
            inputRate: defaultSampleRate,
            outputRate: defaultSampleRate * 2.0 // Double the sample rate
        )
        try audioPipeline.registerNode(sampleRateNode)

        // Create a test signal
        let testBuffer = generateSineWave(frequency: 440.0, duration: 0.1)
        let inputSampleCount = testBuffer.count

        // Start the pipeline
        try audioPipeline.start()

        // Process the buffer
        let processedBuffer = try audioPipeline.processAudio(testBuffer)

        // Stop the pipeline
        try audioPipeline.stop()

        // Verify the sample rate conversion
        // Output buffer should have approximately twice as many samples
        XCTAssertEqual(processedBuffer.count, inputSampleCount * 2, accuracy: 10)

        // Verify the signal properties are preserved (check frequency content, etc.)
        // This would typically involve additional signal analysis
        // For this test we'll just verify that the peak amplitude is preserved
        let inputPeak = testBuffer.map { abs($0) }.max() ?? 0
        let outputPeak = processedBuffer.map { abs($0) }.max() ?? 0
        XCTAssertEqual(outputPeak, inputPeak, accuracy: 0.1)
    }

    /// Tests channel mapping operations
    func testChannelMapping() throws {
        // Configure pipeline with a channel mapper node
        try audioPipeline.reset()
        let channelMapperNode = ChannelMapperNode()
        try audioPipeline.registerNode(channelMapperNode)

        // Create a stereo test buffer with different content in each channel
        let leftChannel = generateSineWave(frequency: 440.0, duration: 0.1)
        let rightChannel = generateSineWave(frequency: 880.0, duration: 0.1)

        var stereoBuffer = [Float](repeating: 0.0, count: leftChannel.count * 2)
        for i in 0..<leftChannel.count {
            stereoBuffer[i * 2] = leftChannel[i]       // Left channel
            stereoBuffer[i * 2 + 1] = rightChannel[i]  // Right channel
        }

        // Start the pipeline
        try audioPipeline.start()

        // Process the buffer
        let processedBuffer = try audioPipeline.processAudio(stereoBuffer)

        // Stop the pipeline
        try audioPipeline.stop()

        // ChannelMapperNode should swap left and right channels
        for i in 0..<leftChannel.count {
            // Processed buffer should have channels swapped
            XCTAssertEqual(processedBuffer[i * 2], stereoBuffer[i * 2 + 1], accuracy: 0.001)
            XCTAssertEqual(processedBuffer[i * 2 + 1], stereoBuffer[i * 2], accuracy: 0.001)
        }
    }

    // MARK: - Error Handling Tests

    /// Tests that the pipeline handles invalid buffer sizes correctly
    func testInvalidBufferHandling() throws {
        // Start the pipeline
        try audioPipeline.start()

        // Try to process a buffer with invalid size (not matching pipeline buffer size)
        let invalidBuffer = [Float](repeating: 0.0, count: 123) // Not a power of 2

        XCTAssertThrowsError(try audioPipeline.processAudio(invalidBuffer)) { error in
            // Verify the error type
            XCTAssertTrue(error is AudioPipelineError, "Should throw an AudioPipelineError")
        }

        // Try to process an empty buffer
        XCTAssertThrowsError(try audioPipeline.processAudio([])) { error in
            XCTAssertTrue(error is AudioPipelineError, "Should throw an AudioPipelineError")
        }

        // Stop the pipeline
        try audioPipeline.stop()
    }

    /// Tests that processing errors from nodes are handled correctly
    func testProcessingErrors() throws {
        // Register a node that throws an error during processing
        try audioPipeline.reset()
        try audioPipeline.registerNode(ErrorThrowingNode())

        // Create a valid test buffer
        let testBuffer = generateSineWave(frequency: 440.0, duration: 0.1)

        // Start the pipeline
        try audioPipeline.start()

        // Process the buffer - should throw
        XCTAssertThrowsError(try audioPipeline.processAudio(testBuffer)) { error in
            // Verify it's the expected error
            guard case ErrorThrowingNode.TestError.processingFailed = error else {
                XCTFail("Unexpected error type: \(error)")
                return
            }
        }

        // Stop the pipeline
        try audioPipeline.stop()
    }

    /// Tests that errors propagate correctly through the pipeline
    func testErrorPropagation() throws {
        // Register multiple nodes, with one that throws an error
        try audioPipeline.reset()
        let counterNode = OperationCounterNode()
        try audioPipeline.registerNode(PassthroughNode())
        try audioPipeline.registerNode(counterNode)
        try audioPipeline.registerNode(ErrorThrowingNode())
        try audioPipeline.registerNode(PassthroughNode())

        // Create a valid test buffer
        let testBuffer = generateSineWave(frequency: 440.0, duration: 0.1)

        // Start the pipeline
        try audioPipeline.start()

        // Process the buffer - should throw
        XCTAssertThrowsError(try audioPipeline.processAudio(testBuffer))

        // Stop the pipeline
        try audioPipeline.stop()

        // Verify that only nodes before the error-throwing node were processed
        XCTAssertEqual(counterNode.processCount, 1, "Node before error should be processed once")
    }

    // MARK: - Test Node Implementations

    /// Simple passthrough node that doesn't modify the signal
    class PassthroughNode: AudioProcessingNode {
        func process(buffer: [Float], config: AudioProcessingConfig) throws -> [Float] {
            return buffer
        }
    }

    /// Node that applies a fixed gain to the signal
    class GainNode: AudioProcessingNode {
        let gain: Float

        init(gain: Float) {
            self.gain = gain
        }

        func process(buffer: [Float], config: AudioProcessingConfig) throws -> [Float] {
            return buffer.map { $0 * gain }
        }
    }

    /// Node that inverts the phase of the signal
    class InvertNode: AudioProcessingNode {
        func process(buffer: [Float], config: AudioProcessingConfig) throws -> [Float] {
            return buffer.map { -$0 }
        }
    }

    /// Node that applies different processing to each channel
    class ChannelAwareNode: AudioProcessingNode {
        func process(buffer: [Float], config: AudioProcessingConfig) throws -> [Float] {
            var result = buffer

            for i in stride(from: 0, to: buffer.count, by: 2) {
                if i < buffer.count {
                    result[i] = buffer[i] * 2.0 // Amplify left channel
                }

                if i + 1 < buffer.count {
                    result[i + 1] = buffer[i + 1] * 0.5 // Attenuate right channel
                }
            }

            return result
        }
    }

    /// Node that converts between sample rates
    class SampleRateConverterNode: AudioProcessingNode {
        let inputRate: Double
        let inputToOutputRatio: Double

        init(inputRate: Double, outputRate: Double) {
            self.inputRate = inputRate
            self.inputToOutputRatio = outputRate / inputRate
        }

        func process(buffer: [Float], config: AudioProcessingConfig) throws -> [Float] {
            // Simple sample rate conversion simulation
            let outputSize = Int(Double(buffer.count) * inputToOutputRatio)
            var output = [Float](repeating: 0.0, count: outputSize)

            for i in 0..<outputSize {
                let inputIndex = Double(i) / inputToOutputRatio
                let inputIndexInt = Int(inputIndex)

                if inputIndexInt < buffer.count {
                    output[i] = buffer[inputIndexInt]
                }
            }

            return output
        }
    }

    /// Node that swaps stereo channels
    class ChannelMapperNode: AudioProcessingNode {
        func process(buffer: [Float], config: AudioProcessingConfig) throws -> [Float] {
            guard config.channelCount == 2 else {
                // Only works on stereo signals
                return buffer
            }

            var output = buffer

            for i in stride(from: 0, to: buffer.count, by: 2) {
                if i + 1 < buffer.count {
                    // Swap left and right channels
                    output[i] = buffer[i + 1]
                    output[i + 1] = buffer[i]
                }
            }

            return output
        }
    }

    /// Node that throws an error during processing
    class ErrorThrowingNode: AudioProcessingNode {
        enum TestError: Error {
            case processingFailed
        }

        func process(buffer: [Float], config: AudioProcessingConfig) throws -> [Float] {
            throw TestError.processingFailed
        }
    }

    /// Node that counts how many times it has processed audio
    class OperationCounterNode: AudioProcessingNode {
        var processCount = 0

        func process(buffer: [Float], config: AudioProcessingConfig) throws -> [Float] {
            processCount += 1
            return buffer
        }
    }
}

