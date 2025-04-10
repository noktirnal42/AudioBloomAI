import XCTest
import AVFoundation
@testable import AudioProcessor
@testable import AudioBloomCore

/// Tests for performance characteristics of the FFT processing node
final class FFTProcessorPerformanceTests: FFTProcessorBaseTests {

    // MARK: - Performance Tests

    /// Tests the processing performance of the FFT node
    func testProcessingPerformance() async throws {
        // Create a complex test signal with multiple frequency components
        let frequencies = [261.63, 440.0, 880.0, 1760.0, 3520.0]
        let amplitudes: [Float] = [0.8, 0.6, 0.4, 0.3, 0.2]

        let testSignal = generateMultiFrequencySignal(
            frequencies: frequencies,
            amplitudes: amplitudes,
            sampleCount: defaultFFTSize,
            sampleRate: sampleRate
        )

        // Measure the processing performance
        let iterations = 100

        measure {
            // Process the signal multiple times to get a good measurement
            for _ in 0..<iterations {
                do {
                    _ = try processTestData(testSignal)
                } catch {
                    XCTFail("FFT processing failed: \(error)")
                }
            }
        }

        // Alternative direct measurement
        let startTime = DispatchTime.now()

        for _ in 0..<iterations {
            _ = try await processTestData(testSignal)
        }

        let endTime = DispatchTime.now()
        let elapsedNanoseconds = endTime.uptimeNanoseconds - startTime.uptimeNanoseconds
        let elapsedMilliseconds = Double(elapsedNanoseconds) / 1_000_000.0
        let timePerOperation = elapsedMilliseconds / Double(iterations)

        // For real-time audio processing, the processing time should be less than
        // the buffer duration to avoid dropouts
        let bufferDurationMs = (Double(defaultFFTSize) / sampleRate) * 1000.0

        XCTAssertLessThan(
            timePerOperation,
            bufferDurationMs,
            "FFT processing should be faster than real-time: \(timePerOperation) ms per operation"
        )
    }

    /// Tests the real-time capability with continuous processing
    func testRealTimeCapability() async throws {
        // Configure a more demanding scenario
        try fftNode.setFFTSize(4096) // Larger FFT size
        try fftNode.setWindowFunction(.blackman) // More complex window function

        // Create a complex signal
        let testSignal = generateMultiFrequencySignal(
            frequencies: [110.0, 220.0, 440.0, 880.0, 1760.0, 3520.0, 7040.0],
            amplitudes: [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            sampleCount: 4096,
            sampleRate: sampleRate
        )

        // Simulate real-time audio processing by processing
        // buffers at intervals matching the buffer duration
        let bufferDuration = Double(4096) / sampleRate
        let iterations = 20
        var processingTimes: [TimeInterval] = []

        for _ in 0..<iterations {
            let startTime = Date()

            // Process the buffer
            _ = try await processTestData(testSignal)

            let processingTime = Date().timeIntervalSince(startTime)
            processingTimes.append(processingTime)

            // Simulate waiting for the next buffer
            if processingTime < bufferDuration {
                try await Task.sleep(nanoseconds: UInt64((bufferDuration - processingTime) * 1_000_000_000))
            }
        }

        // Calculate statistics
        let maxProcessingTime = processingTimes.max() ?? 0
        let avgProcessingTime = processingTimes.reduce(0, +) / Double(processingTimes.count)

        // For reliable real-time processing, max time should be less than buffer duration
        XCTAssertLessThan(
            maxProcessingTime,
            bufferDuration,
            "Maximum processing time should be less than buffer duration"
        )

        // Average time should be significantly less than buffer duration
        XCTAssertLessThan(
            avgProcessingTime,
            bufferDuration * 0.7,
            "Average processing time should be well below buffer duration"
        )

        // Reset FFT size for other tests
        try fftNode.setFFTSize(defaultFFTSize)
    }

    /// Tests the memory usage patterns during FFT processing
    func testMemoryUsage() throws {
        // Metric for tracking memory usage
        measure(metrics: [XCTMemoryMetric()]) {
            // Create FFT nodes with different sizes to test memory scaling
            let fftSizes = [512, 1024, 2048, 4096, 8192]
            var nodes: [FFTProcessingNode] = []

            for fftSize in fftSizes {
                let node = FFTProcessingNode(name: "Memory Test \(fftSize)", fftSize: fftSize)
                nodes.append(node)
            }

            // Create test signals of different sizes
            var signals: [[Float]] = []
            for fftSize in fftSizes {
                let signal = generateSineWave(
                    frequency: 440.0,
                    amplitude: 0.8,
                    sampleCount: fftSize,
                    sampleRate: sampleRate
                )
                signals.append(signal)
            }

            // Perform FFT operations
            for (index, node) in nodes.enumerated() {
                for _ in 0..<10 { // Multiple iterations to stress memory
                    let input = signals[index]
                    let output = node.processBuffer(input)
                    // Use the output to prevent compiler optimization
                    XCTAssertNotNil(output)
                }
            }

            // Clean up
            nodes.removeAll()
            signals.removeAll()
        }
    }

    // MARK: - Configuration Impact Tests

    /// Tests the impact of different FFT sizes on performance
    func testFFTSizePerformanceImpact() async throws {
        // Test different FFT sizes
        let fftSizes = [512, 1024, 2048, 4096, 8192]
        var processingTimes: [Int: Double] = [:]

        for fftSize in fftSizes {
            // Configure node with the test FFT size
            try fftNode.setFFTSize(fftSize)

            // Create a test signal
            let testSignal = generateSineWave(
                frequency: 440.0,
                amplitude: 0.8,
                sampleCount: fftSize,
                sampleRate: sampleRate
            )

            // Measure processing time
            let iterations = 20
            let startTime = DispatchTime.now()

            for _ in 0..<iterations {
                _ = try await processTestData(testSignal)
            }

            let endTime = DispatchTime.now()
            let elapsedNanoseconds = endTime.uptimeNanoseconds - startTime.uptimeNanoseconds
            let elapsedMilliseconds = Double(elapsedNanoseconds) / 1_000_000.0

            // Record the average time per operation
            processingTimes[fftSize] = elapsedMilliseconds / Double(iterations)
        }

        // Log the results
        for (size, time) in processingTimes.sorted(by: { $0.key < $1.key }) {
            print("FFT size \(size): \(time) ms per operation")
        }

        // Verify that processing time scales with FFT size
        // FFT complexity is O(n log n), so doubling the size should less than double the time
        for index in 1..<fftSizes.count {
            let smallerSize = fftSizes[index - 1]
            let largerSize = fftSizes[index]
            let smallerTime = processingTimes[smallerSize] ?? 0
            let largerTime = processingTimes[largerSize] ?? 0

            // The ratio of times should be less than the ratio of sizes
            // For an O(n log n) algorithm, we expect:
            // time_ratio < size_ratio * log(size_ratio)
            let sizeRatio = Double(largerSize) / Double(smallerSize)
            let expectedMaxTimeRatio = sizeRatio * log2(sizeRatio)
            let actualTimeRatio = largerTime / smallerTime

            XCTAssertLessThan(
                actualTimeRatio,
                expectedMaxTimeRatio,
                "Processing time should scale with FFT size complexity"
            )
        }

        // Reset FFT size for other tests
        try fftNode.setFFTSize(defaultFFTSize)
    }

    /// Tests the impact of window functions on performance
    func testWindowFunctionPerformanceImpact() async throws {
        // Test different window functions
        let windowFunctions: [WindowFunction] = [.none, .hann, .hamming, .blackman]
        var processingTimes: [WindowFunction: Double] = [:]

        for windowFunction in windowFunctions {
            // Configure node with the test window function
            try fftNode.setWindowFunction(windowFunction)

            // Create a test signal
            let testSignal = generateSineWave(
                frequency: 440.0,
                amplitude: 0.8,
                sampleCount: defaultFFTSize,
                sampleRate: sampleRate
            )

            // Measure processing time
            let iterations = 20
            let startTime = DispatchTime.now()

            for _ in 0..<iterations {
                _ = try await processTestData(testSignal)
            }

            let endTime = DispatchTime.now()
            let elapsedNanoseconds = endTime.uptimeNanoseconds - startTime.uptimeNanoseconds
            let elapsedMilliseconds = Double(elapsedNanoseconds) / 1_000_000.0

            // Record the average time per operation
            processingTimes[windowFunction] = elapsedMilliseconds / Double(iterations)
        }

        // Log the results
        for (function, time) in processingTimes {
            print("Window function \(function): \(time) ms per operation")
        }

        // Verify that using a window function adds reasonable overhead
        let noWindowTime = processingTimes[.none] ?? 0

        for (function, time) in processingTimes where function != .none {
            // Window functions should add less than 50% overhead
            XCTAssertLessThan(
                time,
                noWindowTime * 1.5,
                "Window function \(function) should not significantly impact performance"
            )
        }

        // Reset window function for other tests
        try fftNode.setWindowFunction(.hann)
    }
}
