import XCTest
@testable import AudioProcessor
@testable import AudioBloomCore

/// Tests focused on the performance characteristics of the AudioPipelineCore
final class AudioPipelinePerformanceTests: AudioPipelineBaseTests {

    // MARK: - Setup

    /// Standard buffer size used for performance testing
    let performanceBufferSize = 8192

    /// Number of test iterations to run for performance tests
    let iterationCount = 10

    /// Prepare a specific audio pipeline configuration for performance testing
    override func setUp() async throws {
        try await super.setUp()

        // Use a larger buffer size for performance tests
        try audioPipeline.updateConfiguration(
            bufferSize: performanceBufferSize,
            sampleRate: nil,
            channelCount: nil
        )
    }

    // MARK: - Performance Benchmarks

    /// Measures the raw throughput of the pipeline with different node counts
    func testProcessingThroughput() throws {
        // Setup different pipeline configurations to test
        let configurations = [
            ("Empty Pipeline", 0),
            ("Single Node", 1),
            ("Five Nodes", 5),
            ("Ten Nodes", 10)
        ]

        // Test buffer to process (1 second of audio)
        let testBuffer = generateSineWave(
            frequency: 440.0,
            sampleRate: Double(defaultSampleRate),
            duration: 1.0
        )

        for (name, nodeCount) in configurations {
            // Configure pipeline with the specified number of nodes
            try audioPipeline.reset()

            for _ in 0..<nodeCount {
                try audioPipeline.registerNode(PassthroughNode())
            }

            // Start pipeline
            try audioPipeline.start()

            // Measure performance
            measure(metrics: [XCTClockMetric(), XCTCPUMetric(), XCTMemoryMetric()]) {
                for _ in 0..<iterationCount {
                    // Process the buffer repeatedly to get a good measurement
                    _ = try? audioPipeline.processAudio(testBuffer)
                }
            }

            // Stop pipeline
            try audioPipeline.stop()

            // Report results
            print("\(name): Processed \(iterationCount) buffers of \(testBuffer.count) samples")
        }
    }

    /// Tests processing performance with computationally intensive nodes
    func testIntensiveProcessing() throws {
        // Create a test buffer
        let testBuffer = generateSineWave(frequency: 440.0, duration: 1.0)

        // Reset pipeline and add a computationally intensive node
        try audioPipeline.reset()
        try audioPipeline.registerNode(IntensiveProcessingNode())

        // Start pipeline
        try audioPipeline.start()

        // Measure performance
        measure(metrics: [XCTClockMetric(), XCTCPUMetric()]) {
            for _ in 0..<5 { // Fewer iterations for intensive processing
                _ = try? audioPipeline.processAudio(testBuffer)
            }
        }

        // Stop pipeline
        try audioPipeline.stop()
    }

    // MARK: - Resource Utilization Tests

    /// Tests memory allocation patterns during audio processing
    func testMemoryUtilization() throws {
        // Reset pipeline with nodes that allocate memory during processing
        try audioPipeline.reset()
        try audioPipeline.registerNode(MemoryAllocatingNode())

        // Start the pipeline
        try audioPipeline.start()

        // Track memory usage
        var initialMemory: mach_vm_size_t = 0
        var peakMemory: mach_vm_size_t = 0

        // Record initial memory
        initialMemory = currentMemoryUsage()

        // Process multiple buffers
        for _ in 0..<20 {
            let testBuffer = generateSineWave(frequency: 440.0, duration: 0.1)
            _ = try audioPipeline.processAudio(testBuffer)

            // Update peak memory
            let currentMemory = currentMemoryUsage()
            peakMemory = max(peakMemory, currentMemory)
        }

        // Stop the pipeline
        try audioPipeline.stop()

        // Verify memory usage
        XCTAssertLessThan(
            peakMemory - initialMemory,
            mach_vm_size_t(10 * 1024 * 1024), // 10MB limit
            "Memory usage should be reasonable"
        )

        // Force garbage collection and verify cleanup
        autoreleasepool {
            // Force memory cleanup
        }

        // Verify memory returned to near initial state
        let finalMemory = currentMemoryUsage()
        XCTAssertLessThan(
            finalMemory - initialMemory,
            mach_vm_size_t(1 * 1024 * 1024), // 1MB tolerance
            "Memory should be properly released"
        )
    }

    /// Tests CPU utilization during audio processing
    func testCPUUtilization() throws {
        // Configure pipeline with CPU-intensive nodes
        try audioPipeline.reset()

        // Add a mix of nodes with different CPU requirements
        try audioPipeline.registerNode(PassthroughNode())
        try audioPipeline.registerNode(IntensiveProcessingNode())

        // Start the pipeline
        try audioPipeline.start()

        // Create a large test buffer
        let testBuffer = generateSineWave(frequency: 440.0, duration: 2.0)

        // Measure CPU time
        let startTime = DispatchTime.now()
        let startCPU = currentCPUTime()

        // Process audio
        for _ in 0..<5 {
            _ = try audioPipeline.processAudio(testBuffer)
        }

        // Calculate CPU usage
        let endCPU = currentCPUTime()
        let endTime = DispatchTime.now()

        // Convert to milliseconds
        let elapsedTime = Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000
        let cpuTime = endCPU - startCPU

        // Stop the pipeline
        try audioPipeline.stop()

        // Calculate CPU utilization as a percentage
        let utilization = (cpuTime / elapsedTime) * 100.0

        // Real-time audio processing should use reasonable CPU
        XCTAssertLessThan(utilization, 80.0, "CPU utilization should be under 80%")
        print("CPU Utilization: \(utilization)%")
    }

    // MARK: - Concurrency Tests

    /// Tests pipeline performance with concurrent audio processing
    func testConcurrentProcessing() throws {
        // Configure a pipeline
        try audioPipeline.reset()
        try audioPipeline.registerNode(PassthroughNode())

        // Start the pipeline
        try audioPipeline.start()

        // Create test buffer
        let testBuffer = generateSineWave(frequency: 440.0, duration: 0.1)

        // Measure parallel performance
        measure {
            DispatchQueue.concurrentPerform(iterations: 10) { _ in
                // Process audio on multiple threads
                _ = try? audioPipeline.processAudio(testBuffer)
            }
        }

        // Stop the pipeline
        try audioPipeline.stop()
    }

    /// Tests performance impact of runtime configuration changes
    func testDynamicReconfiguration() throws {
        // Start with a simple pipeline
        try audioPipeline.reset()
        try audioPipeline.registerNode(PassthroughNode())

        // Start the pipeline
        try audioPipeline.start()

        // Create test buffer
        let testBuffer = generateSineWave(frequency: 440.0, duration: 0.1)

        // Baseline performance
        var baselineTime: TimeInterval = 0

        measure {
            baselineTime = XCTClock().measure {
                for _ in 0..<10 {
                    _ = try? audioPipeline.processAudio(testBuffer)
                }
            }
        }

        // Stop the pipeline to reconfigure
        try audioPipeline.stop()

        // Add and remove nodes repeatedly
        var reconfigurationTime: TimeInterval = 0

        measure {
            reconfigurationTime = XCTClock().measure {
                for _ in 0..<5 {
                    // Add a node
                    try? audioPipeline.registerNode(PassthroughNode())
                    try? audioPipeline.start()

                    // Process some audio
                    _ = try? audioPipeline.processAudio(testBuffer)

                    // Stop and remove the node
                    try? audioPipeline.stop()
                    try? audioPipeline.reset()
                    try? audioPipeline.registerNode(PassthroughNode())
                }
            }
        }

        // Compare times
        XCTAssertLessThan(
            reconfigurationTime / baselineTime,
            5.0,
            "Dynamic reconfiguration overhead should be reasonable"
        )
    }

    // MARK: - Memory Management Tests

    /// Tests for memory leaks during pipeline lifecycle
    func testMemoryLeaks() throws {
        // Track initial memory usage
        let initialMemory = currentMemoryUsage()

        for _ in 0..<10 {
            // Create and destroy pipelines repeatedly
            let pipeline = AudioPipelineCore()
            try pipeline.registerNode(MemoryAllocatingNode())
            try pipeline.start()

            let testBuffer = generateSineWave(frequency: 440.0, duration: 0.1)
            _ = try pipeline.processAudio(testBuffer)

            try pipeline.stop()
            // Pipeline will be deallocated when it goes out of scope
        }

        // Force garbage collection
        autoreleasepool {
            // Force memory cleanup
        }

        // Check final memory usage
        let finalMemory = currentMemoryUsage()
        XCTAssertLessThan(
            finalMemory - initialMemory,
            mach_vm_size_t(2 * 1024 * 1024), // 2MB tolerance
            "Memory should be properly released after pipeline destruction"
        )
    }

    // MARK: - Helper Methods

    /// Get current memory usage in bytes
    private func currentMemoryUsage() -> mach_vm_size_t {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: count) {
                task_info(mach_task_self_,
                          task_flavor_t(MACH_TASK_BASIC_INFO),
                          $0,
                          &count)
            }
        }

        if kerr == KERN_SUCCESS {
            return info.resident_size
        } else {
            return 0
        }
    }

    /// Get current CPU time in milliseconds
    private func currentCPUTime() -> Double {
        var rusage = rusage()
        getrusage(RUSAGE_SELF, &rusage)

        let userTime = Double(rusage.ru_utime.tv_sec) * 1000.0 + Double(rusage.ru_utime.tv_usec) / 1000.0
        let systemTime = Double(rusage.ru_stime.tv_sec) * 1000.0 + Double(rusage.ru_stime.tv_usec) / 1000.0

        return userTime + systemTime
    }

    // MARK: - Test Node Implementations

    /// Simple passthrough node for baseline performance
    class PassthroughNode: AudioProcessingNode {
        func process(buffer: [Float], config: AudioProcessingConfig) throws -> [Float] {
            return buffer
        }
    }

    /// Node that performs intensive processing to stress CPU
    class IntensiveProcessingNode: AudioProcessingNode {
        func process(buffer: [Float], config: AudioProcessingConfig) throws -> [Float] {
            var output = [Float](repeating: 0.0, count: buffer.count)

            // Perform CPU-intensive operations
            for index in 0..<buffer.count {
                // Simulate complex processing with multiple math operations
                let sample = buffer[index]
                let phase = 2.0 * Float.pi * sample

                // Multiple trigonometric operations to stress CPU
                output[index] = sin(phase) * cos(phase) * tanh(phase)
                output[index] *= sqrt(abs(sample))
                output[index] += log(abs(sample) + 1.0)
            }

            return output
        }
    }

    /// Node that allocates memory during processing for memory tests
    class MemoryAllocatingNode: AudioProcessingNode {
        func process(buffer: [Float], config: AudioProcessingConfig) throws -> [Float] {
            // Allocate temporary buffers to test memory management
            let temp1 = [Float](repeating: 0.0, count: buffer.count)
            let temp2 = [Float](repeating: 0.0, count: buffer.count)

            // Perform some processing using the temporary buffers
            var output = [Float](repeating: 0.0, count: buffer.count)
            for index in 0..<buffer.count {
                output[index] = temp1[index] + temp2[index] + buffer[index]
            }

            return output
        }
    }
}
