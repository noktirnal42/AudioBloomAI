// FFTMemoryTests.swift
import XCTest
@testable import AudioProcessor

@available(macOS 15.0, *)
final class FFTMemoryTests: XCTestCase {
    var fftNode: FFTNode!
    var pipelineCore: AudioPipelineCore!
    
    override func setUp() async throws {
        super.setUp()
        fftNode = FFTNode()
        pipelineCore = try AudioPipelineCore()
    }
    
    override func tearDown() {
        fftNode = nil
        pipelineCore = nil
        super.tearDown()
    }
    
    func testMemoryManagement() throws {
        let bufferSizes = [1024, 2048, 4096]
        var bufferIds: [AudioBufferID] = []
        
        for bufferSize in bufferSizes {
            for _ in 0..<10 {
                let bufferId = try pipelineCore.allocateBuffer(
                    size: bufferSize * MemoryLayout<Float>.stride
                )
                bufferIds.append(bufferId)
            }
        }
        
        for bufferId in bufferIds {
            pipelineCore.releaseBuffer(id: bufferId)
        }
        
        let stats = pipelineCore.bufferStats
        XCTAssertEqual(stats.activeBuffers, 0, "No active buffers expected")
        XCTAssertEqual(stats.totalBuffers, 0, "All buffers should be released")
    }
    
    func testBufferReuse() throws {
        let bufferSize = 1024
        var bufferIds = Set<AudioBufferID>()
        
        // Allocate and release buffers multiple times
        for _ in 0..<5 {
            let bufferId = try pipelineCore.allocateBuffer(
                size: bufferSize * MemoryLayout<Float>.stride
            )
            bufferIds.insert(bufferId)
            pipelineCore.releaseBuffer(id: bufferId)
        }
        
        // Verify buffer reuse
        XCTAssertLessThan(
            bufferIds.count,
            5,
            "Buffers should be reused from the pool"
        )
    }
}
