// MetalMemoryTests.swift
import XCTest
@testable import AudioProcessor

@available(macOS 15.0, *)
final class MetalMemoryTests: XCTestCase {
    var metalCore: MetalComputeCore!
    
    override func setUp() {
        super.setUp()
        metalCore = try! MetalComputeCore()
    }
    
    override func tearDown() {
        metalCore = nil
        super.tearDown()
    }
    
    func testMemoryManagement() throws {
        let bufferCount = 100
        var bufferIds: [UInt64] = []
        
        // Allocate buffers of different sizes
        for bufferSize in [1024, 2048, 4096] {
            for _ in 0..<bufferCount {
                let (bufferId, _) = try metalCore.allocateBuffer(
                    length: bufferSize * MemoryLayout<Float>.stride,
                    options: .storageModeShared
                )
                bufferIds.append(bufferId)
            }
        }
        
        // Release all buffers
        for bufferId in bufferIds {
            metalCore.releaseBuffer(id: bufferId)
        }
        
        let statsBefore = metalCore.getBufferStats()
        XCTAssertEqual(statsBefore.activeCount, 0, "No active buffers expected")
        XCTAssertGreaterThan(statsBefore.pooledCount, 0, "Buffer pool should not be empty")
        
        metalCore.cleanupUnusedBuffers()
        
        let statsAfter = metalCore.getBufferStats()
        XCTAssertEqual(statsAfter.activeCount, 0, "No active buffers expected")
        XCTAssertEqual(statsAfter.pooledCount, 0, "Buffer pool should be empty")
    }
}
