// Structured types for AudioBloomAI
import Foundation

// MARK: - Processing Types

public struct ProcessingStats {
    public let frameCount: Int
    public let processingTime: TimeInterval
    public let errorCount: Int
    
    public init(
        frameCount: Int,
        processingTime: TimeInterval,
        errorCount: Int
    ) {
        self.frameCount = frameCount
        self.processingTime = processingTime
        self.errorCount = errorCount
    }
}

// MARK: - Buffer Types

public struct BufferStats {
    public let activeCount: Int
    public let pooledCount: Int
    
    public init(activeCount: Int, pooledCount: Int) {
        self.activeCount = activeCount
        self.pooledCount = pooledCount
    }
}

// MARK: - Test Types

public struct TestStats {
    public let successCount: Int
    public let failureCount: Int
    public let skippedCount: Int
    
    public init(
        successCount: Int,
        failureCount: Int,
        skippedCount: Int
    ) {
        self.successCount = successCount
        self.failureCount = failureCount
        self.skippedCount = skippedCount
    }
}

// MARK: - Error Types

public enum RenderError: Error {
    case invalidBuffer
    case invalidCommandQueue
    case invalidTexture
    case processingFailed(String)
}
