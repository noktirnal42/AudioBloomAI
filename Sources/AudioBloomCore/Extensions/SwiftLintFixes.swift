// Fix SwiftLint issues in source files
import Foundation

// MARK: - Neural Engine Extensions

extension NeuralEngine {
    func updateBufferCount(increment: Int) {
        bufferCount += increment
    }
    
    func updateConfiguration(updates: ModelConfiguration) {
        modelConfiguration += updates
    }
}

// MARK: - Audio Bridge Extensions

extension AudioBridge {
    func incrementFrameCount() {
        frameCount += 1
    }
}

// MARK: - Model Configuration Extensions

extension ModelConfiguration {
    func getModel() throws -> Model {
        guard let model = try? configuration.model else {
            throw ConfigurationError.invalidModel
        }
        return model
    }
}

// MARK: - Metal Compute Extensions

extension MetalComputeCore {
    struct BufferStats {
        let activeCount: Int
        let pooledCount: Int
    }
    
    func getBufferStats() -> BufferStats {
        return BufferStats(
            activeCount: activeBuffers.count,
            pooledCount: pooledBuffers.count
        )
    }
}

// MARK: - Preset Controls Extensions

extension PresetControlsView {
    var isSelectedExport: Bool {
        get { isSelected_export }
        set { isSelected_export = newValue }
    }
}

// MARK: - Error Types

enum ConfigurationError: Error {
    case invalidModel
}
