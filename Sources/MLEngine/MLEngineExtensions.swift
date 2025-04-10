// Updated MLEngine extensions
import Foundation

extension NeuralEngine {
    func updateFrameCount() {
        frameCount += 1
    }
    
    func updateBufferCount(by amount: Int) {
        bufferCount += amount
    }
}

extension AudioBridge {
    func incrementCount() {
        count += 1
    }
}

extension ModelConfiguration {
    func getModel() throws -> Model {
        guard let model = try? configuration.model else {
            throw ConfigError.invalidModel
        }
        return model
    }
}
