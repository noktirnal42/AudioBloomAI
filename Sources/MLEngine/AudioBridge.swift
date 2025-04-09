import Foundation
import AVFoundation
import Combine
import AudioBloomCore

/// Bridge connecting AudioEngine and MLProcessor
public class AudioBridge: ObservableObject {
    /// The ML processor for audio analysis
    private let mlProcessor: MLProcessor
    
    /// The audio engine providing audio data
    private var audioProvider: AudioDataProvider?
    
    /// Subscription to audio data updates
    private var audioDataSubscription: AnyCancellable?
    
    /// Format converter for efficient audio processing
    private let formatConverter: FormatConverter
    
    /// Audio processing queue
    private let processingQueue = DispatchQueue(label: "com.audiobloom.audiobridge", qos: .userInteractive)
    
    /// Whether the bridge is currently active
    private var isActive = false
    
    /// Subscription to visualization data for forwarding
    private var visualizationSubscription: AnyCancellable?
    
    /// Publisher for visualization data
    private let visualizationForwarder = PassthroughSubject<VisualizationData, Never>()
    
    /// Publisher for visualization data
    public var visualizationPublisher: AnyPublisher<VisualizationData, Never> {
        visualizationForwarder.eraseToAnyPublisher()
    }
    
    /// Initializes the audio bridge with the specified ML processor
    /// - Parameter mlProcessor: The ML processor to use for audio analysis
    public init(mlProcessor: MLProcessor) {
        self.mlProcessor = mlProcessor
        self.formatConverter = FormatConverter()
        
        // Subscribe to visualization data from the ML processor
        visualizationSubscription = mlProcessor.visualizationDataPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] data in
                self?.visualizationForwarder.send(data)
            }
    }
    
    /// Connects to the specified audio provider
    /// - Parameter provider: The audio data provider
    public func connect(to provider: AudioDataProvider) {
        // Disconnect from any existing provider
        disconnect()
        
        // Store new provider
        audioProvider = provider
        
        // Subscribe to audio data updates
        audioDataSubscription = provider.audioDataPublisher
            .receive(on: processingQueue)
            .sink { [weak self] audioData in
                self?.processAudioData(audioData)
            }
        
        // Prepare ML processor if needed
        if !mlProcessor.isReady {
            prepareMLProcessor()
        }
    }
    
    /// Disconnects from the current audio provider
    public func disconnect() {
        audioDataSubscription?.cancel()
        audioDataSubscription = nil
        audioProvider = nil
    }
    
    /// Activates the bridge to start ML processing
    public func activate() {
        guard !isActive else { return }
        
        isActive = true
        
        // Start continuous processing if ready
        if mlProcessor.isReady {
            do {
                try mlProcessor.startContinuousProcessing()
            } catch {
                print("Failed to start ML processing: \(error)")
            }
        }
    }
    
    /// Deactivates the bridge to stop ML processing
    public func deactivate() {
        guard isActive else { return }
        
        isActive = false
        mlProcessor.stopContinuousProcessing()
    }
    
    /// Prepares the ML processor
    private func prepareMLProcessor() {
        Task {
            do {
                // Create a standard audio format for ML processing
                let format = AVAudioFormat(
                    standardFormatWithSampleRate: 44100,
                    channels: 1
                )
                
                // Prepare the ML processor
                try await mlProcessor.prepareMLModel(with: format)
                
                // Start processing if already active
                if isActive {
                    try mlProcessor.startContinuousProcessing()
                }
            } catch {
                print("Failed to prepare ML processor: \(error)")
            }
        }
    }
    
    /// Processes audio data received from the audio provider
    /// - Parameter audioData: The audio data to process
    private func processAudioData(_ audioData: AudioData) {
        guard isActive, mlProcessor.isReady else { return }
        
        // Convert frequency data to a format suitable for ML processing
        let processableData = formatConverter.convertForProcessing(
            frequencyData: audioData.frequencyData,
            levels: audioData.levels
        )
        
        // Process the data through the ML processor
        Task {
            do {
                try await mlProcessor.processAudioData(processableData)
            } catch {
                // Silently handle errors during normal operation to avoid flooding logs
                if case MLProcessorError.processingFailed = error {
                    // Only log serious errors
                    print("ML processing error: \(error)")
                }
            }
        }
    }
}

/// Utility class for converting between audio data formats
private class FormatConverter {
    /// Converts frequency data to a format suitable for ML processing
    /// - Parameters:
    ///   - frequencyData: The frequency spectrum data
    ///   - levels: The audio level data
    /// - Returns: Array of float values for ML processing
    func convertForProcessing(frequencyData: [Float], levels: (left: Float, right: Float)) -> [Float] {
        // For ML processing, we need to ensure a consistent format
        // Here we combine the frequency data with level information
        
        // Start with the frequency data
        var result = frequencyData
        
        // If the frequency data is empty, create a placeholder
        if result.isEmpty {
            result = [Float](repeating: 0, count: 512) // Use a standard size
        }
        
        // Pad to ensure consistent length if needed
        if result.count < 512 {
            result = result + [Float](repeating: 0, count: 512 - result.count)
        } else if result.count > 512 {
            result = Array(result[0..<512])
        }
        
        // Add level information at the end
        // This allows the ML model to consider overall amplitude
        result.append(levels.left)
        result.append(levels.right)
        
        return result
    }
    
    /// Creates an audio buffer from raw audio data
    /// - Parameters:
    ///   - data: The audio sample data
    ///   - format: The audio format
    /// - Returns: An AVAudioPCMBuffer containing the data
    func createBuffer(from data: [Float], format: AVAudioFormat) -> AVAudioPCMBuffer? {
        // Create a new buffer with the appropriate capacity
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(data.count)) else {
            return nil
        }
        
        // Set frame length
        buffer.frameLength = AVAudioFrameCount(data.count)
        
        // Copy data to buffer
        if let channelData = buffer.floatChannelData {
            for i in 0..<min(data.count, Int(buffer.frameLength)) {
                channelData[0][i] = data[i]
            }
        }
        
        return buffer
    }
}

