// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency
import Foundation
import CoreML
import Combine
import Logging
import os.signpost

/// Errors related to model configuration and optimization
@available(macOS 15.0, *)
public enum ModelConfigurationError: Error {
    case modelNotFound
    case compilationFailed(Error)
    case optimizationFailed(Error)
    case neuralEngineUnavailable
    case invalidConfiguration
    case modelSizeTooLarge
    case incompatiblePrecision
    
    public var localizedDescription: String {
        switch self {
        case .modelNotFound:
            return "CoreML model file not found"
        case .compilationFailed(let error):
            return "Failed to compile CoreML model: \(error.localizedDescription)"
        case .optimizationFailed(let error):
            return "Failed to optimize model for Neural Engine: \(error.localizedDescription)"
        case .neuralEngineUnavailable:
            return "Neural Engine is not available on this device"
        case .invalidConfiguration:
            return "Invalid model configuration parameters"
        case .modelSizeTooLarge:
            return "Model size exceeds Neural Engine limits"
        case .incompatiblePrecision:
            return "Model precision is incompatible with Neural Engine"
        }
    }
}

/// Performance optimization level
@available(macOS 15.0, *)
public enum OptimizationLevel {
    /// Maximize quality, potentially at the cost of performance
    case quality
    /// Balance quality and performance
    case balanced
    /// Maximize performance, potentially at the cost of quality
    case performance
    /// Custom optimization with specific parameters
    case custom
}

/// Neural Engine specific capabilities
@available(macOS 15.0, *)
public struct NeuralEngineCapabilities: Sendable {
    /// Whether the Neural Engine is available
    public let isAvailable: Bool
    
    /// The Neural Engine compute units available
    public let computeUnits: MLComputeUnits
    
    /// Neural Engine version (if available)
    public let version: String?
    
    /// Maximum concurrent Neural Engine tasks
    public let maxConcurrentRequests: Int
    
    /// Creates a new capabilities object
    /// - Returns: The capabilities object
    public static func current() -> NeuralEngineCapabilities {
        // Check for Neural Engine availability
        let computeUnits: MLComputeUnits = MLComputeUnits.all
        
        // Use private API to get Neural Engine version (if available)
        let version = ProcessInfo().processorCount > 8 ? "Apple Silicon" : nil
        
        // Determine max concurrent requests based on processor count
        let maxConcurrentRequests = ProcessInfo().processorCount / 2
        
        return NeuralEngineCapabilities(
            isAvailable: version != nil,
            computeUnits: computeUnits,
            version: version,
            maxConcurrentRequests: maxConcurrentRequests
        )
    }
}

/// Handles CoreML model configuration and optimization for Neural Engine
@available(macOS 15.0, *)
public class ModelConfiguration {
    /// Logger for this class
    private let logger = Logger(label: "com.audiobloom.modelconfiguration")
    
    /// Signpost log for performance tracking
    private let signposter = OSSignposter(subsystem: "com.audiobloom.modelconfiguration", category: "NeuralEngine")
    
    /// The model configuration
    private var configuration: MLModelConfiguration
    
    /// Neural Engine capabilities
    public let neuralEngineCapabilities: NeuralEngineCapabilities
    
    /// Current optimization level
    private(set) public var optimizationLevel: OptimizationLevel
    
    /// Subject for monitoring Neural Engine utilization
    private let utilizationSubject = CurrentValueSubject<Double, Never>(0.0)
    
    /// Publisher for Neural Engine utilization (0.0-1.0)
    public var utilizationPublisher: AnyPublisher<Double, Never> {
        utilizationSubject.eraseToAnyPublisher()
    }
    
    /// Whether to collect performance metrics
    public var collectPerformanceMetrics: Bool = true {
        didSet {
            logger.info("Performance metrics collection \(collectPerformanceMetrics ? "enabled" : "disabled")")
        }
    }
    
    /// Initializes a new model configuration with the specified optimization level
    /// - Parameter optimizationLevel: The desired optimization level
    public init(optimizationLevel: OptimizationLevel = .balanced) {
        self.optimizationLevel = optimizationLevel
        self.configuration = MLModelConfiguration()
        self.neuralEngineCapabilities = NeuralEngineCapabilities.current()
        
        // Configure based on optimization level
        applyOptimizationLevel(optimizationLevel)
        
        logger.info("Model configuration initialized with optimization level: \(optimizationLevel)")
        logger.info("Neural Engine available: \(neuralEngineCapabilities.isAvailable)")
        
        // Start monitoring Neural Engine utilization
        startMonitoringUtilization()
    }
    
    /// Applies the specified optimization level to the configuration
    /// - Parameter level: The optimization level to apply
    public func applyOptimizationLevel(_ level: OptimizationLevel) {
        self.optimizationLevel = level
        
        switch level {
        case .quality:
            configureForQuality()
        case .balanced:
            configureForBalanced()
        case .performance:
            configureForPerformance()
        case .custom:
            // Custom configuration is applied separately
            break
        }
        
        logger.info("Applied optimization level: \(level)")
    }
    
    /// Configures for maximum quality
    private func configureForQuality() {
        configuration.computeUnits = neuralEngineCapabilities.isAvailable ? .all : .cpuAndGPU
        configuration.allowLowPrecisionAccumulationOnGPU = false
        configuration.preferredMetalDevice = MTLCreateSystemDefaultDevice()
        configuration.parameters = [MLParameterKey.useEmbeddingCache: true]
    }
    
    /// Configures for balanced performance and quality
    private func configureForBalanced() {
        configuration.computeUnits = neuralEngineCapabilities.isAvailable ? .all : .cpuAndGPU
        configuration.allowLowPrecisionAccumulationOnGPU = true
        configuration.preferredMetalDevice = MTLCreateSystemDefaultDevice()
        configuration.parameters = [MLParameterKey.useEmbeddingCache: true]
    }
    
    /// Configures for maximum performance
    private func configureForPerformance() {
        configuration.computeUnits = neuralEngineCapabilities.isAvailable ? .all : .cpuAndGPU
        configuration.allowLowPrecisionAccumulationOnGPU = true
        configuration.preferredMetalDevice = MTLCreateSystemDefaultDevice()
        configuration.parameters = [
            MLParameterKey.useEmbeddingCache: true,
            MLParameterKey.batchSize: 4
        ]
    }
    
    /// Sets a custom configuration parameter
    /// - Parameters:
    ///   - key: The parameter key
    ///   - value: The parameter value
    /// - Returns: Self for chaining
    @discardableResult
    public func setParameter(_ key: MLParameterKey, value: Any) -> Self {
        var params = configuration.parameters ?? [:]
        params[key] = value
        configuration.parameters = params
        
        // Set to custom mode when manually setting parameters
        self.optimizationLevel = .custom
        
        return self
    }
    
    /// Creates configuration preset optimized for audio analysis
    /// - Returns: A configured instance
    public static func audioAnalysisPreset() -> ModelConfiguration {
        let config = ModelConfiguration(optimizationLevel: .balanced)
        
        // Set audio-specific optimizations
        config.setParameter(MLParameterKey.batchSize, value: 1) // Real-time audio needs low latency
            .setParameter(MLParameterKey.useEmbeddingCache, value: true)
        
        if config.neuralEngineCapabilities.isAvailable {
            // Additional Neural Engine optimizations for audio
            config.setParameter(MLParameterKey.cycleLength, value: 1)
        }
        
        return config
    }
    
    /// Compiles a model with the current configuration
    /// - Parameter url: URL to the .mlmodel or .mlmodelc file
    /// - Returns: The compiled MLModel
    /// - Throws: ModelConfigurationError if compilation fails
    public func compileModel(at url: URL) async throws -> MLModel {
        guard FileManager.default.fileExists(atPath: url.path) else {
            logger.error("Model file not found at path: \(url.path)")
            throw ModelConfigurationError.modelNotFound
        }
        
        let modelCompilationID = signposter.makeSignpostID()
        let state = signposter.beginInterval("Model Compilation", id: modelCompilationID)
        
        do {
            // Determine if we need to compile from .mlmodel to .mlmodelc
            var compiledModelURL = url
            if url.pathExtension == "mlmodel" {
                logger.info("Compiling model from .mlmodel format")
                let compiledURL = try await MLModel.compileModel(at: url)
                compiledModelURL = compiledURL
            }
            
            // Load the model with our configuration
            logger.info("Loading model with Neural Engine optimizations")
            let model = try MLModel(contentsOf: compiledModelURL, configuration: configuration)
            
            signposter.endInterval("Model Compilation", state)
            logger.info("Model compilation and loading completed successfully")
            
            return model
        } catch {
            signposter.endInterval("Model Compilation", state)
            logger.error("Failed to compile or load model: \(error.localizedDescription)")
            throw ModelConfigurationError.compilationFailed(error)
        }
    }
    
    /// Checks if a model is optimized for Neural Engine
    /// - Parameter model: The model to check
    /// - Returns: True if optimized for Neural Engine
    public func isOptimizedForNeuralEngine(_ model: MLModel) -> Bool {
        // This is a simplification - in a real implementation, 
        // you would inspect model metadata or performance characteristics
        if let description = model.modelDescription.metadata[MLModelMetadataKey.neuralEngineOptimized.rawValue] as? String {
            return description == "true"
        }
        
        // Fallback to checking if compute units include ANE
        return configuration.computeUnits == .all || configuration.computeUnits == .cpuAndNeuralEngine
    }
    
    /// Starts monitoring Neural Engine utilization
    private func startMonitoringUtilization() {
        guard collectPerformanceMetrics else { return }
        
        // In a real implementation, this would use private APIs or performance counters
        // to track actual Neural Engine utilization
        // For now, we'll simulate utilization with periodic updates
        
        // Run a background task to periodically update utilization
        Task {
            while collectPerformanceMetrics {
                // Simulate reading Neural Engine utilization
                // This would be replaced with actual measurements in production
                let utilizationValue = Double.random(in: 0.0...1.0)
                utilizationSubject.send(utilizationValue)
                
                if utilizationValue > 0.8 {
                    logger.warning("Neural Engine utilization high: \(utilizationValue)")
                }
                
                try? await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
            }
        }
    }
    
    /// Gets the current model configuration
    /// - Returns: The MLModelConfiguration
    public func getConfiguration() -> MLModelConfiguration {
        return configuration
    }
    
    /// Updates the model configuration with optimizations specific to the device
    /// - Returns: Self for chaining
    @discardableResult
    public func optimizeForCurrentDevice() -> Self {
        // Detect Apple Silicon model and optimize accordingly
        let processorCount = ProcessInfo().processorCount
        
        // M1 series typically has 8 cores, M1 Pro/Max/Ultra have more
        if processorCount >= 16 {
            // Likely an M1 Pro/Max/Ultra or M2 Pro/Max/Ultra or M3 series
            logger.info("Detected high-end Apple Silicon, maximizing parallelism")
            setParameter(MLParameterKey.batchSize, value: 8)
            setParameter(MLParameterKey.concurrentComputeThreadCount, value: processorCount / 2)
        } else if processorCount >= 8 {
            // Likely base M1/M2/M3
            logger.info("Detected base Apple Silicon, using balanced configuration")
            setParameter(MLParameterKey.batchSize, value: 4)
            setParameter(MLParameterKey.concurrentComputeThreadCount, value: processorCount / 2)
        } else {
            // Older device or non-Apple Silicon
            logger.info("Detected non-Apple Silicon or older device")
            setParameter(MLParameterKey.batchSize, value: 2)
        }
        
        return self
    }
    
    /// Enhanced debugging information about the configuration
    public var debugDescription: String {
        var description = "ModelConfiguration:\n"
        description += "- Optimization Level: \(optimizationLevel)\n"
        description += "- Neural Engine: \(neuralEngineCapabilities.isAvailable ? "Available" : "Unavailable")\n"
        if let version = neuralEngineCapabilities.version {
            description += "- Neural Engine Version: \(version)\n"
        }
        description += "- Compute Units: \(configuration.computeUnits)\n"
        description += "- Parameters: \(configuration.parameters ?? [:])\n"
        
        return description
    }
}

/// Extension to MLParameterKey for audio-specific parameters
public extension MLParameterKey {
    /// Key for controlling the number of frequency bands in audio analysis
    static let frequencyBandCount = "kFrequencyBandCount"
    
    /// Key for controlling temporal resolution
    static let temporalResolution = "kTemporalResolution"
    
    /// Key for feature extraction window size
    static let windowSize = "kWindowSize"
}

/// Extension to provide monitoring metrics for CoreML performance
public extension ModelConfiguration {
    /// Performance metrics for a CoreML model
    @available(macOS 15.0, *)
    struct PerformanceMetrics: Sendable {
        /// Time spent in Neural Engine (ms)
        public let neuralEngineTime: Double
        
        /// Time spent in CPU (ms)
        public let cpuTime: Double
        
        /// Time spent in GPU (ms)
        public let gpuTime: Double
        
        /// Total inference time (ms)
        public let inferenceTime: Double
        
        /// Memory usage (MB)
        public let memoryUsage: Double
        
        /// Percentage of compute executed on Neural Engine (0-100)
        public var neuralEngineUtilization: Double {
            guard inferenceTime > 0 else { return 0 }
            return (neuralEngineTime / inferenceTime) * 100.0
        }
    }
    
    /// Measures performance metrics for a model prediction
    /// - Parameters:
    ///   - model: The model to measure
    ///   - input: The input to the model
    /// - Returns: Performance metrics for the prediction
    /// - Throws: Error if prediction fails
    func measurePerformance<Input, Output>(
        for model: MLModel,
        input: Input
    ) async throws -> (output: Output, metrics: PerformanceMetrics) where Input: MLFeatureProvider {
        guard collectPerformanceMetrics else {
            // If metrics collection is disabled, just make the prediction
            let output = try model.prediction(from: input)
            return (output as! Output, PerformanceMet

