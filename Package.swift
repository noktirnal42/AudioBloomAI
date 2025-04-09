// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "AudioBloomAI",
    platforms: [
        .macOS(.v15),
        .iOS(.v17)
    ],
    products: [
        // Main framework
        .library(
            name: "AudioBloomAI",
            targets: ["AudioBloomAI"]
        ),
        // Individual components
        .library(
            name: "AudioProcessor",
            targets: ["AudioProcessor"]
        ),
        .library(
            name: "MLEngine",
            targets: ["MLEngine"]
        ),
        .library(
            name: "Visualizer",
            targets: ["Visualizer"]
        ),
        .library(
            name: "AudioBloomCore",
            targets: ["AudioBloomCore"]
        )
    ],
    dependencies: [
        // External dependencies can be added here
        // For example:
        // .package(url: "https://github.com/apple/swift-numerics", from: "1.0.0"),
    ],
    targets: [
        // Main umbrella framework
        .target(
            name: "AudioBloomAI",
            dependencies: [
                "AudioProcessor",
                "MLEngine",
                "Visualizer",
                "AudioBloomCore"
            ],
            resources: [
                .process("Resources")
            ]
        ),
        
        // Core module with shared components
        .target(
            name: "AudioBloomCore",
            dependencies: []
        ),
        
        // Audio processing module
        .target(
            name: "AudioProcessor",
            dependencies: ["AudioBloomCore"]
        ),
        
        // Machine learning module
        .target(
            name: "MLEngine",
            dependencies: ["AudioBloomCore"]
        ),
        
        // Visualization module
        .target(
            name: "Visualizer",
            dependencies: ["AudioBloomCore"]
        ),
        
        // Tests for the main framework
        .testTarget(
            name: "AudioBloomAITests",
            dependencies: ["AudioBloomAI"]
        ),
        
        // Tests for individual components
        .testTarget(
            name: "AudioProcessorTests",
            dependencies: ["AudioProcessor"]
        ),
        
        .testTarget(
            name: "MLEngineTests",
            dependencies: ["MLEngine"]
        ),
        
        .testTarget(
            name: "VisualizerTests",
            dependencies: ["Visualizer"]
        ),
        
        .testTarget(
            name: "AudioBloomCoreTests",
            dependencies: ["AudioBloomCore"]
        )
    ],
    swiftLanguageVersions: [.v5]
)

// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "AudioBloomAI",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        // Main application
        .executable(
            name: "AudioBloomAI",
            targets: ["AudioBloomApp"]
        ),
        // Libraries that can be reused
        .library(
            name: "AudioBloomCore",
            targets: ["AudioBloomCore"]
        ),
        .library(
            name: "AudioProcessor",
            targets: ["AudioProcessor"]
        ),
        .library(
            name: "MLEngine",
            targets: ["MLEngine"]
        ),
        .library(
            name: "Visualizer",
            targets: ["Visualizer"]
        ),
        .library(
            name: "AudioBloomUI",
            targets: ["AudioBloomUI"]
        )
    ],
    dependencies: [
        // External dependencies
        .package(url: "https://github.com/apple/swift-algorithms", from: "1.0.0"),
        .package(url: "https://github.com/apple/swift-numerics", from: "1.0.0"),
        .package(url: "https://github.com/apple/swift-log.git", from: "1.5.3"),
        .package(url: "https://github.com/pointfreeco/swift-composable-architecture", from: "1.5.0"),
        .package(url: "https://github.com/AudioKit/AudioKit", from: "5.6.0")
    ],
    targets: [
        // Main application target
        .executableTarget(
            name: "AudioBloomApp",
            dependencies: [
                "AudioBloomCore",
                "AudioProcessor",
                "Visualizer",
                "MLEngine",
                "AudioBloomUI"
            ],
            resources: [
                .process("Resources")
            ]
        ),
        
        // Core shared functionality
        .target(
            name: "AudioBloomCore",
            dependencies: [
                .product(name: "Algorithms", package: "swift-algorithms"),
                .product(name: "Numerics", package: "swift-numerics"),
                .product(name: "Logging", package: "swift-log")
            ]
        ),
        
        // Audio processing module
        .target(
            name: "AudioProcessor",
            dependencies: [
                "AudioBloomCore",
                .product(name: "Numerics", package: "swift-numerics"),
                .product(name: "AudioKit", package: "AudioKit")
            ],
            linkerSettings: [
                .linkedFramework("AVFoundation", .when(platforms: [.macOS])),
                .linkedFramework("CoreAudio", .when(platforms: [.macOS]))
            ]
        ),
        
        // Visualization module
        .target(
            name: "Visualizer",
            dependencies: [
                "AudioBloomCore",
                .product(name: "Algorithms", package: "swift-algorithms"),
                .product(name: "Numerics", package: "swift-numerics")
            ],
            resources: [
                .process("Resources/Shaders")
            ],
            cSettings: [
                .unsafeFlags(["-fmodules"], .when(platforms: [.macOS]))
            ],
            linkerSettings: [
                .linkedFramework("Metal", .when(platforms: [.macOS])),
                .linkedFramework("MetalKit", .when(platforms: [.macOS])),
                .linkedFramework("CoreImage", .when(platforms: [.macOS]))
            ]
        ),
        
        // Neural Engine integration module
        .target(
            name: "MLEngine",
            dependencies: [
                "AudioBloomCore",
                .product(name: "Numerics", package: "swift-numerics"),
                .product(name: "Logging", package: "swift-log")
            ],
            resources: [
                .process("Resources/Models")
            ],
            cSettings: [
                .unsafeFlags(["-fmodules"], .when(platforms: [.macOS]))
            ],
            linkerSettings: [
                .linkedFramework("CoreML", .when(platforms: [.macOS])),
                .linkedFramework("Accelerate", .when(platforms: [.macOS]))
            ],
            swiftSettings: [
                .define("ENABLE_NEURAL_ENGINE")
            ]
        ),
        
        // UI Components module
        .target(
            name: "AudioBloomUI",
            dependencies: [
                "AudioBloomCore",
                "MLEngine",
                .product(name: "ComposableArchitecture", package: "swift-composable-architecture")
            ]
        ),
        
        // Test targets
        .testTarget(
            name: "AudioBloomTests",
            dependencies: ["AudioBloomCore"],
            path: "Tests/AudioBloomTests"
        ),
        .testTarget(
            name: "AudioProcessorTests",
            dependencies: ["AudioProcessor"],
            path: "Tests/AudioProcessorTests"
        ),
        .testTarget(
            name: "VisualizerTests",
            dependencies: ["Visualizer"],
            path: "Tests/VisualizerTests"
        ),
        .testTarget(
            name: "MLEngineTests",
            dependencies: ["MLEngine"],
            path: "Tests/MLEngineTests"
        ),
        .testTarget(
            name: "AudioBloomUITests",
            dependencies: ["AudioBloomUI"],
            path: "Tests/AudioBloomUITests"
        )
    ]
)
