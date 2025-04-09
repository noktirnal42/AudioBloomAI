// swift-tools-version: 5.9
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
        // Libraries that can be reused
        .library(
            name: "AudioBloomAICore",
            targets: ["AudioBloomCore"]
    ],
    dependencies: [
        // External dependencies can be added here as needed
        .package(url: "https://github.com/apple/swift-algorithms", from: "1.0.0"),
        .package(url: "https://github.com/apple/swift-numerics", from: "1.0.0"),
        .package(url: "https://github.com/apple/swift-log.git", from: "1.5.3"),
    ],
    targets: [
        // Main application target
        .executableTarget(
            name: "AudioBloomApp",
            dependencies: [
                "AudioBloomCore",
                "AudioProcessor",
                "Visualizer",
                "MLEngine"
            ],
            sources: ["Sources/AudioBloomApp/AudioBloomApp.swift", "Sources/AudioBloomApp/ContentView.swift"],
            resources: [
                .process("Resources")
            ]
        ),
        
        // Core shared functionality
        .target(
            name: "AudioBloomCore",
            dependencies: []
        ),
        
        // Audio processing module
        .target(
            name: "AudioProcessor",
            dependencies: ["AudioBloomCore"]
        ),
        
        .target(
            name: "Visualizer",
            dependencies: [
                "AudioBloomCore",
                .product(name: "Algorithms", package: "swift-algorithms"),
                .product(name: "Numerics", package: "swift-numerics")
            ],
            resources: [
                .process("Resources/Shaders")
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
            swiftSettings: [
                .unsafeFlags(["-Xfrontend", "-enable-experimental-cxx-interop"]),
                .define("ENABLE_NEURAL_ENGINE", .when(platforms: [.macOS]))
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
        )
    ]
)
