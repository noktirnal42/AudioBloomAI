// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.
// Optimized for macOS 15+ and Xcode 16+

import PackageDescription

let package = Package(
    name: "AudioBloomAI",
    platforms: [
        .macOS(.v15) // Specific requirement for macOS 15+
    ],
    products: [
        // Main executable application
        .executable(
            name: "AudioBloomApp",
            targets: ["AudioBloomApp"]
        ),
        .library(
            name: "AudioBloomCore",
            targets: ["AudioBloomCore"]
        ),
        .library(
            name: "AudioProcessor",
            targets: ["AudioProcessor"]
        ),
        .library(
            name: "Visualizer", 
            targets: ["Visualizer"]
        ),
        .library(
            name: "MLEngine", 
            targets: ["MLEngine"]
        ),
        .library(
            name: "AudioBloomUI", 
            targets: ["AudioBloomUI"]
        )
    ],
    dependencies: [
        // External dependencies
        .package(url: "https://github.com/apple/swift-algorithms", from: "1.2.1"),
        .package(url: "https://github.com/apple/swift-numerics", from: "1.0.3"),
        .package(url: "https://github.com/apple/swift-log.git", from: "1.6.3"),
        .package(url: "https://github.com/pointfreeco/swift-composable-architecture", from: "1.19.0"),
        .package(url: "https://github.com/AudioKit/AudioKit", from: "5.6.5")
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
            exclude: ["README.md", "SupportFiles/Info.plist"],
            resources: [
                .copy("Resources")
            ]
        ),
        
        // Core shared functionality
        .target(
            name: "AudioBloomCore",
            dependencies: [
                .product(name: "Algorithms", package: "swift-algorithms"),
                .product(name: "Numerics", package: "swift-numerics"),
                .product(name: "Logging", package: "swift-log")
            ],
            exclude: ["README.md"],
            resources: [
                .process("Resources")
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
            exclude: ["README.md"],
            resources: [
                .process("Resources")
            ],
            swiftSettings: [
                // Any swift settings would go here
            ],
            linkerSettings: [
                .linkedFramework("AVFoundation", .when(platforms: [.macOS])),
                .linkedFramework("CoreAudio", .when(platforms: [.macOS])),
                .linkedFramework("Accelerate", .when(platforms: [.macOS]))
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
            exclude: ["README.md"],
            resources: [
                .copy("Resources/Shaders")
            ],
            cSettings: [
                .unsafeFlags(["-fmodules"], .when(platforms: [.macOS]))
            ],
            swiftSettings: [
                // Any swift settings would go here
            ],
            linkerSettings: [
                .linkedFramework("Metal", .when(platforms: [.macOS])),
                .linkedFramework("MetalKit", .when(platforms: [.macOS])),
                .linkedFramework("CoreImage", .when(platforms: [.macOS])),
                .linkedFramework("QuartzCore", .when(platforms: [.macOS]))
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
            exclude: ["README.md"],
            resources: [
                .process("Resources")
            ],
            cSettings: [
                .unsafeFlags(["-fmodules"], .when(platforms: [.macOS]))
            ],
            swiftSettings: [
                .define("ENABLE_NEURAL_ENGINE")
            ],
            linkerSettings: [
                .linkedFramework("CoreML", .when(platforms: [.macOS])),
                .linkedFramework("Accelerate", .when(platforms: [.macOS])),
                .linkedFramework("CreateML", .when(platforms: [.macOS])),
                .linkedFramework("SoundAnalysis", .when(platforms: [.macOS]))
            ]
        ),
        
        // UI Components module
        .target(
            name: "AudioBloomUI",
            dependencies: [
                "AudioBloomCore",
                "MLEngine",
                "Visualizer",
                .product(name: "ComposableArchitecture", package: "swift-composable-architecture")
            ],
            exclude: ["README.md"],
            resources: [
                .process("Resources/Assets"),
                .copy("Views/Resources")
            ],
            swiftSettings: [
                // Any swift settings would go here
            ],
            linkerSettings: [
                .linkedFramework("SwiftUI", .when(platforms: [.macOS])),
                .linkedFramework("Combine", .when(platforms: [.macOS]))
            ]
        ),
        
        // Test targets
        .testTarget(
            name: "AudioBloomTests",
            dependencies: [
                "AudioBloomCore",
                "AudioProcessor",
                "Visualizer",
                "MLEngine",
                "AudioBloomUI"
            ],
            resources: [
                .copy("Resources"),
                .copy("TestResources")
            ]
        ),
        .testTarget(
            name: "AudioProcessorTests",
            dependencies: ["AudioProcessor"],
            resources: [
                .copy("Resources")
            ]
        ),
        .testTarget(
            name: "VisualizerTests",
            dependencies: ["Visualizer"],
            resources: [
                .copy("Resources")
            ]
        ),
        .testTarget(
            name: "MLEngineTests",
            dependencies: ["MLEngine"],
            resources: [
                .copy("Resources"),
                .copy("TestResources")
            ]
        ),
        .testTarget(
            name: "AudioBloomUITests",
            dependencies: ["AudioBloomUI"],
            resources: [
                .copy("Resources")
            ]
        )
    ],
    swiftLanguageVersions: [.v6]
)
