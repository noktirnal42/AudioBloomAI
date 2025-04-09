#include <metal_stdlib>
using namespace metal;

// Audio visualization parameters struct
struct AudioVisualizerParameters {
    // Audio analysis parameters
    float bassLevel;             // Low frequency intensity
    float midLevel;              // Mid frequency intensity
    float trebleLevel;           // High frequency intensity
    float leftLevel;             // Left channel volume
    float rightLevel;            // Right channel volume

    // Theme colors
    float4 primaryColor;         // Primary theme color
    float4 secondaryColor;       // Secondary theme color
    float4 backgroundColor;      // Background color
    float4 accentColor;          // Accent color for highlights

    // Animation parameters
    float time;                  // Current time in seconds
    float sensitivity;           // Audio sensitivity (0.0-1.0)
    float motionIntensity;       // Motion intensity (0.0-1.0)
    float themeIndex;            // Current theme index (0-7 for different themes)

    // Visualization settings
    float visualizationMode;     // 0: Spectrum, 1: Waveform, 2: Particles, 3: Neural
    float previousMode;          // Previous visualization mode for transitions
    float transitionProgress;    // Progress between mode transitions (0.0-1.0)
    float colorIntensity;        // Color intensity parameter (0.0-1.0)

    // Processing parameters
};

// Example vertex shader function using the parameters
vertex float4 vertexShader(uint vertexID [[vertex_id]],
                           constant float2 *positions [[buffer(0)]],
                           constant AudioVisualizerParameters &params [[buffer(1)]]) {
    // Access parameters like params.bassLevel, params.primaryColor, etc.
    float2 position = positions[vertexID];
    
    // Example of using audio parameters to modify the vertex position
    position.y *= 1.0 + params.bassLevel * params.sensitivity;
    
    return float4(position, 0.0, 1.0);
}

// Example fragment shader function
fragment float4 fragmentShader(float2 texCoord [[stage_in]],
                               constant AudioVisualizerParameters &params [[buffer(0)]]) {
    // Base color from theme
    float4 color = params.backgroundColor;
    
    // Modify color based on audio levels and time
    float intensity = (params.bassLevel + params.midLevel + params.trebleLevel) / 3.0;
    color = mix(color, params.primaryColor, intensity * params.colorIntensity);
    
    // Add time-based effects
    float timeEffect = sin(params.time * 2.0) * 0.5 + 0.5;
    color = mix(color, params.accentColor, timeEffect * params.motionIntensity * 0.3);
    
    return color;
}
