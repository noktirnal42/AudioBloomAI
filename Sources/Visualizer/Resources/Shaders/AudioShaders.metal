// AudioShaders.metal
// Enhanced visualization effects for AudioBloomAI
//

#include <metal_stdlib>
using namespace metal;

// Import shared structures for compatibility with AudioVisualizer.metal
#import "AudioVisualizerShared.h"

//-----------------------------------------------------------
// Enhanced Spectrum Visualization Effects
//-----------------------------------------------------------

/// Structure to hold spectrum peak data
struct SpectrumPeaks {
    float peakValues[128];       // Peak values
    float peakDecay[128];        // Decay rates
    float peakHold[128];         // Hold timers
};

/// Compute kernel for spectrum peak tracking
kernel void compute_spectrum_peaks(
    const device float *audioData [[buffer(0)]],
    device SpectrumPeaks *peaks [[buffer(1)]],
    constant float &deltaTime [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    // Only process if within valid range
    if (id >= 128) {
        return;
    }
    
    // Map thread ID to audio data index with proper scaling
    uint audioIndex = min(id * 8, 1023u);
    float currentValue = audioData[audioIndex];
    
    // Get current peak value
    float currentPeak = peaks->peakValues[id];
    
    // Check if new peak detected
    if (currentValue > currentPeak) {
        // New peak found, update value and reset hold timer
        peaks->peakValues[id] = currentValue;
        peaks->peakHold[id] = 0.5; // Hold for 0.5 seconds
    } else {
        // Update hold timer
        peaks->peakHold[id] = max(0.0f, peaks->peakHold[id] - deltaTime);
        
        // If hold timer expired, start decay
        if (peaks->peakHold[id] <= 0) {
            // Apply peak decay
            float decayRate = 0.5 * deltaTime; // Adjust for smoother decay
            peaks->peakValues[id] = max(currentValue, currentPeak - decayRate);
        }
    }
    
    // Calculate decay rate based on value (higher peaks decay faster)
    peaks->peakDecay[id] = 0.1 + currentPeak * 0.2;
}

/// Apply 3D perspective effect to spectrum
float2 spectrum_3d_effect(float2 position, float2 texCoord, constant AudioUniforms &uniforms) {
    // Create 3D perspective effect
    float depth = 0.2 * uniforms.motionIntensity;
    
    // Calculate center offset
    float2 center = float2(0.5, 0.5);
    float2 fromCenter = texCoord - center;
    
    // Apply perspective distortion
    float zFactor = 1.0 - fromCenter.y * depth;
    float xOffset = fromCenter.x / zFactor;
    
    // Apply audio-reactive scaling
    float audioReactivity = uniforms.bassLevel * 0.3 * sin(uniforms.time * 2.0);
    xOffset *= 1.0 + audioReactivity;
    
    // Calculate new position
    float2 newPos = position;
    newPos.x = center.x + xOffset;
    
    // Apply subtle wave effect based on audio
    float freqIndex = int(texCoord.x * 32) % 1024;
    float audioValue = uniforms.audioData[freqIndex] * uniforms.sensitivity;
    newPos.y += sin(texCoord.x * 10.0 + uniforms.time) * 0.02 * audioValue;
    
    return newPos;
}

/// Enhanced spectrum fragment effect with peak highlighting
float4 enhanced_spectrum_fragment(float2 uv, float4 baseColor, constant AudioUniforms &uniforms, device SpectrumPeaks *peaks) {
    float4 color = baseColor;
    
    // Get frequency data
    int binIndex = int(uv.x * 128);
    int freqIndex = binIndex * 8;
    float freqValue = uniforms.audioData[freqIndex] * uniforms.sensitivity;
    
    // Get peak value
    float peakValue = 0.0;
    if (peaks != nullptr && binIndex < 128) {
        peakValue = peaks->peakValues[binIndex];
    }
    
    // Create 3D bar effect with depth shading
    float barWidth = 0.007; // Narrower bars for more detail
    float barX = fract(uv.x / barWidth);
    float barIntensity = smoothstep(0.1, 0.0, abs(barX - 0.5)); // Centered bar with smooth edges
    
    // Bar height based on audio data
    float barHeight = freqValue * 0.8;
    float barY = step(1.0 - barHeight, uv.y);
    
    // Peak indicator line
    float peakHeight = peakValue * 0.8;
    float peakLine = smoothstep(0.01, 0.0, abs(uv.y - (1.0 - peakHeight)));
    
    // Create depth shading effect (darker toward the "back")
    float depthShading = mix(0.6, 1.0, barX); // Lighting from the right
    
    // Frequency-based coloring with enhanced gradient
    float hue = mix(0.6, 0.0, uv.x); // Blue to red spectrum
    float sat = 0.8 + freqValue * 0.2; // Increase saturation with amplitude
    float val = 0.7 + freqValue * 0.3; // Brighter for higher amplitudes
    
    float3 barColor = hsv2rgb(float3(hue, sat, val)) * depthShading;
    
    // Apply peak highlighting with glow effect
    float3 peakColor = hsv2rgb(float3(hue - 0.1, 1.0, 1.0)); // Slightly shifted hue for peaks
    float peakGlow = smoothstep(0.05, 0.0, abs(uv.y - (1.0 - peakHeight))) * 0.7;
    
    // Combine bar and peak effects
    color.rgb = mix(color.rgb, barColor, barIntensity * barY);
    color.rgb += peakColor * peakLine * (0.3 + 0.2 * sin(uniforms.time * 10.0)); // Pulsing peak
    color.rgb += peakColor * peakGlow; // Add glow
    
    // Add subtle grid in the background
    float gridIntensity = 0.03 * (1.0 - barY); // Only visible where there's no bar
    float gridX = step(0.98, fract(uv.x * 16.0));
    float gridY = step(0.98, fract(uv.y * 8.0));
    color.rgb += uniforms.accentColor.rgb * (gridX + gridY) * gridIntensity;
    
    return color;
}

//-----------------------------------------------------------
// Enhanced Waveform Visualization Effects
//-----------------------------------------------------------

/// Structure to hold waveform history for persistence effect
struct WaveformHistory {
    float waveData[8][1024];  // Circular buffer of 8 frames, 1024 points each
    int currentFrame;         // Current frame index
    float opacity[8];         // Opacity for each historical frame
};

/// Compute kernel for waveform history processing
kernel void compute_waveform_history(
    const device float *audioData [[buffer(0)]],
    device WaveformHistory *history [[buffer(1)]],
    constant float &deltaTime [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    // Only process within valid range
    if (id >= 1024) {
        return;
    }
    
    // Update current frame index
    uint currentFrame = history->currentFrame;
    
    // Store current audio data in history
    history->waveData[currentFrame][id] = audioData[id];
    
    // Update opacities (fade out older frames)
    for (int i = 0; i < 8; i++) {
        if (i == currentFrame) {
            history->opacity[i] = 1.0; // Current frame is fully opaque
        } else {
            history->opacity[i] = max(0.0f, history->opacity[i] - deltaTime * 0.5);
        }
    }
    
    // If this is thread 0, advance current frame index (once per kernel dispatch)
    if (id == 0) {
        history->currentFrame = (currentFrame + 1) % 8;
    }
}

/// Enhanced waveform visualization with multiple display modes
float4 enhanced_waveform_fragment(float2 uv, float4 baseColor, constant AudioUniforms &uniforms, device WaveformHistory *history) {
    float4 color = baseColor;
    
    // Mode selection based on uniforms parameter
    int waveformMode = int(uniforms.themeIndex * 4) % 3; // Use theme index to select from 3 modes
    
    // Create base waveform visualization
    float centerY = 0.5; // Center of waveform
    float waveThickness = 0.005 * (1.0 + uniforms.sensitivity);
    
    // Calculate waveform value at this x-position
    int waveIndex = int(uv.x * 1024) % 1024;
    float waveHeight = uniforms.audioData[waveIndex] * uniforms.sensitivity;
    
    // Apply different waveform modes
    if (waveformMode == 0) {
        // Mode 0: Standard oscilloscope with thickness
        float waveY = centerY + waveHeight * 0.4;
        float dist = abs(uv.y - waveY);
        float waveIntensity = smoothstep(waveThickness, 0.0, dist);
        
        // Add subtle glow
        float glow = smoothstep(waveThickness * 3.0, 0.0, dist) * 0.3;
        
        // Calculate color based on position and audio
        float3 waveColor = mix(uniforms.primaryColor.rgb, uniforms.secondaryColor.rgb, 
                              (uv.y - centerY + 0.5) + sin(uv.x * 20.0 + uniforms.time * 3.0) * 0.1);
        
        color.rgb = mix(color.rgb, waveColor, waveIntensity);
        color.rgb += waveColor * glow * (0.5 + 0.5 * sin(uniforms.time * 5.0));
        
    } else if (waveformMode == 1) {
        // Mode 1: Persistence effect with history
        float currentWaveY = centerY + waveHeight * 0.4;
        float currentDist = abs(uv.y - currentWaveY);
        float currentIntensity = smoothstep(waveThickness, 0.0, currentDist);
        
        // Add current waveform
        float3 currentColor = mix(uniforms.primaryColor.rgb, uniforms.secondaryColor.rgb, 
                                 (sin(uv.x * 30.0 + uniforms.time * 2.0) * 0.5 + 0.5));
        color.rgb = mix(color.rgb, currentColor, currentIntensity);
        
        // Add historical waveforms if available
        if (history != nullptr) {
            for (int i = 0; i < 8; i++) {
                if (i != history->currentFrame && history->opacity[i] > 0.01) {
                    float historyWaveHeight = history->waveData[i][waveIndex] * uniforms.sensitivity;
                    float historyWaveY = centerY + historyWaveHeight * 0.4;
                    float historyDist = abs(uv.y - historyWaveY);
                    float historyIntensity = smoothstep(waveThickness * 1.5, 0.0, historyDist) * history->opacity[i] * 0.6;
                    
                    // Color based on age (older = more faded to accent color)
                    float3 historyColor = mix(currentColor, uniforms.accentColor.rgb, 1.0 - history->opacity[i]);
                    
                    color.rgb = mix(color.rgb, historyColor, historyIntensity);
                }
            }
        }
        
    } else if (waveformMode == 2) {
        // Mode 2: Stacked waveform bands
        int bands = 5;
        float bandHeight = 1.0 / float(bands);
        float bandOffset = bandHeight * 0.5;
        
        // Calculate which band this pixel is in
        int band = int(uv.y * bands);
        float bandCenter = bandHeight * (band + 0.5);
        
        // Calculate frequency range for this band
        int bandStart = 1024 / bands * band;
        int bandEnd = 1024 / bands * (band + 1);
        
        // Get average value for this band at this x-position
        float bandValue = 0.0;
        int samplesInBand = 0;
        for (int i = bandStart; i < bandEnd; i++) {
            int index = (waveIndex + i) % 1024;
            bandValue += uniforms.audioData[index];
            samplesInBand++;
        }
        bandValue /= float(samplesInBand);
        
        // Calculate waveform
        float waveY = bandCenter + bandValue * bandHeight * 0.8;
        float dist = abs(uv.y - waveY);
        float waveIntensity = smoothstep(waveThickness, 0.0, dist);
        
        // Create per-band coloring
        float bandHue = float(band) / float(bands);
        float3 bandColor = hsv2rgb(float3(bandHue, 0.7, 0.9));
        
        // Add band background
        float bandBackground = smoothstep(bandHeight * 0.48, bandHeight * 0.4, abs(uv.y - bandCenter)) * 0.1;
        color.rgb += bandColor * bandBackground;
        
        // Add waveform
        color.rgb = mix(color.rgb, bandColor, waveIntensity);
        
        // Add separator lines
        float separator = step(0.99, 1.0 - fract(uv.y * bands)) * 0.05;
        color.rgb += float3(1.0) * separator;
    }
    
    return color;
}

//-----------------------------------------------------------
// Physics-Based Particle System
//-----------------------------------------------------------

/// Structure for a physics-based particle
struct Particle {
    float2 position;
    float2 velocity;
    float size;
    float life;
    float maxLife;
    float4 color;
};

/// Particle system state
struct ParticleSystem {
    Particle particles[1024];
    uint activeCount;
    uint nextIndex;
    float spawnRate;
    float2 emitterPosition;
    float2 emitterSize;
    float time;
    float audioReactivity;
    float attractorStrength;
    float2 attractorPosition;
};

/// Compute kernel for particle system physics
kernel void compute_particle_physics(
    device ParticleSystem *system [[buffer(0)]],
    constant AudioUniforms &uniforms [[buffer(1)]],
    constant float &deltaTime [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    // Only process active particles
    if (id >= system->activeCount) {
        return;
    }
    
    // Get particle
    device Particle &particle = system->particles[id];
    
    // Update life
    particle.life -= deltaTime;
    
    // Skip dead particles
    if (particle.life <= 0) {
        // If this is the last active particle, just reduce count
        if (id == system->activeCount - 1) {
            system->activeCount--;
        } else {
            // Otherwise, swap with the last active particle to maintain contiguous array
            particle = system->particles[system->activeCount - 1];
            system->activeCount--;
        }
        return;
    }
    
    // Calculate audio influence
    float audioFactor = system->audioReactivity;
    
    // Get audio value at particle's position
    float2 normalizedPos = particle.position * 0.5 + 0.5; // Convert from [-1,1] to [0,1]
    int audioIndex = int(normalizedPos.x * 64) % 1024;
    float audioValue = uniforms.audioData[audioIndex] * uniforms.sensitivity;
    
    // Apply forces
    
    // 1. Attractor force
    float2 toAttractor = system->attractorPosition - particle.position;
    float distToAttractor = length(toAttractor);
    
    // Avoid division by zero
    if (distToAttractor > 0.001) {
        // Apply attraction force with falloff
        float attractionStrength = system->attractorStrength / (distToAttractor * distToAttractor);
        
        // Modulate by audio and beat
        attractionStrength *= 1.0 + audioValue * 2.0 + uniforms.beatDetected * 1.0;
        
        // Add to velocity
        particle.velocity += normalize(toAttractor) * attractionStrength * deltaTime;
    }
    
    // 2. Apply fluid-like forces based on audio frequency data
    float angle = uniforms.time * 0.2 + normalizedPos.x * 6.28318;
    float2 audioForce = float2(sin(angle), cos(angle)) * audioValue * audioFactor;
    particle.velocity += audioForce * deltaTime;
    
    // 3. Apply random turbulence
    float turbulenceFreq = 2.0 + uniforms.bassLevel * 3.0;
    float2 noisePos = particle.position * 0.1 + float2(uniforms.time * 0.1, uniforms.time * 0.2);
    float noiseX = fbm2d(noisePos, 3);
    float noiseY = fbm2d(noisePos + float2(100, 100), 3);
    float2 turbulence = float2(noiseX, noiseY) * 2.0 - 1.0;
    particle.velocity += turbulence * deltaTime * 0.5;
    
    // Apply drag
    particle.velocity *= (1.0 - min(0.95, deltaTime * 0.5));
    
    // Update position
    particle.position += particle.velocity * deltaTime;
    
    // Boundary conditions (keep particles on screen)
    if (particle.position.x < -1.0) {
        particle.position.x = -1.0;
        particle.velocity.x *= -0.5; // Bounce with energy loss
    } else if (particle.position.x > 1.0) {
        particle.position.x = 1.0;
        particle.velocity.x *= -0.5;
    }
    
    if (particle.position.y < -1.0) {
        particle.position.y = -1.0;
        particle.velocity.y *= -0.5;
    } else if (particle.position.y > 1.0) {
        particle.position.y = 1.0;
        particle.velocity.y *= -0.5;
    }
    
    // Update particle size based on life
    float lifeRatio = particle.life / particle.maxLife;
    particle.size = mix(0.001, particle.size, lifeRatio);
    
    // Update color opacity based on life
    particle.color.a = lifeRatio;
}

/// Particle spawning compute kernel
kernel void spawn_particles(
    device ParticleSystem *system [[buffer(0)]],
    constant AudioUniforms &uniforms [[buffer(1)]],
    constant float &deltaTime [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    // Only one thread should spawn particles
    if (id > 0) return;
    
    // Update system time
    system->time += deltaTime;
    
    // Calculate spawn count based on audio reactivity
    float bassEnergy = uniforms.bassLevel * uniforms.sensitivity;
    float beatBoost = uniforms.beatDetected * 10.0;
    
    // Base spawn rate + audio reactivity + beat detection
    float spawnCount = system->spawnRate * deltaTime * (1.0 + bassEnergy * 3.0 + beatBoost);
    
    // Fractional spawning (stochastic)
    float fractionalPart = fract(spawnCount);
    uint integerPart = uint(spawnCount);
    
    // Add random chance for fractional part
    float random = fract(sin(system->time * 12345.67890) * 43758.5453);
    if (random < fractionalPart) {
        integerPart += 1;
    }
    
    // Spawn particles
    for (uint i = 0; i < integerPart; i++) {
        // Check if we have room for more particles
        if (system->activeCount >= 1024) {
            break;
        }
        
        // Generate position within emitter
        float2 randomVal = float2(
            fract(sin(system->time * (1234.5 + i)) * 43758.5453),
            fract(sin(system->time * (6789.0 + i)) * 12345.6789)
        );
        
        // Calculate spawn position
        float2 position = system->emitterPosition + (randomVal * 2.0 - 1.0) * system->emitterSize;
        
        // Calculate initial velocity (outward from center with audio reactivity)
        float2 direction = normalize(position - system->emitterPosition);
        float speed = 0.2 + uniforms.bassLevel * 0.5 + uniforms.beatDetected * 0.3;
        float2 velocity = direction * speed;
        
        // Create color based on audio frequencies at spawn location
        int audioIdx = int(randomVal.x * 64) % 1024;
        float audioVal = uniforms.audioData[audioIdx];
        
        // Map audio value to color
        float hue = fract(audioVal + system->time * 0.1);
        float3 rgb = hsv2rgb(float3(hue, 0.8, 0.9));
        
        // Create particle
        Particle newParticle;
        newParticle.position = position;
        newParticle.velocity = velocity;
        newParticle.size = 0.005 + randomVal.x * 0.01 + uniforms.bassLevel * 0.01;
        newParticle.life = 2.0 + randomVal.y * 3.0;
        newParticle.maxLife = newParticle.life;
        newParticle.color = float4(rgb, 1.0);
        
        // Add to system
        system->particles[system->activeCount] = newParticle;
        system->activeCount++;
    }
    
    // Update attractor position based on audio
    float angle = system->time * 0.5;
    float radius = 0.5 + 0.3 * sin(system->time * 0.2);
    
    // Modulate by bass for movement
    radius *= 1.0 + uniforms.bassLevel * 0.5;
    angle += uniforms.beatDetected * 1.0; // Jump on beat
    
    // Calculate new position
    system->attractorPosition = float2(sin(angle), cos(angle)) * radius;
    
    // Update audio reactivity based on audio levels
    system->audioReactivity = mix(0.5, 2.0, uniforms.sensitivity * (uniforms.bassLevel + uniforms.trebleLevel) * 0.5);
    
    // Adjust attractor strength based on beat
    system->attractorStrength = 0.1 + uniforms.beatDetected * 0.2;
}

/// 2D fbm noise helper for particle system
float fbm2d(float2 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    for(int i = 0; i < octaves; i++) {
        // Use existing noise function from AudioVisualizer.metal
        value += amplitude * noise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return value;
}

/// Render a particle system
float4 render_particles(float2 uv, float4 baseColor, constant AudioUniforms &uniforms, device ParticleSystem *system) {
    float4 color = baseColor;
    
    // Skip if no particle system
    if (system == nullptr) {
        return color;
    }
    
    // Convert UV to position in [-1,1] range
    float2 pos = uv * 2.0 - 1.0;
    
    // Add background effects based on attractor
    float2 toAttractor = pos - system->attractorPosition;
    float distToAttractor = length(toAttractor);
    
    // Add subtle glow around attractor
    float attractorGlow = smoothstep(0.5, 0.0, distToAttractor) * 0.2 * system->attractorStrength;
    color.rgb += uniforms.primaryColor.rgb * attractorGlow;
    
    // Add flow field visualization (subtle background effect)
    float angle = atan2(toAttractor.y, toAttractor.x) + uniforms.time * 0.1;
    float flowPattern = sin(angle * 10.0 + uniforms.time * 2.0) * 0.5 + 0.5;
    float flowStrength = smoothstep(1.0, 0.0, distToAttractor) * 0.05;
    color.rgb += mix(uniforms.secondaryColor.rgb, uniforms.accentColor.rgb, flowPattern) * flowStrength;
    
    // Render each particle
    for (uint i = 0; i < system->activeCount; i++) {
        Particle particle = system->particles[i];
        
        // Calculate distance to particle center
        float2 toParticle = pos - particle.position;
        float dist = length(toParticle);
        
        // Check if pixel is within particle radius
        if (dist < particle.size) {
            // Calculate intensity (1.0 at center, 0.0 at edge)
            float intensity = 1.0 - smoothstep(0.0, particle.size, dist);
            
            // Apply softening for more organic look
            intensity = pow(intensity, 1.5);
            
            // Get particle color with proper opacity
            float4 particleColor = particle.color;
            particleColor.a *= intensity;
            
            // Add glow based on velocity
            float speed = length(particle.velocity);
            float3 glowColor = mix(particle.color.rgb, uniforms.accentColor.rgb, min(1.0, speed * 2.0));
            float glow = smoothstep(particle.size * 2.0, 0.0, dist) * 0.3 * speed;
            
            // Blend particle with background
            color.rgb = mix(color.rgb, particleColor.rgb, particleColor.a);
            color.rgb += glowColor * glow;
        }
    }
    
    return color;
}
//-----------------------------------------------------------
// Neural Pattern Generation
//-----------------------------------------------------------

/// Structure to store neural visualization state
struct NeuralPatternState {
    // Pattern evolution parameters
    float time;
    float evolutionSpeed;
    float complexity;
    float energy;
    float pleasantness;
    
    // Pattern control points (for organic shapes)
    float2 controlPoints[16];
    float controlPointInfluence[16];
    
    // Color evolution
    float4 colorPalette[4];
    float colorMix;
    
    // Beat reaction
    float beatIntensity;
    float beatDecay;
    
    // Flow field
    float flowFieldScale;
    float flowFieldSpeed;
    float flowFieldStrength;
    
    // Pattern memory (for evolution)
    float patternSeed;
    float patternVariation;
    int patternType;
};

/// Compute kernel to evolve neural patterns
kernel void evolve_neural_pattern(
    device NeuralPatternState *state [[buffer(0)]],
    constant AudioUniforms &uniforms [[buffer(1)]],
    constant float &deltaTime [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    // Only thread 0 updates the pattern state
    if (id > 0) return;
    
    // Update time
    state->time += deltaTime * state->evolutionSpeed;
    
    // Update neural parameters based on uniform inputs
    state->energy = uniforms.neuralEnergy;
    state->complexity = uniforms.neuralComplexity;
    state->pleasantness = uniforms.neuralPleasantness;
    
    // Update evolution speed based on energy and complexity
    state->evolutionSpeed = mix(0.3, 1.5, state->energy * state->complexity);
    
    // Update flow field parameters
    state->flowFieldScale = mix(1.0, 5.0, state->complexity);
    state->flowFieldSpeed = mix(0.2, 1.0, state->energy);
    state->flowFieldStrength = mix(0.1, 0.5, state->energy * state->pleasantness);
    
    // Update beat reaction
    if (uniforms.beatDetected > 0.5) {
        state->beatIntensity = 1.0;
    } else {
        state->beatIntensity = max(0.0, state->beatIntensity - deltaTime * state->beatDecay);
    }
    state->beatDecay = mix(0.5, 2.0, state->energy);
    
    // Update control points for organic shapes
    for (int i = 0; i < 16; i++) {
        // Create smooth movement based on sine waves with different frequencies
        float angle = state->time * (0.1 + 0.05 * i) + i * 0.4;
        float radius = 0.3 + 0.2 * sin(state->time * 0.2 + i);
        
        // Adjust radius based on beat and energy
        radius *= 1.0 + state->beatIntensity * 0.3 * sin(state->time * 10.0 + i);
        radius *= mix(0.8, 1.2, state->energy);
        
        // Calculate position
        state->controlPoints[i] = float2(
            sin(angle) * radius,
            cos(angle) * radius
        );
        
        // Calculate influence based on pleasantness (higher = smoother transitions)
        state->controlPointInfluence[i] = mix(0.1, 0.4, state->pleasantness) * (1.0 + 0.3 * sin(state->time + i));
    }
    
    // Update color palette
    // First color: energy-based (reds/oranges)
    float energyHue = mix(0.0, 0.1, state->energy);
    float energySat = mix(0.7, 1.0, state->energy);
    state->colorPalette[0] = float4(hsv2rgb(float3(energyHue, energySat, 1.0)), 1.0);
    
    // Second color: pleasantness-based (blues/purples)
    float pleasantnessHue = mix(0.5, 0.8, state->pleasantness);
    state->colorPalette[1] = float4(hsv2rgb(float3(pleasantnessHue, 0.8, 0.9)), 1.0);
    
    // Third color: complexity-based (greens/teals)
    float complexityHue = mix(0.3, 0.5, state->complexity);
    state->colorPalette[2] = float4(hsv2rgb(float3(complexityHue, 0.7, 0.8)), 1.0);
    
    // Fourth color: time-based for variation
    float timeHue = fract(state->time * 0.05);
    state->colorPalette[3] = float4(hsv2rgb(float3(timeHue, 0.6, 1.0)), 1.0);
    
    // Update color mix factor based on time and beat
    state->colorMix = fract(state->time * 0.1) + state->beatIntensity * 0.3;
    
    // Update pattern variation based on neural parameters and time
    state->patternVariation = mix(0.0, 1.0, sin(state->time * 0.1) * 0.5 + 0.5);
    
    // Switch pattern type occasionally based on complexity
    if (fract(state->time * 0.05) < 0.01) {
        state->patternType = int(state->complexity * 4.0) % 4;
    }
}

/// Calculate organic pattern value at a position
float calculate_neural_pattern(float2 position, device NeuralPatternState *state) {
    // Default pattern value
    float patternValue = 0.0;
    
    // Base pattern based on pattern type
    switch (state->patternType) {
        case 0: { // Cellular/organic pattern
            // Calculate influence from all control points
            float totalInfluence = 0.0;
            float weightSum = 0.0;
            
            for (int i = 0; i < 16; i++) {
                float2 toPoint = position - state->controlPoints[i];
                float dist = length(toPoint);
                float weight = exp(-dist * 5.0) * state->controlPointInfluence[i];
                
                // Add weighted contribution
                float angle = atan2(toPoint.y, toPoint.x) + state->time * 0.2;
                float contribution = sin(angle * (2.0 + i % 3) + state->time + dist * 5.0);
                
                totalInfluence += contribution * weight;
                weightSum += weight;
            }
            
            // Normalize
            if (weightSum > 0.001) {
                patternValue = totalInfluence / weightSum;
            }
            
            // Add detail noise
            float detailNoise = fbm2d(position * state->flowFieldScale + state->time * 0.1, 3);
            patternValue = patternValue * 0.7 + detailNoise * 0.3;
            break;
        }
        
        case 1: { // Flow field pattern
            // Create flowing noise pattern
            float2 flowOffset = float2(
                sin(position.y * 4.0 + state->time * state->flowFieldSpeed),
                cos(position.x * 4.0 + state->time * state->flowFieldSpeed * 0.7)
            ) * state->flowFieldStrength;
            
            // Apply flow to position
            float2 flowPos = position + flowOffset;
            
            // Multi-layered noise for detail
            float noise1 = fbm2d(flowPos * state->flowFieldScale, 2);
            float noise2 = fbm2d(flowPos * state->flowFieldScale * 2.0 + float2(100, 100), 3);
            
            // Combine layers with different weights
            patternValue = noise1 * 0.7 + noise2 * 0.3;
            
            // Add ripples based on beat
            float beatRipple = sin(length(position) * 20.0 - state->time * 5.0) * 0.5 + 0.5;
            patternValue = mix(patternValue, beatRipple, state->beatIntensity * 0.3);
            break;
        }
        
        case 2: { // Neural network-like pattern
            // Create grid of "neurons"
            float2 grid = fract(position * state->flowFieldScale) - 0.5;
            float gridDist = length(grid);
            
            // Neurons pulsing
            float neuronPulse = smoothstep(0.1, 0.0, gridDist) * (0.5 + 0.5 * sin(state->time * 3.0));
            
            // Create "connections" between neurons
            float connections = smoothstep(0.03, 0.01, 
                min(abs(grid.x), abs(grid.y)) + gridDist * 0.2
            );
            
            // Flowing energy along connections
            float flow = sin(
                (position.x + position.y) * 10.0 + state->time * 2.0
            ) * 0.5 + 0.5;
            
            // Combine elements
            patternValue = max(neuronPulse, connections * flow * 0.7);
            
            // Add beat response
            patternValue = mix(patternValue, 1.0, state->beatIntensity * neuronPulse);
            break;
        }
        
        case 3: { // Mandelbrot-inspired fractal pattern
            // Map position to complex plane
            float scale = mix(1.0, 3.0, state->complexity);
            float2 c = position * scale;
            
            // Offset center based on time and energy
            c += float2(
                sin(state->time * 0.2) * 0.3,
                cos(state->time * 0.15) * 0.3
            ) * state->energy;
            
            // Iterative formula (simplified Mandelbrot-like)
            float2 z = float2(0.0);
            int iterations = int(mix(10.0, 30.0, state->complexity));
            int i = 0;
            
            for (; i < iterations; i++) {
                // z = z^2 + c
                float x = z.x * z.x - z.y * z.y + c.x;
                float y = 2.0 * z.x * z.y + c.y;
                
                z = float2(x, y);
                
                if (length(z) > 2.0) break;
            }
            
            // Normalize iteration count to [0,1]
            patternValue = float(i) / float(iterations);
            
            // Add time-based color cycling
            patternValue = fract(patternValue + state->time * 0.1);
            break;
        }
    }
    
    // Apply variation blend between pattern types for smoother transitions
    if (state->patternVariation > 0.01) {
        // Calculate an alternative pattern value from another pattern type
        float altPatternValue = 0.0;
        int altType = (state->patternType + 1) % 4;
        
        switch (altType) {
            case 0: {
                // Simplified cellular pattern for blending
                float2 toCenter = position;
                float dist = length(toCenter);
                altPatternValue = sin(dist * 10.0 - state->time * 2.0) * 0.5 + 0.5;
                break;
            }
            case 1: {
                // Simplified flow field for blending
                altPatternValue = fbm2d(position * state->flowFieldScale + state->time * 0.2, 2);
                break;
            }
            case 2: {
                // Simplified neural pattern for blending
                float2 grid = fract(position * state->flowFieldScale) - 0.5;
                altPatternValue = smoothstep(0.1, 0.0, length(grid));
                break;
            }
            case 3: {
                // Simplified fractal pattern for blending
                altPatternValue = fract(length(position) * 5.0 + state->time * 0.2);
                break;
            }
        }
        
        // Blend between patterns for smoother transitions
        patternValue = mix(patternValue, altPatternValue, state->patternVariation);
    }
    
    return patternValue;
}
/// Render enhanced neural visualization
float4 render_neural_pattern(float2 uv, float4 baseColor, constant AudioUniforms &uniforms, device NeuralPatternState *state) {
    float4 color = baseColor;
    
    // Return base color if no state is available
    if (state == nullptr) {
        return color;
    }
    
    // Convert uv to position in [-1,1] range for pattern calculation
    float2 pos = uv * 2.0 - 1.0;
    
    // Calculate pattern value at this position
    float patternValue = calculate_neural_pattern(pos, state);
    
    // Apply color mapping based on pattern value
    
    // Get colors from palette and blend based on pattern value
    int colorIndex1 = int(state->colorMix * 4) % 4;
    int colorIndex2 = (colorIndex1 + 1) % 4;
    float colorBlend = fract(state->colorMix * 4);
    
    float4 patternColor1 = state->colorPalette[colorIndex1];
    float4 patternColor2 = state->colorPalette[colorIndex2];
    
    // Blend between colors based on pattern and color mix
    float4 patternColor = mix(patternColor1, patternColor2, colorBlend);
    
    // Modulate color based on pattern value
    float4 finalPatternColor = mix(
        patternColor * 0.3,                           // Dark areas
        patternColor,                                 // Bright areas
        smoothstep(0.2, 0.8, patternValue)            // Smooth transition
    );
    
    // Apply beat reactivity
    if (state->beatIntensity > 0.01) {
        // Add pulsing glow on beat
        float beatPulse = state->beatIntensity * (0.5 + 0.5 * sin(state->time * 20.0));
        
        // Enhanced brightness and contrast during beats
        finalPatternColor.rgb = mix(
            finalPatternColor.rgb,
            patternColor.rgb * 1.5,                   // Brighter colors
            beatPulse * 0.5
        );
        
        // Add radial pulse wave on beat
        float dist = length(pos);
        float pulseWave = smoothstep(0.05, 0.0, abs(dist - fract(state->time * 2.0)) - 0.01);
        finalPatternColor.rgb += state->colorPalette[3].rgb * pulseWave * state->beatIntensity * 0.5;
    }
    
    // Add glow effects
    float glow = 0.0;
    
    // Edge glow based on pattern gradient
    float2 offset1 = float2(0.005, 0.0);
    float2 offset2 = float2(0.0, 0.005);
    
    float patternX1 = calculate_neural_pattern(pos - offset1, state);
    float patternX2 = calculate_neural_pattern(pos + offset1, state);
    float patternY1 = calculate_neural_pattern(pos - offset2, state);
    float patternY2 = calculate_neural_pattern(pos + offset2, state);
    
    float2 patternGradient = float2(
        patternX2 - patternX1,
        patternY2 - patternY1
    ) * 10.0; // Scale up for visibility
    
    // Glow intensity based on gradient magnitude
    float gradientMag = length(patternGradient);
    glow += smoothstep(0.0, 0.5, gradientMag) * 0.3;
    
    // Add glow to high-value areas
    glow += smoothstep(0.7, 0.9, patternValue) * 0.3;
    
    // Add time-based pulsing glow
    float timePulse = (sin(state->time + patternValue * 6.28318) * 0.5 + 0.5) * 0.2;
    glow += timePulse * smoothstep(0.5, 0.8, patternValue);
    
    // Add beat-reactive glow
    glow += state->beatIntensity * 0.3 * smoothstep(0.6, 0.9, patternValue);
    
    // Apply energy parameter to overall glow intensity
    glow *= mix(0.5, 1.5, state->energy);
    
    // Add glow to final color
    float4 glowColor = mix(finalPatternColor, state->colorPalette[3], 0.5);
    finalPatternColor.rgb += glowColor.rgb * glow;
    
    // Apply depth effects with parallax layers
    float depth = 3; // Number of depth layers
    for (int i = 1; i < depth; i++) {
        float layerDepth = float(i) / depth;
        float2 offsetPos = pos * (1.0 - layerDepth * 0.2);
        
        // Add time offset for parallax movement
        offsetPos += float2(
            sin(state->time * 0.2 + layerDepth * 3.14159),
            cos(state->time * 0.15 + layerDepth * 1.57)
        ) * 0.02 * layerDepth;
        
        // Calculate pattern at offset depth
        float depthPattern = calculate_neural_pattern(offsetPos, state);
        
        // Only show higher values for deeper layers
        float layerThreshold = mix(0.4, 0.8, layerDepth);
        float layerIntensity = smoothstep(layerThreshold, 1.0, depthPattern) * (1.0 - layerDepth) * 0.3;
        
        // Mix with different color for each layer
        float layerHue = fract(state->colorMix + layerDepth * 0.2);
        float3 layerColor = hsv2rgb(float3(layerHue, 0.7, 0.9));
        
        finalPatternColor.rgb = mix(
            finalPatternColor.rgb,
            layerColor,
            layerIntensity
        );
    }
    
    // Apply neural visualization to base color
    float patternOpacity = mix(0.7, 0.9, state->energy);
    color.rgb = mix(color.rgb, finalPatternColor.rgb, patternOpacity);
    
    // Add subtle vignette effect
    float vignette = 1.0 - smoothstep(0.5, 1.2, length(pos));
    color.rgb *= mix(0.7, 1.0, vignette);
    
    // Add highlight flares based on neural parameters
    if (state->pleasantness > 0.5) {
        // More pleasant patterns get subtle light flares
        float flareIntensity = (state->pleasantness - 0.5) * 2.0 * 0.3;
        
        // Create several random light points
        for (int i = 0; i < 5; i++) {
            float angle = 2.8 * i + state->time * 0.2;
            float radius = 0.4 + 0.3 * sin(state->time * 0.1 + i);
            
            float2 flarePos = float2(sin(angle), cos(angle)) * radius;
            float flareDist = length(pos - flarePos);
            
            // Create light flare with soft falloff
            float flare = smoothstep(0.2, 0.0, flareDist) * flareIntensity;
            
            // Add to final color
            color.rgb += state->colorPalette[i % 4].rgb * flare;
        }
    }
    
    return color;
}

//-----------------------------------------------------------
// Visualization Processing Utilities
//-----------------------------------------------------------

/// Combine visualization modes with smooth transitions
float4 blend_visualization_modes(
    float2 uv,
    float4 baseColor,
    constant AudioUniforms &uniforms,
    device SpectrumPeaks *peaks,
    device WaveformHistory *waveHistory,
    device ParticleSystem *particles,
    device NeuralPatternState *neuralState
) {
    // Get current visualization mode
    int visualizationMode = int(uniforms.visualizationMode);
    
    // Default to base color
    float4 color = baseColor;
    
    // Apply appropriate visualization effect based on mode
    if (visualizationMode == 0) {
        // Spectrum visualization
        color = enhanced_spectrum_fragment(uv, baseColor, uniforms, peaks);
    }
    else if (visualizationMode == 1) {
        // Waveform visualization
        color = enhanced_waveform_fragment(uv, baseColor, uniforms, waveHistory);
    }
    else if (visualizationMode == 2) {
        // Particle visualization
        color = render_particles(uv, baseColor, uniforms, particles);
    }
    else if (visualizationMode == 3) {
        // Neural visualization
        color = render_neural_pattern(uv, baseColor, uniforms, neuralState);
    }
    
    // If in transition between modes, blend with previous mode
    if (uniforms.transitionProgress > 0.0 && uniforms.transitionProgress < 1.0) {
        // Get previous mode
        int prevMode = int(uniforms.previousMode);
        float4 prevColor = baseColor;
        
        // Apply previous mode visualization
        if (prevMode == 0) {
            prevColor = enhanced_spectrum_fragment(uv, baseColor, uniforms, peaks);
        }
        else if (prevMode == 1) {
            prevColor = enhanced_waveform_fragment(uv, baseColor, uniforms, waveHistory);
        }
        else if (prevMode == 2) {
            prevColor = render_particles(uv, baseColor, uniforms, particles);
        }
        else if (prevMode == 3) {
            prevColor = render_neural_pattern(uv, baseColor, uniforms, neuralState);
        }
        
        // Blend between previous and current mode
        float transition = uniforms.transitionProgress;
        
        // Add transition effect (crossfade with a bit of brightness boost)
        float transitionBoost = sin(transition * 3.14159) * 0.2;
        color = mix(prevColor, color, transition);
        color.rgb += color.rgb * transitionBoost; // Slight brightness boost during transition
    }
    
    return color;
}

//-----------------------------------------------------------
// End of Enhanced Visualization Effects
//-----------------------------------------------------------
