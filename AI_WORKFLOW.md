# AI Agent Workflow Guide

This document provides guidelines and instructions for AI agents working on the AudioBloomAI project.

## AI Task Structure

### Task Components
1. **Technical Requirements**
   - Platform requirements (macOS 15+, Swift 6)
   - Dependencies and frameworks
   - Performance constraints
   - Integration points

2. **AI Agent Instructions**
   - Step-by-step implementation guide
   - Required inputs and expected outputs
   - Error handling requirements
   - Testing specifications

3. **Complexity Assessment**
   - 1-Simple: Clear implementation path, minimal context
   - 2-Straightforward: Well-defined task with some dependencies
   - 3-Moderate: Requires architectural understanding
   - 4-Complex: Significant system knowledge needed
   - 5-Highly Complex: Multiple subsystem interactions

4. **Definition of Done**
   - Implementation criteria
   - Test coverage requirements
   - Documentation standards
   - Performance benchmarks

## Working with Issues

### Reading Issues
1. Review Technical Requirements section
2. Understand AI Agent Instructions
3. Verify Complexity Score accuracy
4. Check Definition of Done criteria
5. Review related documentation

### Updating Issues
1. Keep AI Ready Status current
2. Document implementation progress
3. Update related documentation
4. Link to relevant commits/PRs
5. Flag human review needs

## Code Standards

### Documentation
- Use DocC comment format
- Include code examples
- Document performance implications
- Specify thread safety requirements

### Implementation
- Follow Swift 6.0 guidelines
- Implement error handling
- Add unit tests
- Update relevant documentation

### Testing
- Write unit tests for new code
- Include performance tests
- Document test cases
- Verify edge cases

## Module-Specific Guidelines

### AudioProcessor
- Real-time processing requirements
- Buffer management considerations
- Thread safety requirements
- Error recovery procedures

### Visualizer
- Metal shader optimization
- Frame rate requirements
- Memory management
- State handling

### MLEngine
- Model integration guidelines
- Feature extraction requirements
- Performance optimization
- Error handling

### AudioBloomUI
- SwiftUI best practices
- State management
- Performance considerations
- Accessibility requirements

## Communication Protocol

### Flagging Human Review
- Complex architectural decisions
- Security implications
- Performance trade-offs
- User experience considerations

### Progress Updates
- Implementation status
- Technical blockers
- Performance metrics
- Test results

## Performance Standards

### Audio Processing
- Maximum latency: 10ms
- Buffer size optimization
- CPU usage limits
- Memory constraints

### Visualization
- Target frame rate: 60 FPS
- GPU resource usage
- Memory footprint
- Transition smoothness

## Error Handling

### Critical Errors
- Audio stream failures
- GPU context loss
- Memory warnings
- File system errors

### Recovery Procedures
- State restoration
- Resource cleanup
- User notification
- Logging requirements

## Documentation Updates

### When to Update
- New feature implementation
- API changes
- Performance improvements
- Bug fixes

### What to Include
- Technical specifications
- Implementation details
- Usage examples
- Performance characteristics

## Continuous Integration

### Pre-commit Checks
- Swift 6.0 compilation
- Unit test passage
- Documentation coverage
- Performance benchmarks

### Quality Gates
- Code coverage thresholds
- Performance requirements
- Documentation standards
- API compatibility

This guide ensures consistent AI agent participation in the AudioBloomAI project development process. Follow these guidelines for all AI-assisted development tasks.

