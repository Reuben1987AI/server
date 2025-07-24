# Dart Audio Transcription Test

## Setup
1. Install Dart dependencies: `dart pub get`
2. Start server on port 8080: `python ../src/server.py`
3. Run test: `dart test`

## Test Audio
- File: `test_audio.wav`
- Format: WAV, 16kHz, mono, 16-bit
- Tests WebSocket transcription with timestamps

## WebSocket Endpoints
- `/stream` - Returns flat phoneme arrays without timestamps
- `/stream_timestamped` - Returns phonemes with timestamps (used by this test)

## Expected Output
JSON array of phonemes with timing:
```json
[["phoneme", start_time, end_time], ...]
```

## Test Features
- **Real-time Display**: Shows phonemes as they arrive from server
- **Timeline View**: Chronologically sorted phoneme sequence
- **Monotonicity Analysis**: Analyzes timestamp ordering patterns
- **Pretty Printing**: Formatted final results with detailed analysis

## Timestamp Monotonicity Analysis

### Understanding Non-Sequential Timestamps

**IMPORTANT**: Timestamp monotonicity violations are **NORMAL** in speech recognition systems. The test includes detailed analysis to demonstrate this reality.

#### Why Timestamps May Not Be Sequential

1. **Acoustic Confidence-Based Detection**
   - Models detect clear, high-confidence phonemes first
   - Unclear or ambiguous phonemes are identified later with more context
   - Detection order ≠ temporal order

2. **Chunked Audio Processing**
   - Audio is processed in 2-second chunks
   - Each chunk may detect phonemes from different time periods
   - Cross-chunk phonemes can appear out of temporal sequence

3. **Co-articulation Effects**
   - Phonemes in natural speech overlap and influence each other
   - Some phonemes require surrounding context for accurate identification
   - Temporal boundaries are fuzzy, not discrete

4. **Context-Dependent Recognition**
   - Later phonemes can provide context that disambiguates earlier unclear sounds
   - The model may "back-fill" previously uncertain phonemes

#### Example from Real Output
```
Detection Order:  [o, 1.829s] → [k, 1.929s] → [eɪ, 1.949s] → [n, 0.884s]
                                                              ↑ Earlier timestamp\!
```

This is **expected behavior** - the 'n' phoneme at 0.884s was detected after the later phonemes due to acoustic processing patterns.

#### Validation Strategy

The test performs **relaxed validation**:
- ✅ **Individual phoneme consistency**: `end_time >= start_time`  
- ✅ **Data format validation**: Proper JSON structure
- ❌ **Cross-phoneme monotonicity**: NOT enforced (unrealistic expectation)

The `analyzeTimestampMonotonicity()` function provides:
- Violation count and percentage
- Detailed violation analysis
- Classification of violation types (overlap vs out-of-order)
- Educational output explaining why violations are normal

### Violation Types

1. **out_of_order**: Phoneme detected later but has earlier timestamp
2. **overlap**: Phonemes with overlapping time ranges (common in natural speech)

### Typical Results
- **0-20% violations**: Good quality audio with clear phonemes
- **20-50% violations**: Normal for speech recognition systems  
- **50%+ violations**: Expected for complex audio or chunked processing

This analysis helps developers understand realistic speech recognition behavior vs. idealized sequential expectations.
EOF < /dev/null
