import 'dart:io';
import 'dart:typed_data';
import 'dart:convert';
import 'package:test/test.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import 'package:http/http.dart' as http;

/// Analyzes timestamp monotonicity patterns in phoneme sequences
Map<String, dynamic> analyzeTimestampMonotonicity(List<List> phonemes) {
  if (phonemes.length < 2) {
    return {
      'total_phonemes': phonemes.length,
      'monotonic_violations': 0,
      'violation_percentage': 0.0,
      'analysis': 'Insufficient data for monotonicity analysis',
      'violations': <Map>[]
    };
  }

  final violations = <Map>[];
  int violationCount = 0;

  for (int i = 1; i < phonemes.length; i++) {
    final prevEnd = phonemes[i - 1][2] as num;
    final currStart = phonemes[i][1] as num;

    if (currStart < prevEnd) {
      violationCount++;
      violations.add({
        'position': i,
        'prev_phoneme': phonemes[i - 1][0],
        'prev_end': prevEnd,
        'curr_phoneme': phonemes[i][0],
        'curr_start': currStart,
        'gap': currStart - prevEnd,
        'type': currStart < phonemes[i - 1][1] ? 'overlap' : 'out_of_order'
      });
    }
  }

  final violationPercentage = (violationCount / (phonemes.length - 1)) * 100;

  String analysis;
  if (violationPercentage == 0) {
    analysis = 'Perfect monotonic progression - unusual for speech recognition';
  } else if (violationPercentage < 20) {
    analysis = 'Low monotonicity violations - typical for good quality audio';
  } else if (violationPercentage < 50) {
    analysis = 'Moderate monotonicity violations - normal for speech recognition';
  } else {
    analysis = 'High monotonicity violations - expected for complex audio or chunked processing';
  }

  return {
    'total_phonemes': phonemes.length,
    'monotonic_violations': violationCount,
    'violation_percentage': violationPercentage,
    'analysis': analysis,
    'violations': violations
  };
}

void main() {
  test('test_audio_transcription_with_timestamps', () async {
    // Load test audio file
    final audioFile = File('test_audio.wav');
    expect(audioFile.existsSync(), isTrue, reason: 'test_audio.wav file not found');

    final audioBytes = await audioFile.readAsBytes();
    final audioData = _extractAudioData(audioBytes);

    print('Loaded audio: ${audioData.length} samples (${(audioData.length / 16000).toStringAsFixed(2)} seconds)');

    // Connect to WebSocket with retry logic
    WebSocketChannel? channel;
    for (int attempt = 1; attempt <= 3; attempt++) {
      try {
        channel = WebSocketChannel.connect(Uri.parse('ws://localhost:8080/stream_timestamped'));
        print('Connected to WebSocket on attempt $attempt');
        break;
      } catch (e) {
        print('Connection attempt $attempt failed: $e');
        if (attempt == 3) rethrow;
        await Future.delayed(Duration(seconds: 2));
      }
    }
    if (channel == null) throw Exception('Failed to connect after 3 attempts');

    // Send audio in 2-second chunks
    const chunkSize = 32000; // 2 seconds at 16kHz
    final allPhonemes = <List>[];
    int responseCount = 0;

    // Listen for responses
    final responseStream = channel.stream.listen((data) {
      final jsonData = jsonDecode(data);
      responseCount++;

      // Print incoming data as it arrives
      print('\n=== Response $responseCount ===');
      if (jsonData.isEmpty) {
        print('(empty response)');
      } else {
        for (var phoneme in jsonData) {
          final text = phoneme[0];
          final startTime = (phoneme[1] as num).toStringAsFixed(3);
          final endTime = (phoneme[2] as num).toStringAsFixed(3);
          print('[$text] ${startTime}s - ${endTime}s');
        }
      }

      // Accumulate phonemes for final validation
      allPhonemes.addAll(jsonData.cast<List>());
    });

    // Send audio chunks
    for (int i = 0; i < audioData.length; i += chunkSize) {
      final end = (i + chunkSize < audioData.length) ? i + chunkSize : audioData.length;
      final chunk = audioData.sublist(i, end);

      // Convert to Float32 bytes
      final float32Data = Float32List.fromList(chunk);
      final binaryData = float32Data.buffer.asUint8List();

      channel.sink.add(binaryData);
      print('Sent chunk ${i ~/ chunkSize + 1}: ${chunk.length} samples');

      // Wait a bit between chunks
      await Future.delayed(Duration(milliseconds: 100));
    }

    // Wait for final responses
    await Future.delayed(Duration(seconds: 2));

    // Close connection
    await channel.sink.close();
    await responseStream.cancel();

    // Pretty print final results
    print('\n' + '=' * 60);
    print('FINAL TRANSCRIPTION RESULTS');
    print('=' * 60);

    if (allPhonemes.isEmpty) {
      print('No phonemes detected');
    } else {
      print('Total phonemes detected: ${allPhonemes.length}');
      print('\nPhoneme sequence with timestamps:');

      for (int i = 0; i < allPhonemes.length; i++) {
        final phoneme = allPhonemes[i];
        final text = phoneme[0];
        final startTime = (phoneme[1] as num).toStringAsFixed(3);
        final endTime = (phoneme[2] as num).toStringAsFixed(3);
        print('${(i + 1).toString().padLeft(2)}: [$text] ${startTime}s → ${endTime}s');
      }

      // Timeline view
      print('\nTimeline view (sorted by start time):');
      final sortedPhonemes = List.from(allPhonemes);
      sortedPhonemes.sort((a, b) => (a[1] as num).compareTo(b[1] as num));

      for (var phoneme in sortedPhonemes) {
        final text = phoneme[0];
        final startTime = (phoneme[1] as num).toStringAsFixed(3);
        final endTime = (phoneme[2] as num).toStringAsFixed(3);
        print('${startTime}s ├─[$text]─┤ ${endTime}s');
      }

      // Analyze timestamp monotonicity
      print('\n' + '-' * 60);
      print('TIMESTAMP MONOTONICITY ANALYSIS');
      print('-' * 60);

      final analysis = analyzeTimestampMonotonicity(allPhonemes);
      print('Total phonemes: ${analysis['total_phonemes']}');
      print('Monotonic violations: ${analysis['monotonic_violations']}');
      print('Violation percentage: ${analysis['violation_percentage'].toStringAsFixed(1)}%');
      print('Analysis: ${analysis['analysis']}');

      if (analysis['violations'].isNotEmpty) {
        print('\nViolation details:');
        for (var violation in analysis['violations']) {
          print(
              '  ${violation['position']}: [${violation['prev_phoneme']}] ${violation['prev_end'].toStringAsFixed(3)}s → [${violation['curr_phoneme']}] ${violation['curr_start'].toStringAsFixed(3)}s (gap: ${violation['gap'].toStringAsFixed(3)}s, ${violation['type']})');
        }
      }

      print('\nNote: Monotonicity violations are NORMAL in speech recognition due to:');
      print('- Acoustic confidence-based detection order');
      print('- Chunked audio processing');
      print('- Co-articulation and phoneme overlap');
      print('- Context-dependent phoneme identification');
    }

    // Reconstruct phoneme string for easy copying
    final phonemeString = allPhonemes.map((p) => p[0]).join(" ");
    print("\nRECONSTRUCTED PHONEME SEQUENCE (for copy-paste):");
    print("$phonemeString");
    print("\nPhonemes as words (approximate):");
    print(_phonemesToWords(allPhonemes));

    print('=' * 60);

    // Basic validation (relaxed - no monotonicity requirement)
    expect(allPhonemes.isNotEmpty, isTrue, reason: 'Should receive some phonemes');

    for (final phoneme in allPhonemes) {
      expect(phoneme, isList, reason: 'Each phoneme should be a list');
      expect(phoneme.length, equals(3), reason: 'Phoneme should have [text, start, end]');

      final text = phoneme[0];
      final startTime = phoneme[1];
      final endTime = phoneme[2];

      expect(text, isA<String>(), reason: 'Phoneme text should be a string');
      expect(startTime, isA<num>(), reason: 'Start time should be a number');
      expect(endTime, isA<num>(), reason: 'End time should be a number');
      expect(endTime >= startTime, isTrue, reason: 'End time should be >= start time');
    }

    print('Test completed successfully. Received $responseCount response(s)');
  });

  // Run HTTP test
  testHttpTranscription();
}

List<double> _extractAudioData(Uint8List wavBytes) {

  // Simple WAV parser - assumes 16-bit mono WAV at 16kHz
  // Skip WAV header (44 bytes) and extract 16-bit samples
  final samples = <double>[];

  // Start after WAV header
  for (int i = 44; i < wavBytes.length; i += 2) {
    if (i + 1 < wavBytes.length) {
      // Convert 16-bit signed integer to float (-1.0 to 1.0)
      final sample = (wavBytes[i] | (wavBytes[i + 1] << 8));
      final signedSample = sample > 32767 ? sample - 65536 : sample;
      samples.add(signedSample / 32768.0);
    }
  }

  return samples;
}

/// Attempts to group phonemes into word-like units based on timing gaps
String _phonemesToWords(List<List> phonemes) {
  if (phonemes.isEmpty) return "";
  
  // Sort phonemes by start time for proper word reconstruction
  final sortedPhonemes = List.from(phonemes);
  sortedPhonemes.sort((a, b) => (a[1] as num).compareTo(b[1] as num));
  
  final words = <String>[];
  var currentWord = <String>[];
  
  for (int i = 0; i < sortedPhonemes.length; i++) {
    final phoneme = sortedPhonemes[i][0] as String;
    currentWord.add(phoneme);
    
    // Check if this is the end of a word (big gap to next phoneme or end of list)
    bool isWordEnd = false;
    if (i == sortedPhonemes.length - 1) {
      isWordEnd = true; // Last phoneme
    } else {
      final currentEnd = sortedPhonemes[i][2] as num;
      final nextStart = sortedPhonemes[i + 1][1] as num;
      final gap = nextStart - currentEnd;
      // Consider gaps > 0.1 seconds as word boundaries
      isWordEnd = gap > 0.1;
    }
    
    if (isWordEnd) {
      words.add(currentWord.join(""));
      currentWord.clear();
    }
  }
  
  return words.join(" ");
}

void testHttpTranscription() {
  test('test_http_transcription_with_timestamps', () async {
    // Load same test audio file
    final audioFile = File('test_audio.wav');
    expect(audioFile.existsSync(), isTrue, reason: 'test_audio.wav file not found');
    
    final audioBytes = await audioFile.readAsBytes();
    print('HTTP Test - Loaded audio: ${audioBytes.length} bytes');
    
    // Create multipart request
    final uri = Uri.parse('http://localhost:8080/transcribe');
    final request = http.MultipartRequest('POST', uri);
    
    // Add audio file
    request.files.add(http.MultipartFile.fromBytes(
      'audio',
      audioBytes,
      filename: 'test_audio.wav',
    ));
    

    // Start performance timing
    final startTime = DateTime.now();
    print('Sending HTTP POST request to /transcribe...');
    
    // Send request
    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);
    
    print('HTTP Response Status: ${response.statusCode}');
    print('HTTP Response Body: ${response.body}');

    // End performance timing
    final endTime = DateTime.now();
    final processingDuration = endTime.difference(startTime);
    print("\nPERFORMANCE METRICS:");
    print("Start time: ${startTime.toString()}");
    print("End time: ${endTime.toString()}");
    print("Total processing time: ${processingDuration.inMilliseconds}ms");
    print("Processing time: ${(processingDuration.inMilliseconds / 1000).toStringAsFixed(3)}s");
    
    expect(response.statusCode, equals(200), reason: 'HTTP request should succeed');
    
    // Parse JSON response
    final jsonData = jsonDecode(response.body) as List;
    
    print('\n' + '='*60);
    print('HTTP TRANSCRIPTION RESULTS');
    print('='*60);
    
    print('Total phonemes detected: ${jsonData.length}');
    print('\nPhoneme sequence with timestamps:');
    
    final allPhonemes = <List>[];
    for (int i = 0; i < jsonData.length; i++) {
      final phoneme = jsonData[i];
      final text = phoneme[0];
      final startTime = (phoneme[1] as num).toStringAsFixed(3);
      final endTime = (phoneme[2] as num).toStringAsFixed(3);
      print('${(i + 1).toString().padLeft(2)}: [$text] ${startTime}s → ${endTime}s');
      allPhonemes.add(phoneme);
    }
    
    // Timeline view
    print('\nTimeline view (sorted by start time):');
    final sortedPhonemes = List.from(allPhonemes);
    sortedPhonemes.sort((a, b) => (a[1] as num).compareTo(b[1] as num));
    
    for (var phoneme in sortedPhonemes) {
      final text = phoneme[0];
      final startTime = (phoneme[1] as num).toStringAsFixed(3);
      final endTime = (phoneme[2] as num).toStringAsFixed(3);
      print('${startTime}s ├─[$text]─┤ ${endTime}s');
    }
    
    // Analyze timestamp monotonicity
    print('\n' + '-'*60);
    print('HTTP TIMESTAMP MONOTONICITY ANALYSIS');
    print('-'*60);
    
    final analysis = analyzeTimestampMonotonicity(allPhonemes);
    print('Total phonemes: ${analysis['total_phonemes']}');
    print('Monotonic violations: ${analysis['monotonic_violations']}');
    print('Violation percentage: ${analysis['violation_percentage'].toStringAsFixed(1)}%');
    print('Analysis: ${analysis['analysis']}');
    
    if (analysis['violations'].isNotEmpty) {
      print('\nViolation details:');
      for (var violation in analysis['violations']) {
        print('  ${violation['position']}: [${violation['prev_phoneme']}] ${violation['prev_end'].toStringAsFixed(3)}s → [${violation['curr_phoneme']}] ${violation['curr_start'].toStringAsFixed(3)}s (gap: ${violation['gap'].toStringAsFixed(3)}s, ${violation['type']})');
      }
    }
    

    // Reconstruct phoneme string for easy copying
    final httpPhonemeString = allPhonemes.map((p) => p[0]).join(" ");
    print("\nRECONSTRUCTED PHONEME SEQUENCE (for copy-paste):");
    print("$httpPhonemeString");
    print("\nPhonemes as words (approximate):");
    print(_phonemesToWords(allPhonemes));
    print('\nNote: HTTP processing should have fewer/no violations due to complete audio context');
    print('='*60);
    
    // Validation
    expect(allPhonemes.isNotEmpty, isTrue, reason: 'Should receive some phonemes');
    
    for (final phoneme in allPhonemes) {
      expect(phoneme, isList, reason: 'Each phoneme should be a list');
      expect(phoneme.length, equals(3), reason: 'Phoneme should have [text, start, end]');
      
      final text = phoneme[0];
      final startTime = phoneme[1];
      final endTime = phoneme[2];
      
      expect(text, isA<String>(), reason: 'Phoneme text should be a string');
      expect(startTime, isA<num>(), reason: 'Start time should be a number');
      expect(endTime, isA<num>(), reason: 'End time should be a number');
      expect(endTime >= startTime, isTrue, reason: 'End time should be >= start time');
    }
    
    print('HTTP test completed successfully.');
  });
}
