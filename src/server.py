from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS, cross_origin
from flask_sock import Sock
import torch
from transformers import AutoProcessor, AutoModelForCTC
import numpy as np
import os
import io, base64
import json

from feedback import (
    score_words_cer,
    score_words_wfed,
    phoneme_written_feedback,
    user_phonetic_errors,
)
import json

from phoneme_utils import phrase_bounds, ALL_MAPPINGS
import soundfile as sf
from scipy.io import wavfile

DEBUG = True

# Constants
SAMPLE_RATE = 16000
NUM_SECONDS_PER_CHUNK = 2
NUM_CHUNKS_ACCUMULATED = 5

# Initialize Flask app
app = Flask(__name__)
cors = CORS(app)  # allow CORS for all domains on all routes.
app.config["CORS_HEADERS"] = "Content-Type"
sock = Sock(app)

# Load Wav2Vec2 model
model_id = "KoelLabs/xlsr-english-01"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCTC.from_pretrained(model_id)

model_vocab_json = os.path.join(os.path.dirname(__file__), "model_vocab_feedback.json")


def transcribe_audio(audio: np.ndarray) -> list[str]:
    """
    Transcribe audio and return both transcription and timestamp information.
    """
    transcription, _, _ = _run_inference(audio, model, processor)
    return transcription


def transcribe_timestamped(audio):
    transcription, duration_sec, predicted_ids = _run_inference(audio, model, processor)

    ids_w_time = [
        (i / len(predicted_ids) * duration_sec, _id)
        for i, _id in enumerate(predicted_ids)
    ]

    current_phoneme_id = processor.tokenizer.pad_token_id
    current_start_time = 0
    phonemes_with_time = []
    for time, _id in ids_w_time:
        if current_phoneme_id != _id:
            if current_phoneme_id != processor.tokenizer.pad_token_id:
                phonemes_with_time.append(
                    (
                        processor.decode(current_phoneme_id),
                        current_start_time,
                        time,
                    )
                )
            current_start_time = time
            current_phoneme_id = _id

    return transcription, phonemes_with_time


def _run_inference(audio, model, processor):
    input_values = (
        processor(
            audio,
            sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        .input_values.type(torch.float32)
        .to(model.device)
    )
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)[0].tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(predicted_ids)
    transcription = [
        ALL_MAPPINGS.get(t, t)
        for t in tokens
        if t not in processor.tokenizer.all_special_tokens
    ]
    duration_sec = input_values.shape[1] / processor.feature_extractor.sampling_rate
    return transcription, duration_sec, predicted_ids


def confidence_score(logits, predicted_ids) -> tuple[np.ndarray, float]:
    scores = torch.nn.functional.softmax(logits, dim=-1)
    pred_scores = scores.gather(-1, predicted_ids.unsqueeze(-1))[:, :, 0]
    mask = torch.logical_and(
        predicted_ids.not_equal(processor.tokenizer.word_delimiter_token_id),
        predicted_ids.not_equal(processor.tokenizer.pad_token_id),
    )

    character_scores = pred_scores.masked_select(mask)
    total_average = torch.sum(character_scores) / len(character_scores)
    return character_scores.numpy(), total_average.float().item()


def get_error_audio_clip(word_phone_pairings, timestamps, speech_audio, word_idx):

    start, end = phrase_bounds(word_phone_pairings, timestamps, word_idx)
    clip = speech_audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]

    # Encode the clip as a WAV file in-memory and then base64 so it can be JSON-serialised.
    buf = io.BytesIO()
    wavfile.write(buf, SAMPLE_RATE, clip)
    audio_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return audio_b64


def feedback(target, target_by_words, speech_audio, speech_phones, timestamps):
    if len(speech_phones) == 0:
        return []
    # retrieve main info
    user_phonetic_errors_dict = user_phonetic_errors(
        target, target_by_words, speech_phones, topk=3
    )
    # load all phoneme feedback
    all_phoneme_feedback = phoneme_written_feedback(target, speech_phones)
    feedback_items = []
    for phoneme, error_info in user_phonetic_errors_dict.items():
        num_mistakes, which_words, mistake_severities, phoneme_spoken_as, score = (
            error_info
        )
        phoneme_feedback = all_phoneme_feedback.get(phoneme)
        target_phoneme_spelling = phoneme_feedback["phonetic spelling"]
        target_phoneme_explanation = phoneme_feedback["explanation"]
        # target_phoneme_video = phoneme_feedback["video"] # NOTE: this can be added once we create a page of all audio trancripts for videos
        # target_phoneme_audio = phoneme_feedback["audio"] # NOTE: this can be added once we create a page of all audio trancripts for videos
        speech_phoneme_words = target_by_words[which_words]
        speech_phoneme_spelling = phoneme_feedback["phonetic spelling"]
        speech_phoneme_audio = get_error_audio_clip(
            target_by_words, timestamps, speech_audio, which_words
        )

        feedback_item = [
            target_phoneme_spelling,
            target_phoneme_explanation,
            # target_phoneme_video,
            # target_phoneme_audio = None,
            speech_phoneme_words,
            speech_phoneme_spelling,
            speech_phoneme_audio,
        ]
        feedback_items.append(feedback_item)

    return feedback_items


def get_user_word_audio_clip(speech_word, timestamps, speech_audio):

    first_phoneme = speech_word[0]
    last_phoneme = speech_word[-1]

    word_start_time = int(float(timestamps[first_phoneme][0]) * SAMPLE_RATE)
    word_end_time = int(float(timestamps[last_phoneme][1]) * SAMPLE_RATE)

    return speech_audio[word_start_time:word_end_time]


# server /
@app.route("/")
@cross_origin()
def index():
    return send_from_directory("static", "index.html")


# serve static files
@app.route("/<path:path>")
@cross_origin()
def send_static(path):
    return send_from_directory("static", path)


@app.route("/user_phonetic_errors", methods=["GET"])
@cross_origin()
def get_user_phonetic_errors():

    target = json.loads(request.args.get("target", "[]"))
    target_by_word = json.loads(request.args.get("tbw", "[]"))
    speech = json.loads(request.args.get("speech", "[]"))

    result = user_phonetic_errors(target, target_by_word, speech)

    result_sorted_freq = sorted(result.items(), key=lambda x: x[1][0], reverse=True)
    # Convert sets to lists for JSON serialization
    serializable_result = {}
    for phoneme, (count, words, severities, spoken_as, score) in result_sorted_freq:
        serializable_result[phoneme] = [
            count,
            list(words),
            severities,
            list(spoken_as),
            score,
        ]

    return jsonify(serializable_result)


@app.route("/phoneme_written_feedback", methods=["GET"])
@cross_origin()
# This function takes in the target and speech and returns a dictionary
# of phoneme: {explanation, phonetic-spelling} for ALL phonemes in the target
# and speech.
def get_phoneme_written_feedback():
    try:
        target = json.loads(request.args.get("target", "[]"))
        speech = json.loads(request.args.get("speech", "[]"))

        result = phoneme_written_feedback(target, speech)
        return jsonify(result)
    except Exception as e:
        return jsonify({"server error from get_phoneme_written_feedback": str(e)}), 500


# REST endpoint
@app.route("/score_words_cer", methods=["GET"])
@cross_origin()
def get_score_words_cer():
    try:
        target = json.loads(request.args.get("target", "[]"))
        target_by_word = json.loads(request.args.get("tbw") or "null")
        speech = json.loads(request.args.get("speech", "[]"))
        if not speech:
            return jsonify([[[word, seq, "", 0] for word, seq in target_by_word], 0])
        word_scores = score_words_cer(target, target_by_word, speech)
        return jsonify(word_scores)
    except Exception as e:
        return jsonify({"server error from get_score_words_cer": str(e)}), 500


@app.route("/score_words_wfed", methods=["GET"])
@cross_origin()
def get_score_words_wfed():
    try:
        target = json.loads(request.args.get("target", "[]"))
        target_by_word = json.loads(request.args.get("tbw") or "null")
        speech = json.loads(request.args.get("speech", "[]"))
        if not speech:
            return jsonify([[[word, seq, "", 0] for word, seq in target_by_word], 0])
        word_scores = score_words_cer(target, target_by_word, speech)
        return jsonify(word_scores)
    except Exception as e:
        return jsonify({"server error from get_score_words_wfed": str(e)}), 500


@app.route("/transcribe", methods=["POST"])
@cross_origin()
def transcribe():
    """
    Endpoint for general timestamped transcription of audio files.
    Accepts audio files via multipart/form-data and returns phonemes with timestamps.
    """
    try:
        # Check if audio file is provided
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400
        
        # Read and process audio file using soundfile
        audio_bytes = audio_file.read()
        
        try:
            # Use soundfile to properly decode audio files (WAV, MP3, etc.)
            audio_io = io.BytesIO(audio_bytes)
            speech_audio, sample_rate = sf.read(audio_io, dtype='float32')
            
            # Ensure mono audio
            if speech_audio.ndim > 1:
                speech_audio = np.mean(speech_audio, axis=1)
            
            # Resample to 16kHz if necessary
            if sample_rate != SAMPLE_RATE:
                # Simple resampling - for production use librosa.resample
                from scipy import signal
                speech_audio = signal.resample(speech_audio, int(len(speech_audio) * SAMPLE_RATE / sample_rate))
            
        except Exception as decode_error:
            # Fallback: try to parse as raw float32 data (for WebSocket compatibility)
            try:
                speech_audio = np.frombuffer(audio_bytes, dtype=np.float32)
            except ValueError:
                return jsonify({"error": f"Unable to decode audio file: {str(decode_error)}"}), 400
        
        # Validate audio length
        if len(speech_audio) < SAMPLE_RATE * 0.1:
            return jsonify({"error": f"Audio too short: {len(speech_audio)} samples, minimum required: {int(SAMPLE_RATE * 0.1)}"}), 400
        
        # Transcribe with timestamps
        _, phonemes_with_time = transcribe_timestamped(speech_audio)
        
        # Convert to the required format: [["phoneme", start_time, end_time], ...]
        result = [[phoneme, start_time, end_time] for phoneme, start_time, end_time in phonemes_with_time]
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/feedback", methods=["POST"])
@cross_origin()
def get_feedback():
    try:
        target = request.form["target"].strip()
        target_by_words = json.loads(request.form["tbw"])
        audio_bytes = request.files["audio"].read()
        
        try:
            # Use soundfile to properly decode audio files (WAV, MP3, etc.)
            audio_io = io.BytesIO(audio_bytes)
            speech_audio, sample_rate = sf.read(audio_io, dtype='float32')
            
            # Ensure mono audio
            if speech_audio.ndim > 1:
                speech_audio = np.mean(speech_audio, axis=1)
            
            # Resample to 16kHz if necessary
            if sample_rate != SAMPLE_RATE:
                # Simple resampling - for production use librosa.resample
                from scipy import signal
                speech_audio = signal.resample(speech_audio, int(len(speech_audio) * SAMPLE_RATE / sample_rate))
                
        except Exception as decode_error:
            # Fallback: try to parse as raw float32 data (for WebSocket compatibility)
            try:
                speech_audio = np.frombuffer(audio_bytes, dtype=np.float32)
            except ValueError:
                return jsonify({"error": f"Unable to decode audio file: {str(decode_error)}"}), 400

        if len(speech_audio) < SAMPLE_RATE * 0.1:
            raise ValueError(f"Audio too short: {len(speech_audio)} samples")

        speech_phones, timestamps = transcribe_timestamped(speech_audio)
        out = feedback(
            target,
            target_by_words,
            speech_audio,
            speech_phones,
            timestamps,  # per-phoneme timing info
        )
        return jsonify(out)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@sock.route("/stream")
def stream(ws):
    buffer = b""  # Buffer to hold audio chunks

    full_transcription = []
    combined = np.array([], dtype=np.float32)
    num_chunks_accumulated = 0
    transcription = []

    while True:
        try:
            # Receive audio data from the client
            data = ws.receive()
            if data:
                buffer += data
                # Process chunks when buffer reaches certain size
                if (
                    len(buffer) // 4 > SAMPLE_RATE * NUM_SECONDS_PER_CHUNK
                ):  # Adjust size for chunk processing
                    num_chunks_accumulated += 1

                    audio = np.frombuffer(buffer, dtype=np.float32)
                    combined = np.concatenate([combined, audio])

                    if num_chunks_accumulated < NUM_CHUNKS_ACCUMULATED:
                        # transcribe_audio returns a list, so extend directly
                        new_transcription = transcribe_audio(audio)
                        transcription.extend(new_transcription)
                        ws.send(json.dumps(full_transcription + transcription))
                    else:
                        new_transcription = transcribe_audio(combined)
                        full_transcription.extend(new_transcription)
                        ws.send(json.dumps(full_transcription))
                        combined = np.array([], dtype=np.float32)
                        num_chunks_accumulated = 0
                        transcription = []

                    if DEBUG:
                        wavfile.write("audio.wav", SAMPLE_RATE, audio)
                        wavfile.write("combined.wav", SAMPLE_RATE, combined)

                    buffer = b""  # Clear the buffer
        except Exception as e:
            print(f"Error: {e}")
            print(f"Line: {e.__traceback__.tb_lineno if e.__traceback__ else -1}")
            break


@sock.route("/stream_timestamped")
def stream_timestamped(ws):
    buffer = b""  # Buffer to hold audio chunks

    full_transcription = []
    combined = np.array([], dtype=np.float32)
    num_chunks_accumulated = 0
    transcription = []
    
    # Track what has been sent to client to enable incremental responses
    previously_sent_count = 0

    while True:
        try:
            # Receive audio data from the client
            data = ws.receive()
            if data:
                buffer += data
                # Process chunks when buffer reaches certain size
                if (
                    len(buffer) // 4 > SAMPLE_RATE * NUM_SECONDS_PER_CHUNK
                ):  # Adjust size for chunk processing
                    num_chunks_accumulated += 1

                    audio = np.frombuffer(buffer, dtype=np.float32)
                    combined = np.concatenate([combined, audio])

                    if num_chunks_accumulated < NUM_CHUNKS_ACCUMULATED:
                        # transcribe_timestamped returns (transcription_text, phonemes_with_time)
                        _, new_phonemes_with_time = transcribe_timestamped(audio)
                        # Convert tuples to lists for JSON serialization: [[phoneme, start_time, end_time], ...]
                        new_transcription = [[phoneme, start_time, end_time] for phoneme, start_time, end_time in new_phonemes_with_time]
                        transcription.extend(new_transcription)
                        
                        # Send only newly detected phonemes (incremental response)
                        current_total = full_transcription + transcription
                        new_phonemes = current_total[previously_sent_count:]
                        ws.send(json.dumps(new_phonemes))
                        previously_sent_count = len(current_total)
                    else:
                        _, combined_phonemes_with_time = transcribe_timestamped(combined)
                        # Convert tuples to lists for JSON serialization: [[phoneme, start_time, end_time], ...]
                        new_transcription = [[phoneme, start_time, end_time] for phoneme, start_time, end_time in combined_phonemes_with_time]
                        full_transcription.extend(new_transcription)
                        
                        # Send only newly detected phonemes (incremental response)
                        new_phonemes = full_transcription[previously_sent_count:]
                        ws.send(json.dumps(new_phonemes))
                        previously_sent_count = len(full_transcription)
                        
                        combined = np.array([], dtype=np.float32)
                        num_chunks_accumulated = 0
                        transcription = []

                    if DEBUG:
                        wavfile.write("audio.wav", SAMPLE_RATE, audio)
                        wavfile.write("combined.wav", SAMPLE_RATE, combined)

                    buffer = b""  # Clear the buffer
        except Exception as e:
            print(f"Error: {e}")
            print(f"Line: {e.__traceback__.tb_lineno if e.__traceback__ else -1}")
            break


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
