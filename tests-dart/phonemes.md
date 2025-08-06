# Phoneme Processing in KoelLabs/xlsr-english-01

This document details everything we have verified about how phonemes work in the KoelLabs speech recognition system, with exact code references and what remains unknown.

## Overview

The KoelLabs/xlsr-english-01 model processes audio and returns phonemes with timestamps. This document traces the complete phoneme processing pipeline from audio input to final JSON output.

## What We Know (With Sources)

### 1. Model Specification

**Model ID:** `KoelLabs/xlsr-english-01`
- **Source:** `/home/maoholden/Documents/Reuben1987AI-github/ai-avatars-private/ai-avatar/koel-labs/server/src/server.py:37`
- **Code:** `model_id = "KoelLabs/xlsr-english-01"`

**Model Type:** Wav2Vec2 with CTC (Connectionist Temporal Classification)
- **Source:** `/home/maoholden/Documents/Reuben1987AI-github/ai-avatars-private/ai-avatar/koel-labs/server/src/server.py:38-39`
- **Code:** 
  ```python
  processor = AutoProcessor.from_pretrained(model_id)
  model = AutoModelForCTC.from_pretrained(model_id)
  ```

### 2. Phoneme Vocabulary

**Total Documented Phonemes:** 79 IPA symbols
- **Source:** `/home/maoholden/Documents/Reuben1987AI-github/ai-avatars-private/ai-avatar/koel-labs/server/src/model_vocab_feedback.json`
- **Verification:** `grep -c '"phoneme":' model_vocab_feedback.json` returns 79
- **Python verification:** `len(json.load(open('model_vocab_feedback.json')))` returns 79

**Phoneme Format:** International Phonetic Alphabet (IPA) Unicode symbols
- **Examples from vocabulary file:**
  - Index 0: `a` (open front unrounded vowel)
  - Index 1: `æ` (near-open front unrounded vowel)  
  - Index 2: `ɔ` (open-mid back rounded vowel)
  - Index 77: `ə̥` (voiceless schwa)
  - Index 78: `ʉ` (close central rounded vowel)

**Complete Phoneme Categories Found:**
- Basic vowels: `a`, `e`, `i`, `o`, `u`
- IPA vowels: `æ`, `ɑ`, `ɒ`, `ɔ`, `ə`, `ɛ`, `ɜ`, `ɪ`, `ʊ`, `ʌ`
- Diphthongs: `aɪ`, `aʊ`, `eɪ`, `oʊ`, `ɔɪ`, `əʊ`
- Consonants: `b`, `d`, `f`, `g`, `h`, `j`, `k`, `l`, `m`, `n`, `p`, `r`, `s`, `t`, `v`, `w`, `x`, `z`
- IPA consonants: `ð`, `ŋ`, `ɡ`, `ɹ`, `ɾ`, `ʃ`, `ʒ`, `θ`
- Consonant clusters: `dʒ`, `tʃ`
- Aspirated variants: `kʰ`, `pʰ`, `sʰ`, `θʰ`
- Nasalized vowels: `æ̃`, `ɑ̃`, `aɪ̃`, `ĩ`, `ə̃`, `ɛ̃`, `oʊ̃`, `ɾ̃`, `ʊ̃`
- Syllabic consonants: `l̩`, `m̩`, `n̩`, `ŋ̍`
- Other variants: `ə̥`, `ɨ`, `ɝ`, `ɚ`, `ɣ`, `ɦ`, `ʔ`, `β`, `ʉ`

### 3. Phoneme Generation Process

**Step 1: Model Inference**
- **Source:** `server.py:79-101`
- **Process:** Audio → Wav2Vec2 model → token IDs → token strings

**Step 2: Token ID to Phoneme Conversion**
- **Source:** `server.py:94`
- **Code:** `tokens = processor.tokenizer.convert_ids_to_tokens(predicted_ids)`
- **Function:** Converts numerical token IDs to phoneme strings

**Step 3: Special Token Filtering**
- **Source:** `server.py:98`
- **Code:** `if t not in processor.tokenizer.all_special_tokens`
- **Purpose:** Removes non-phoneme tokens from output

**Step 4: Phoneme Mapping**
- **Source:** `server.py:96` and `phoneme_utils.py:34`
- **Code:** `ALL_MAPPINGS.get(t, t)`
- **Purpose:** Applies phoneme normalization mappings

### 4. Phoneme Mapping System

**Mapping Sources:** `/home/maoholden/Documents/Reuben1987AI-github/ai-avatars-private/ai-avatar/koel-labs/server/src/phoneme_utils.py:21-34`

**PANPHONE_MAPPINGS** (for panphon library compatibility):
```python
PANPHONE_MAPPINGS = {
    "ɝ": "ɜ˞",  # r-colored schwa (U+025D) -> schwa + r-coloring diacritic
    "ɚ": "ə˞",   # r-colored schwa variant
    "g": "ɡ",    # model vocab has correct ɡ but we add this since its hard to distinguish
}
```

**PHONEMES_TO_MASK** (simplification mappings):
```python
PHONEMES_TO_MASK = {
    "ʌ": "ə",    # Simplify similar phonemes
    "ɔ": "ɑ",
    "kʰ": "k",   # Remove aspiration
    "sʰ": "s",
}
```

**Combined Mappings:**
```python
ALL_MAPPINGS = {**PANPHONE_MAPPINGS, **PHONEMES_TO_MASK}
```

### 5. Timestamping Process

**Timing Calculation:**
- **Source:** `server.py:55-58`
- **Code:**
  ```python
  ids_w_time = [
      (i / len(predicted_ids) * duration_sec, _id)
      for i, _id in enumerate(predicted_ids)
  ]
  ```

**Phoneme Boundary Detection:**
- **Source:** `server.py:63-76`
- **Logic:** Groups consecutive identical token IDs into single phonemes with start/end times

**Final Format Generation:**
- **Source:** `server.py:66-72`
- **Code:**
  ```python
  phonemes_with_time.append(
      (
          processor.decode(current_phoneme_id),  # Phoneme string
          current_start_time,                    # Start time (float)
          time,                                  # End time (float)
      )
  )
  ```

### 6. Output Format

**JSON Structure:**
- **Source:** `server.py:314`
- **Code:** `result = [[phoneme, start_time, end_time] for phoneme, start_time, end_time in phonemes_with_time]`
- **Format:** `[["phoneme_symbol", start_time, end_time], ...]`

**Example Output:**
```json
[
  ["h", 0.123, 0.245],
  ["ɛ", 0.245, 0.367], 
  ["l", 0.367, 0.489],
  ["oʊ", 0.489, 0.712]
]
```

### 7. Special Token Handling

**Evidence of Special Tokens:**
- **References in code:**
  - `server.py:60`: `processor.tokenizer.pad_token_id`
  - `server.py:98`: `processor.tokenizer.all_special_tokens`
  - `server.py:108`: `processor.tokenizer.word_delimiter_token_id`

**Filtering Behavior:**
- **Source:** `server.py:98`
- **Code:** `if t not in processor.tokenizer.all_special_tokens`
- **Result:** Special tokens are excluded from transcription output

### 8. Panphon Integration

**Library Usage:**
- **Source:** `phoneme_utils.py:15-17`
- **Code:** 
  ```python
  import panphon
  ft = panphon.FeatureTable()
  IPA_SYMBOLS = [ipa for ipa, *_ in ft.segments]
  ```
- **Purpose:** Provides IPA feature analysis and distance calculations

## What We Don't Know

### 1. Model Vocabulary Details

**Total Vocabulary Size:**
- The actual tokenizer vocabulary size (beyond the 79 documented phonemes)
- Whether the model has additional tokens not documented in `model_vocab_feedback.json`

**Special Token Specifics:**
- Exact names of special tokens (e.g., `<pad>`, `<unk>`, `<s>`, `</s>`)
- Token IDs assigned to special tokens
- Total count of special tokens

**Reason:** Cached model files (`.cache/huggingface/hub/models--KoelLabs--xlsr-english-01/.no_exist/570c64263fb1440aacf7de0a91328183ee410ac1/tokenizer.json`) are empty in this codebase.

### 2. Model Training Details

**Training Data:**
- What dataset was used to train the model
- Language varieties represented in training data
- Phoneme annotation methodology

**Model Architecture:**
- Exact Wav2Vec2 configuration parameters
- Fine-tuning procedure from base model
- Performance metrics or benchmarks

### 3. Phoneme Selection Rationale

**Vocabulary Choices:**
- Why these specific 79 phonemes were chosen
- Relationship to standard phoneme inventories (like CMU ARPAbet)
- Coverage of English phoneme variants

**Regional Variants:**
- Whether phonemes represent specific English dialects
- How regional pronunciation differences are handled

### 4. Processing Limitations

**Timestamp Accuracy:**
- Precision limitations of frame-based timing
- How chunked processing affects timestamp reliability
- Validation of timestamp monotonicity assumptions

**Phoneme Detection:**
- Confidence thresholds for phoneme detection
- Handling of unclear or ambiguous sounds
- Error rates for different phoneme types

## Verification Methods Used

1. **Direct file inspection:** Counted entries in `model_vocab_feedback.json`
2. **Code analysis:** Traced function calls and data transformations
3. **Grep searches:** Found all references to tokenizer and phoneme processing
4. **JSON parsing:** Verified vocabulary structure and content

## Recommendations for Further Investigation

1. **Access live model:** Run actual inference to inspect tokenizer properties
2. **Model hub inspection:** Check HuggingFace model card for official documentation
3. **Token ID mapping:** Create test cases to understand ID-to-phoneme relationships
4. **Comparison study:** Compare output with other phoneme recognition systems

---

*This document reflects our current understanding based on static code analysis. Claims marked with sources can be verified by examining the referenced files and line numbers.*