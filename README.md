<div align="center">
<h1 style="border-bottom: none; margin-bottom: 0px ">PyannoteAI-enabled SpeechLLM for SA-ASR</h1>

[**Artem Fedorchenko**](https://scholar.google.com/citations?user=9dsO2ssAAAAJ&hl=en)

</div>

This repository contains code and instructions for using **PyannoteAI-enabled SpeechLLM** for Speaker-Attributed Automatic Speech Recognition (SA-ASR). The model integrates speaker embedding extraction capabilities from PyannoteAI with advanced speech recognition by OpenAI's Whisper to provide accurate transcriptions along with speaker attributions. In the attempt to lessen the number of trainable parameters, the provided implementation has yileded two crucial insights:

- 💎 A **unified modality adapter (projector)**: combines speech encodings with speaker embeddings before LLAMA autoregresive decoding.
- ✨ **LoRA-tuned LLAMA-3.2** : leverages Low-Rank Adaptation (LoRA) to fine-tune LLAMA-3.2 with extended vocabulary for the SA-ASR task.

⚡ **Versatility of the Approach** <br>
Because of the underlying LLM usage, the proposed method could be used to do a variety of tasks related simultaneously to speech such as:
- Transcript summarization
- Keyword extraction
- Topic segmentation
- Translation
- And more...

## Architecture Overview
<p align="center">
 <img src="assets/images/high-level-arch.png" alt="Architecture Overview" width="60%">
</p>


## Projector Module
<p align="center">
 <img src="assets/images/projector.png" alt="Architecture Overview" width="70%">
</p>

Both speaker projector and speech projector modules are implemented as Qformer networks.
