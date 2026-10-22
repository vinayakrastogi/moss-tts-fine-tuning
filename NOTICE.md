# Source and attribution notes

- MOSS-TTS-Nano model, tokenizer, and their original runtime/training design are from [OpenMOSS](https://github.com/OpenMOSS/MOSS-TTS-Nano). Their Apache-2.0 license is preserved in `third_party/OPENMOSS_LICENSE`. Model weights are downloaded separately and are not included in this repository.
- The Indian-English speech data come from [auraCodes/indian-english-hindi-tts-60min](https://huggingface.co/datasets/auraCodes/indian-english-hindi-tts-60min). The dataset card states CC-BY 4.0; the original card and local split/provenance summary are in `training/indian_payment_tts/external/aura_indian_english/`. Raw clips are not included.
- The payment prompt manifest is project-generated text and has no paired payment-domain recordings. The training speech comes from educational lectures.

This project does not upload third-party checkpoints, source recordings, or generated voice samples. Review the source terms and clip-level attribution before redistributing those assets.
