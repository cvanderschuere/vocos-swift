# Vocos for Swift

Implementation of [Vocos](https://github.com/gemelo-ai/vocos) with the [MLX](https://github.com/ml-explore/mlx) framework in Swift. Vocos allows for high quality reconstruction of audio from Mel spectrograms.

## Installation

The `Vocos` Swift package can be built and run from Xcode or SwiftPM.

A pretrained model is available [on Huggingface](https://hf.co/lucasnewman/vocos-mel-24khz-mlx).

## Usage

```swift
import Vocos

// Load audio as an MLXArray
let audio = try AudioUtilities.loadAudioFile(url: ...)

// Reconstruct the audio from a Mel spectrogram
let vocos = try await Vocos.fromPretrained(repoId: "lucasnewman/vocos-mel-24khz-mlx")
let reconstructedAudio = vocos(audio)

// Save the reconstructed audio to a file.
try AudioUtilities.saveAudioFile(url: ..., samples: reconstructedAudio)
```

## Citations

```bibtex
@article{siuzdak2023vocos,
  title={Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis},
  author={Siuzdak, Hubert},
  journal={arXiv preprint arXiv:2306.00814},
  year={2023}
}
```

## License

The code in this repository is released under the MIT license as found in the
[LICENSE](LICENSE) file.
