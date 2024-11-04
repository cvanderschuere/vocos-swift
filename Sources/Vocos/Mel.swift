import Foundation
import MLX
import MLXFFT
import MLXNN
import MLXRandom

class MelSpectrogramFeatures: Module {
    let sampleRate: Int
    let nFFT: Int
    let hopLength: Int
    let nMels: Int
    let filtersT: MLXArray
    
    init(
        sampleRate: Int = 24000,
        nFFT: Int = 1024,
        hopLength: Int = 256,
        nMels: Int = 100
    ) {
        self.sampleRate = sampleRate
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.nMels = nMels
        self.filtersT = MelSpectrogramFeatures.melFilters(
            sampleRate: sampleRate,
            nFft: nFFT,
            nMels: nMels
        ).T
    }

    enum MelScale {
        case htk
        case slaney
    }

    static private func melFilters(
        sampleRate: Int,
        nFft: Int,
        nMels: Int,
        fMin: Float = 0.0,
        fMax: Float? = nil,
        norm: String? = nil,
        melScale: MelScale = .htk
    ) -> MLXArray {
        func hzToMel(freq: Float, melScale: MelScale) -> Float {
            if melScale == .htk {
                return 2595.0 * log10(1.0 + freq / 700.0)
            }

            let fMin: Float = 0.0
            let fSp: Float = 200.0 / 3

            var mels = (freq - fMin) / fSp
            let minLogHz: Float = 1000.0
            let minLogMel = (minLogHz - fMin) / fSp
            let logStep = Float(log(6.4) / 27.0)

            if freq >= minLogHz {
                mels = minLogMel + log(freq / minLogHz) / logStep
            }

            return mels
        }

        func melToHz(mels: MLXArray, melScale: MelScale) -> MLXArray {
            if melScale == .htk {
                return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
            }

            // TODO: implement this later

            return MLXArray(0)
        }

        let fMax = fMax ?? Float(sampleRate) / 2.0

        // generate frequency points

        let nFreqs = nFft / 2 + 1 // integer division
        let allFreqs = MLX.linspace(0, sampleRate / 2, count: nFreqs)

        // convert frequencies to mel and back to hz
        let m_min = hzToMel(freq: fMin, melScale: melScale)
        let m_max = hzToMel(freq: fMax, melScale: melScale)
        let m_pts = MLX.linspace(m_min, m_max, count: nMels + 2)
        let f_pts = melToHz(mels: m_pts, melScale: melScale)
    
        // compute slopes for filterbank

        let f_diff = f_pts[1...] - f_pts[0..<f_pts.count - 1]
        let slopes = MLX.expandedDimensions(f_pts, axis: 0) - MLX.expandedDimensions(allFreqs, axis: 1)

        // calculate overlapping triangular filters
        let downSlopes = (-slopes[0..., ..<(slopes.shape[1]-2)]) / f_diff[..<(f_diff.shape[0]-1)]
        let upSlopes = slopes[0..., 2...] / f_diff[1...]
        let filterbank = MLX.maximum(
            MLX.zeros(like: downSlopes),
            MLX.minimum(downSlopes, upSlopes)
        )

        if norm == "slaney" {
            let enorm = 2.0 / (f_pts[2..<nMels + 2] - f_pts[0..<nMels])
            filterbank *= MLX.expandedDimensions(enorm, axis: 0)
        }

        return filterbank.transposed()
    }

    
    func callAsFunction(x: MLXArray) -> MLXArray {
        logMelSpectrogram(audio: x, nMels: nMels, nFFT: nFFT, hopLength: hopLength)
    }

    func stft(x: MLXArray, window: MLXArray, nperseg: Int, noverlap: Int? = nil, nfft: Int? = nil) -> MLXArray {
        let nfft = nfft ?? nperseg
        let noverlap = noverlap ?? nfft
        let padding = nperseg / 2
        let x = MLX.padded(x, width: IntOrPair(padding))
        let strides = [noverlap, 1]
        let t = (x.shape[0] - nperseg + noverlap) / noverlap
        let shape = [t, nfft]
        let stridedX = MLX.asStrided(x, shape, strides: strides)
        return MLXFFT.rfft(stridedX * window)
    }

    func logMelSpectrogram(audio: MLXArray, nMels: Int = 100, nFFT: Int = 1024, hopLength: Int = 256) -> MLXArray {
        let freqs = stft(x: audio, window: hanning(nFFT), nperseg: nFFT, noverlap: hopLength)
        let magnitudes = freqs[0 ..< freqs.shape[0] - 1].abs()

        let melSpec = MLX.matmul(magnitudes, filtersT)
        let logSpec = MLX.maximum(melSpec, 1e-5).log()
        return MLX.expandedDimensions(logSpec, axis: 0)
    }
}

// ISTFT head

class ISTFTHead: Module {
    let nFFT: Int
    let hopLength: Int
    
    let out: Linear
    
    init(dim: Int, nFFT: Int, hopLength: Int) {
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.out = Linear(dim, nFFT + 2)
    }
    
    func callAsFunction(input: MLXArray) -> MLXArray {
        let input = out(input).swappedAxes(1, 2)
        let split = input.split(parts: 2, axis: 1)
        let p = split[1]
        let mag = MLX.exp(split[0])
        
        let x = MLX.cos(p)
        let y = MLX.sin(p).asImaginary()
        let S = mag * (x + y)
        
        let audio = istft(
            x: S.squeezed(axis: 0).swappedAxes(0, 1),
            window: hanning(nFFT),
            nperseg: nFFT,
            noverlap: hopLength,
            nfft: nFFT
        )
        return audio
    }

    func istft(x: MLXArray, window: MLXArray, nperseg: Int = 256, noverlap: Int? = nil, nfft: Int? = nil) -> MLXArray {
        let nfft = nfft ?? nperseg
        let noverlap = noverlap ?? nfft
        let t = [(x.shape[0] - 1) * noverlap + nperseg]
        let reconstructed = MLX.zeros(t)
        let window_sum = MLX.zeros(t)
        
        for i in 0 ..< x.shape[0] {
            // inverse FFT of each frame
            let frame_time = MLXFFT.irfft(x[i])
            
            // get the position in the time-domain signal to add the frame
            let start = i * noverlap
            let end = start + nperseg
            
            // overlap-add the inverse transformed frame, scaled by the window
            reconstructed[start ..< end] = reconstructed[start ..< end] + (frame_time * window)
            window_sum[start ..< end] = window_sum[start ..< end] + window
        }
        
        // normalize by the sum of the window values
        return MLX.where(window_sum .!= 0, reconstructed / window_sum, reconstructed)
    }
}


// mel spec
func hanning(_ size: Int) -> MLXArray {
    let window = (0 ..< size).map { 0.5 * (1.0 - cos(2.0 * .pi * Double($0) / Double(size - 1))) }
    return MLXArray(converting: window)
}