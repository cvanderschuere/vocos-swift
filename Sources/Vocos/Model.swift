import Foundation
import Hub
import MLX
import MLXFFT
import MLXNN
import MLXRandom

// ConvNeXT blocks

open class GroupableConv1d: Module, UnaryLayer {
    public let weight: MLXArray
    public let bias: MLXArray?
    public let padding: Int
    public let groups: Int
    public let stride: Int
    
    convenience init(_ inputChannels: Int, _ outputChannels: Int, kernelSize: Int, padding: Int, groups: Int) {
        self.init(inputChannels: inputChannels, outputChannels: outputChannels, kernelSize: kernelSize, padding: padding, groups: groups)
    }
    
    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        bias: Bool = true
    ) {
        let scale = sqrt(1 / Float(inputChannels * kernelSize))
        
        self.weight = uniform(
            low: -scale, high: scale, [outputChannels, kernelSize, inputChannels / groups]
        )
        self.bias = bias ? MLXArray.zeros([outputChannels]) : nil
        self.padding = padding
        self.stride = stride
        self.groups = groups
    }
    
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv1d(x, weight, stride: stride, padding: padding, groups: groups)
        if let bias {
            y = y + bias
        }
        return y
    }
}

class ConvNeXtBlock: Module {
    let norm: LayerNorm
    let dwconv: GroupableConv1d
    let pwconv1: Linear
    let act: GELU
    let pwconv2: Linear
    let gamma: MLXArray
    
    init(
        dim: Int,
        intermediateDim: Int,
        layerScaleInitValue: Float
    ) {
        self.dwconv = GroupableConv1d(dim, dim, kernelSize: 7, padding: 3, groups: dim)
        self.norm = LayerNorm(dimensions: dim, eps: 1e-6)
        self.pwconv1 = Linear(dim, intermediateDim)
        self.act = GELU()
        self.pwconv2 = Linear(intermediateDim, dim)
        self.gamma = layerScaleInitValue * MLXArray.ones([dim])
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var x = dwconv(x)
        x = norm(x)
        x = pwconv1(x)
        x = act(x)
        x = pwconv2(x)
        x = gamma * x
        x = residual + x
        return x
    }
}

// backbone

class VocosBackbone: Module {
    var embed: Conv1d
    var norm: LayerNorm
    var convnext: [ConvNeXtBlock]
    let final_layer_norm: LayerNorm
    
    init(
        inputChannels: Int,
        dim: Int,
        intermediateDim: Int,
        numLayers: Int,
        layerScaleInitValue: Float? = nil
    ) {
        self.embed = Conv1d(inputChannels: inputChannels, outputChannels: dim, kernelSize: 7, padding: 3)
        self.norm = LayerNorm(dimensions: dim, eps: 1e-6)
        let layerScaleInitValue = layerScaleInitValue ?? 1 / Float(numLayers)
        self.convnext = (0 ..< numLayers).map { _ in ConvNeXtBlock(dim: dim, intermediateDim: intermediateDim, layerScaleInitValue: layerScaleInitValue) }
        self.final_layer_norm = LayerNorm(dimensions: dim, eps: 1e-6)
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = embed(x)
        x = norm(x)
        for convBlock in convnext {
            x = convBlock(x)
        }
        x = final_layer_norm(x)
        return x
    }
}

// main class

public class Vocos: Module {
    enum VocosError: Error {
        case unableToLoadModel
    }
    
    let feature_extractor: MelSpectrogramFeatures
    let backbone: VocosBackbone
    let head: ISTFTHead
    
    init(feature_extractor: MelSpectrogramFeatures, backbone: VocosBackbone, head: ISTFTHead) {
        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head
    }
    
    public func decode(_ featuresInput: MLXArray) -> MLXArray {
        let x = backbone(featuresInput)
        return head(input: x)
    }
    
    public func callAsFunction(_ audioInput: MLXArray) -> MLXArray {
        let features = feature_extractor(x: audioInput)
        return decode(features)
    }
}

// pre-trained models

public extension Vocos {
    static func fromPretrained(repoId: String) async throws -> Vocos {
        let modelDirectoryURL = try await Hub.snapshot(from: repoId, matching: ["*.safetensors", "*.json"])
        return try fromPretrained(modelDirectoryURL: modelDirectoryURL)
    }
    
    static func fromPretrained(modelDirectoryURL: URL) throws -> Vocos {
        let modelURL = modelDirectoryURL.appendingPathComponent("model.safetensors")
        var modelWeights = try loadArrays(url: modelURL)
        
        let configURL = modelDirectoryURL.appendingPathComponent("config.json")
        let config = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any]
        guard let config else {
            throw VocosError.unableToLoadModel
        }
        
        let vocos = try fromConfig(config: config)
        
        var weights = [String: MLXArray]()
        for (key, value) in modelWeights {
            weights[key] = value
        }
        let parameters = ModuleParameters.unflattened(weights)
        try vocos.update(parameters: parameters, verify: [.all])
        
        return vocos
    }
    
    static func fromConfig(config: [String: Any]) throws -> Vocos {
        var featureExtractor: MelSpectrogramFeatures?
        
        if let featureExtractorConfig = config["feature_extractor"] as? [String: Any],
           let initArgs = featureExtractorConfig["init_args"] as? [String: Any],
           let sampleRate = initArgs["sample_rate"] as? Int,
           let nFFT = initArgs["n_fft"] as? Int,
           let hopLength = initArgs["hop_length"] as? Int,
           let nMels = initArgs["n_mels"] as? Int {
            featureExtractor = MelSpectrogramFeatures(
                sampleRate: sampleRate,
                nFFT: nFFT,
                hopLength: hopLength,
                nMels: nMels
            )
        }
        
        var backbone: VocosBackbone?
        
        if let backboneConfig = config["backbone"] as? [String: Any],
           let initArgs = backboneConfig["init_args"] as? [String: Any],
           let inputChannels = initArgs["input_channels"] as? Int,
           let dim = initArgs["dim"] as? Int,
           let intermediateDim = initArgs["intermediate_dim"] as? Int,
           let numLayers = initArgs["num_layers"] as? Int {
            backbone = VocosBackbone(
                inputChannels: inputChannels,
                dim: dim,
                intermediateDim: intermediateDim,
                numLayers: numLayers
            )
        }
        
        var head: ISTFTHead?
        
        if let headConfig = config["head"] as? [String: Any],
           let initArgs = headConfig["init_args"] as? [String: Any],
           let dim = initArgs["dim"] as? Int,
           let nFFT = initArgs["n_fft"] as? Int,
           let hopLength = initArgs["hop_length"] as? Int {
            head = ISTFTHead(
                dim: dim,
                nFFT: nFFT,
                hopLength: hopLength
            )
        }
        
        guard let featureExtractor, let backbone, let head else {
            throw VocosError.unableToLoadModel
        }
        
        return Vocos(feature_extractor: featureExtractor, backbone: backbone, head: head)
    }
}
