//
//  PromptEncoder.swift
//  SegmentAnythingMobile
//
//  Created by Alessandro Toschi on 10/01/24.
//

import Foundation
import Accelerate
import CoreML

public struct Point {
  var x: Float
  var y: Float
  var label: Int
  
  public init(
    x: Float,
    y: Float,
    label: Int
  ) {
    self.x = x
    self.y = y
    self.label = label
  }
}

public class PromptEncoder {
  static let positionEmbeddingFeatures: Int = 128
  
  /// 2 x 128
  static let positionalEncodingGaussianMatrix: [Float] = [
    -0.482656, -0.663184, 0.061686, -1.332097, -1.039539, 1.688714, 0.337137, 0.505055,
     -0.881985, 0.255005, -0.872282, 0.260346, 0.704948, -1.151259, 1.008740, -2.611053,
     0.276497, 0.095957, -1.319740, 2.192285, 0.638170, -0.962199, 0.411754, -2.871207,
     -1.212828, 0.022061, -0.409153, 1.586176, 0.955536, -0.962376, 1.359160, 0.063467,
     -2.246242, -0.422021, 0.006252, -0.439328, 1.201703, -1.525892, 1.040158, 0.532553,
     -0.567084, 1.364590, -0.254203, -0.870005, -0.722990, -1.474212, 0.101698, -0.247063,
     -0.313442, -0.391422, 0.084941, 1.377405, 1.153794, -1.548540, 2.283692, 0.760264,
     0.732412, -0.635684, -0.144669, -0.312889, 1.838674, 0.405084, -0.140173, -1.127826,
     -0.257867, -1.040048, -0.784612, -0.178962, -0.920270, -1.108748, 0.307673, -0.044183,
     0.491188, -0.745214, 0.619445, -1.999570, -0.928440, -0.657773, -0.679376, -0.312635,
     -0.854773, -3.244931, 1.729567, -0.711671, 1.257346, -0.567466, -0.173250, -0.212522,
     1.620631, -1.564081, 0.649078, -1.418217, -1.780346, 1.237921, 0.705790, -0.657494,
     0.132715, 0.017687, -1.418223, 0.137719, -1.236982, -1.077979, 1.885404, -1.285900,
     -0.125837, -1.996761, -0.937175, 0.593497, -0.539382, -0.256685, 0.482014, 0.011664,
     -0.650408, -1.480392, 0.860906, -1.219430, 1.469703, -1.823487, -0.249063, -2.091400,
     1.169457, -0.614139, 0.559213, -0.666138, -1.095287, 0.127074, -0.293204, -0.111019,
     //<---------------------------------------------------------------------------------->
     1.119697, -0.098230, -0.612738, -1.279226, -0.746168, 0.156964, 0.138457, 0.745606,
     1.626641, -0.802707, -0.461896, 0.110707, -0.374194, -0.167859, -1.195152, -0.096515,
     -0.559263, 1.222541, 1.712459, 0.314771, -1.115746, 0.709834, 2.557372, -0.772992,
     -0.513007, 0.096198, -0.191071, -1.609740, 1.842144, -0.983048, -0.717913, 0.171054,
     -0.062765, 2.559865, -1.313698, -1.116751, 0.399336, 0.905582, 0.229134, -1.156219,
     1.319926, -1.069413, 2.086009, 0.490560, -1.101044, 0.722387, -0.810239, 2.004468,
     0.862463, -1.458444, 0.433252, 0.672065, 0.821312, -0.193445, -0.822496, 0.129961,
     0.962272, 1.450503, 1.749881, -1.227781, 1.143411, 0.406856, -0.198115, -1.475947,
     0.029823, -0.375736, 0.352293, 0.756294, 0.648893, 0.572397, 0.586783, 0.405391,
     -1.536682, 0.522072, 0.453242, -0.285735, -0.071637, -0.138265, 1.140437, -0.186806,
     1.065424, -0.030777, 0.430445, -0.073228, -0.968438, 0.653986, -1.908400, -1.058312,
     -0.429349, 2.269586, -1.383879, -0.827982, 0.082016, -1.648144, 1.052719, 0.310013,
     -0.592635, -0.417575, 1.560255, -1.676978, -0.240294, 0.318584, -1.965739, 0.118772,
     0.582972, -0.110455, 0.798026, -0.238875, 1.289701, 0.013225, -0.582016, 1.586071,
     -0.099253, 0.169269, -1.043711, -2.229320, 0.637234, 0.960266, -0.711578, -1.173404,
     -0.365608, -1.190365, -0.029625, 0.651816, 0.027668, -1.698194, 0.491347, 0.467691
  ]
  
  /// 1 x 256
  static let notAPointEmbedding: [Float] = [
    -0.143690, -0.074579, -0.023358, -0.045170, 0.247659, -0.035532, 0.030298, 0.001379,
     0.114080, 0.179291, 0.019487, -0.035942, 0.104958, -0.098466, 0.006069, -0.000614,
     -0.016358, 0.157884, 0.044330, -0.012797, 0.050783, 0.067352, -0.106144, 0.029403,
     -0.481777, -0.233779, -0.151571, -0.091199, -0.050169, 0.105958, 0.186438, 0.097241,
     0.013972, -0.120647, -0.025128, 0.228634, 0.147693, 0.311031, -0.372770, 0.072802,
     0.649393, -0.110773, 0.019957, -0.184443, 0.312765, 0.065026, 0.057015, 0.052679,
     0.385177, -0.069108, 0.024219, -0.380682, -0.008654, 0.115135, -0.101268, -0.125061,
     0.231411, -0.220954, -0.013096, 0.140605, -0.040438, -0.303819, 0.021909, 0.098862,
     -0.145781, -0.096034, 0.159563, -0.059214, 0.055015, 0.069827, -0.150121, 0.177692,
     0.209617, -0.023444, -0.008807, 0.006446, -0.189141, -0.122081, 0.075618, -0.087302,
     -0.228912, 0.023473, 0.157178, 0.281704, 0.354768, -0.173795, -0.005742, -0.052782,
     -0.064275, -0.000889, -0.125716, -0.482017, -0.351404, 0.074775, 0.247138, 0.148945,
     0.077673, -0.037981, -0.085022, 0.030898, 0.120913, 0.152114, 0.081732, 0.005515,
     -0.197956, 0.045737, 0.157239, -0.086875, -0.269169, 0.023196, 0.344845, -0.031080,
     -0.124894, 0.282481, -0.144523, -0.148306, -0.480521, -0.263848, -0.046629, 0.049838,
     0.296798, -0.164135, -0.077189, 0.224238, -0.178989, -0.042311, 0.165387, 0.174812,
     0.351932, 0.481001, -0.204945, 0.093721, -0.134118, 0.076065, 0.155065, 0.336494,
     0.128915, 0.224542, 0.144545, -0.066150, 0.083589, 0.453677, 0.269134, 0.137449,
     -0.177875, -0.011074, 0.437229, 0.017900, 0.113846, -0.242051, 0.197023, 0.030083,
     -0.104125, -0.023803, -0.234395, -0.220365, 0.014009, -0.039333, 0.262512, -0.045114,
     -0.020624, 0.072492, -0.144363, 0.065780, -0.370435, 0.015320, -0.157200, -0.281950,
     0.395174, -0.186192, 0.036399, -0.175405, -0.317790, -0.255324, 0.149983, -0.033873,
     -0.129878, 0.037641, 0.113133, 0.203252, 0.054953, -0.008816, -0.041678, -0.351285,
     -0.031078, -0.015388, 0.023301, -0.143843, -0.180854, -0.466276, -0.043415, -0.002800,
     -0.054509, 0.311908, 0.018014, 0.002880, 0.107823, -0.479953, -0.139939, -0.112276,
     0.091101, 0.335048, 0.236161, -0.009280, -0.036430, 0.198709, -0.100972, 0.051006,
     -0.039427, -0.043752, -0.167168, -0.307389, 0.100392, -0.109246, -0.025620, 0.010521,
     0.138467, -0.002999, -0.465927, -0.401200, 0.105219, -0.396212, 0.114808, -0.012057,
     0.118670, 0.099212, 0.086388, -0.004543, -0.161717, 0.322495, 0.145496, -0.171445,
     -0.019095, -0.010015, 0.098319, -0.057554, -0.266394, -0.042315, -0.243109, 0.001583,
     -0.006062, -0.098131, 0.032211, 0.014106, 0.336704, 0.206597, -0.198924, 0.196695,
     0.099056, 0.157444, -0.133866, 0.119109, -0.043700, 0.021047, -0.081539, 0.096629
  ]
  
  /// 1 x 256
  static let backgroundPointEmbedding: [Float] = [
    0.314467, -0.042345, -0.314670, 0.001687, 0.031153, 0.093388, 0.073366, -0.589758,
    0.026044, 0.135801, -0.254796, -0.045571, -0.179766, 0.085031, -0.064190, -0.004523,
    0.207926, 0.302846, 0.218184, -0.003611, 0.079051, -0.498594, 0.026716, 0.006191,
    0.072773, -0.080083, 0.083213, 0.094783, -0.036007, -0.020506, -0.166608, 0.148577,
    0.003918, -0.032400, 0.117349, -0.190319, -0.048129, -0.016907, 0.091191, -0.258790,
    0.015853, -0.075475, -0.015243, -0.059411, 0.019263, -0.398153, 0.034977, 0.017347,
    -0.072777, -0.057703, -0.032422, 0.069647, -0.196411, 0.081763, 0.038794, -0.323075,
    0.105353, -0.254971, -0.013694, 0.048198, -0.130996, -0.001095, 0.069839, -0.027076,
    -0.059853, 0.056577, 0.044969, 0.096052, -0.135313, 0.255577, 0.337485, -0.071054,
    -0.015587, 0.007013, 0.121100, 0.004159, -0.032221, -0.340445, -0.304404, -0.099573,
    -0.070468, 0.010260, -0.010416, 0.106301, 0.016011, -0.114264, 0.002035, -0.061992,
    -0.004742, -0.119928, -0.085462, -0.065721, -0.016952, 0.199446, 0.103580, -0.010434,
    -0.122271, -0.061471, -0.072963, -0.011489, -0.040034, 0.210760, -0.275661, -0.036122,
    -0.070937, 0.001149, 0.065259, 0.053077, 0.006411, 0.050433, -0.125309, 0.020417,
    0.300766, 0.114253, -0.169140, 0.030768, 0.003319, 0.070505, -0.131387, -0.156895,
    0.317553, 0.293435, 0.436127, 0.298064, -0.100207, 0.004907, 0.012927, 0.210157,
    -0.227723, -0.056784, -0.108609, 0.463774, -0.046490, -0.083350, 0.734133, 0.165660,
    0.041176, -0.192763, -0.597696, 0.234441, 0.457881, 0.057367, 0.167917, -0.044341,
    -0.036271, -0.084150, 0.113723, 0.001010, 0.311095, 0.180820, -0.156105, -0.030880,
    0.467812, 0.254662, 0.122340, -0.076034, 0.053246, -0.094367, -0.231955, 0.447092,
    -0.012417, 0.006560, 0.081264, -0.015897, 0.155909, -0.037971, -0.013944, -0.018354,
    -0.128613, 0.063284, 0.009794, -0.099746, 0.122930, 0.060326, -0.057037, 0.011971,
    -0.231942, -0.027108, 0.133270, -0.099647, 0.071280, -0.250341, -0.055742, -0.051924,
    0.078800, -0.023724, -0.006289, -0.319236, -0.086493, 0.110504, 0.277539, -0.210039,
    0.312390, 0.030570, -1.215565, 0.261317, 0.029546, 0.029723, -0.236571, 0.112352,
    0.007392, 0.239647, -0.065658, 0.007975, -0.121136, 0.009581, 0.094524, 0.068537,
    0.046586, 0.032537, 0.109796, -0.021793, 0.069906, -0.004350, -0.013594, 0.651610,
    -0.037753, -0.027848, 0.010027, -0.068152, 0.006198, -0.179061, -0.075734, -0.053689,
    -0.354891, 0.045118, 0.058525, -0.016724, -0.062713, -0.136154, 0.050999, 0.211648,
    -0.013596, 0.026524, 0.092742, -0.076395, 0.453479, 0.037442, 0.020937, 0.026643,
    0.124791, -0.083591, -0.919856, 0.138572, -0.058098, 0.031930, -0.570900, 0.163219,
    0.396815, -0.023959, 0.043649, 0.103690, 0.200380, -0.007794, 0.520561, 0.251410
  ]
  
  static let foregrounPointEmbedding: [Float] = [
    -0.200146, -0.118373, -0.201452, 0.062660, -0.037616, 0.021803, -0.038792, 0.063638,
     -0.054137, -0.107757, 0.094505, 0.048452, 0.166762, -0.162025, 0.098379, 0.000491,
     0.137012, -0.113668, -0.005626, -0.004732, -0.058325, -0.074845, 0.033761, 0.007192,
     -0.019987, -0.035145, -0.020297, 0.047103, -0.007504, 0.074337, 0.162113, 0.239087,
     0.002830, 0.055701, -0.032918, -0.131582, -0.029601, -0.108927, -0.048070, 0.030225,
     -0.080796, -0.030595, -0.010773, -0.218529, -0.048627, 0.050970, -0.088837, 0.002658,
     0.109050, -0.008007, 0.004511, 0.065002, 0.139010, 0.062534, 0.029768, 0.009965,
     0.208052, 0.021828, 0.024177, 0.107214, 0.018883, 0.104636, -0.074406, -0.033156,
     0.062372, 0.003570, 0.206003, 0.175205, 0.164829, 0.023656, 0.053100, -0.073634,
     0.013761, 0.055468, -0.000424, 0.000736, -0.028574, 0.042309, 0.122858, 0.137375,
     0.043009, -0.009136, 0.020972, 0.042014, -0.267456, -0.007735, -0.005288, -0.115985,
     -0.107235, -0.014383, -0.231824, -0.002530, 0.011711, 0.060813, -0.113559, 0.040990,
     0.103148, -0.077393, 0.056043, 0.002392, 0.101131, -0.061056, 0.055590, 0.031136,
     0.090453, 0.000248, -0.019589, -0.147316, -0.140596, -0.034552, 0.171749, 0.006374,
     -0.120701, -0.110215, 0.134095, 0.020904, 0.006190, 0.072194, 0.141325, -0.118365,
     -0.149596, 0.095586, -0.050955, -0.039685, 0.018026, 0.011560, 0.034260, -0.075086,
     0.062099, -0.066229, 0.085386, -0.042627, 0.079021, 0.029846, 0.009073, -0.179083,
     -0.067349, -0.174543, -0.069905, 0.364727, -0.052269, -0.053597, -0.006654, -0.089969,
     0.016697, 0.167254, -0.077618, 0.004147, -0.002422, 0.063941, -0.179484, -0.022300,
     -0.078487, 0.361765, 0.194850, -0.086732, 0.044144, 0.179268, 0.112823, 0.231983,
     0.009072, -0.030715, 0.031055, -0.040276, -0.031072, 0.112072, 0.111688, 0.084725,
     0.010701, 0.204707, -0.017791, -0.081269, -0.076827, -0.073367, -0.297088, 0.022595,
     0.024512, -0.048758, -0.057222, 0.223875, -0.353748, -0.014054, 0.068916, -0.031851,
     -0.061466, -0.049848, -0.017531, -0.007198, 0.096946, -0.256081, 0.349682, 0.002869,
     0.339238, 0.075160, 0.034086, -0.173943, -0.052403, 0.130227, 0.004313, 0.215663,
     -0.012518, -0.089275, 0.211540, 0.007096, 0.035650, -0.167236, -0.161857, 0.026614,
     -0.016688, 0.055885, -0.045137, -0.047713, -0.275129, -0.133915, 0.010801, 0.148446,
     0.069397, 0.024115, 0.265771, 0.050592, -0.034084, 0.092289, 0.297134, -0.184204,
     -0.136499, 0.023787, 0.025361, -0.012054, 0.024112, 0.125122, -0.031824, 0.068978,
     0.018820, -0.009434, -0.006604, -0.020713, -0.016228, 0.340744, 0.338929, -0.006567,
     -0.125107, -0.091327, 0.099463, 0.139341, -0.080701, -0.158803, -0.111486, -0.175969,
     -0.097786, -0.136248, 0.109075, 0.193249, 0.108397, -0.004375, 0.017500, 0.039620
  ]
  
  public init() {
    
  }
  
  public func pointEmbeddings(points: [Point], width: Int, height: Int) -> MLMultiArray {
    // N x 2
    let n = points.count
    var coordinates = [Float](repeating: 0.0, count: 2 * n)
    for (i, point) in points.enumerated() {
      coordinates[i * 2] = 2.0 * ((point.x + 0.5) / Float(width)) - 1.0
      coordinates[i * 2 + 1] = 2.0 * ((point.y + 0.5) / Float(height)) - 1.0
    }
    
    let positionalEncodingSize = n * PromptEncoder.positionEmbeddingFeatures
    var positionalEncoding = [Float](repeating: 0.0, count: positionalEncodingSize)
    
    vDSP_mmul(
      coordinates,
      1,
      PromptEncoder.positionalEncodingGaussianMatrix,
      1,
      &positionalEncoding,
      1,
      vDSP_Length(n),
      vDSP_Length(PromptEncoder.positionEmbeddingFeatures),
      2
    )
    
    var scalar: Float = 2.0 * .pi
    
    vDSP_vsmul(
      positionalEncoding,
      1,
      &scalar,
      &positionalEncoding,
      1,
      vDSP_Length(positionalEncodingSize)
    )
    
    
    let pointEmbeddingsSize = positionalEncodingSize * 2
    
    let pointEmbeddingsPointer = UnsafeMutablePointer<Float>.allocate(capacity: pointEmbeddingsSize)
    pointEmbeddingsPointer.initialize(to: 0.0)
    
    positionalEncoding.withUnsafeBytes {
      positionalEncodingRawBufferPointer in
      let positionalEncodingPointer = positionalEncodingRawBufferPointer.bindMemory(to: Float.self).baseAddress!
      var count = Int32(PromptEncoder.positionEmbeddingFeatures)
      for i in 0 ..< n {
        let sine = pointEmbeddingsPointer + 2 * i * PromptEncoder.positionEmbeddingFeatures
        let cosine = sine + PromptEncoder.positionEmbeddingFeatures
        
        vvsincosf(
          sine,
          cosine,
          positionalEncodingPointer + (i * PromptEncoder.positionEmbeddingFeatures),
          &count
        )
        
        switch points[i].label {
          case 1:
            vDSP_vadd(
              sine,
              1,
              PromptEncoder.foregrounPointEmbedding,
              1,
              sine,
              1,
              vDSP_Length(PromptEncoder.foregrounPointEmbedding.count)
            )
          case -1:
            vDSP_mmov(
              PromptEncoder.notAPointEmbedding,
              sine,
              vDSP_Length(PromptEncoder.notAPointEmbedding.count),
              1,
              vDSP_Length(PromptEncoder.notAPointEmbedding.count),
              vDSP_Length(PromptEncoder.notAPointEmbedding.count)
            )
          case 0:
            vDSP_vadd(
              sine,
              1,
              PromptEncoder.backgroundPointEmbedding,
              1,
              sine,
              1,
              vDSP_Length(PromptEncoder.backgroundPointEmbedding.count)
            )
          default:
            continue
        }
      }
    }
    
    let shape = [1, points.count, 2 * PromptEncoder.positionEmbeddingFeatures]
    let strides = [
      pointEmbeddingsSize,
      2 * PromptEncoder.positionEmbeddingFeatures,
      1
    ]
    
    return try! MLMultiArray(
      dataPointer: pointEmbeddingsPointer,
      shape: shape as [NSNumber],
      dataType: .float32,
      strides: strides as [NSNumber],
      deallocator: { $0.deallocate() }
    )
  }
  
  func imagePointEmbeddings() -> MLMultiArray {
    let shape = [1, 256, 64, 64]
    let count = shape.reduce(1, *)
    
    let denseEmbeddingsUrl = Bundle(for: Self.self).url(
      forResource: "dense_embeddings",
      withExtension: "bin"
    )!
    let denseEmbeddingsPointer = UnsafeMutablePointer<Float>.allocate(capacity: count)
    
    let denseEmbeddingsData = try! Data(contentsOf: denseEmbeddingsUrl)
    (denseEmbeddingsData as NSData).getBytes(denseEmbeddingsPointer, length: denseEmbeddingsData.count)
    assert(denseEmbeddingsData.count == count * MemoryLayout<Float>.stride)
    
    return try! MLMultiArray(
      dataPointer: denseEmbeddingsPointer,
      shape: shape as [NSNumber],
      dataType: .float32,
      strides: [1048576, 4096, 64, 1],
      deallocator: { $0.deallocate() }
    )
  }
  
  /// 1 x 256 x 64 x 64
  func denseEmbeddings() -> MLMultiArray {
    let shape = [1, 256, 64, 64]
    let count = shape.reduce(1, *)
    
    let denseEmbeddingsUrl = Bundle(for: Self.self).url(
      forResource: "dense_embeddings",
      withExtension: "bin"
    )!
    let denseEmbeddingsPointer = UnsafeMutablePointer<Float>.allocate(capacity: count)
    
    let denseEmbeddingsData = try! Data(contentsOf: denseEmbeddingsUrl)
    (denseEmbeddingsData as NSData).getBytes(denseEmbeddingsPointer, length: denseEmbeddingsData.count)
    assert(denseEmbeddingsData.count == count * MemoryLayout<Float>.stride)
    
    return try! MLMultiArray(
      dataPointer: denseEmbeddingsPointer,
      shape: shape as [NSNumber],
      dataType: .float32,
      strides: [1048576, 4096, 64, 1],
      deallocator: { $0.deallocate() }
    )
  }
}