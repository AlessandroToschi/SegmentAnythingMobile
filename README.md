# The Idea

<table>
    <tr>
        <td> 
            <img align="center" src="https://github.com/AlessandroToschi/SegmentAnythingMobile/assets/6044244/982478da-2109-49c1-93b9-bf4763757cc9" width=400/>
        </td>
        <td>
          During the Christmas holidays, I took the following photo and immediately thought that it would be cool to remove the people in the background as well as the wooden stick.

There are plenty of iOS apps that perform magic erasing but usually, it is a premium feature.

So, I challenged myself to implement it.

The first step is **segmentation**, we need to be able to isolate and identify the objects we want to remove. There are many segmentation models out there, the one I choose is **[Segment Anything by Meta AI Research](https://github.com/facebookresearch/segment-anything)**.

The plan was clear: download the model, make it run on my M1 Pro, convert it to CoreML, and plug it into an iOS demo app. **But the journey has been different.**  
        </td>
    </tr>
</table>

# Segment Anything

**Segment Anything** is essentially composed of an ***Image Encoder*** to extract image embeddings, a ***Prompt Encoder*** to encode image points (pixels), and a ***Mask Decoder*** that combines the image and points embeddings to create the mask and assign a probabilistic score.

![segment_anything](https://github.com/AlessandroToschi/SegmentAnythingMobile/assets/6044244/4e462118-1493-4567-a186-3d7f3d963e8b)

Segment Anything Model Architecture

Segment Anything pre-trained models are huge when compared to the size of an iOS app, and most of the weights are related to the image encoder. We have ViT-H (2.56 GB), ViT-L (1.25 GB), and ViT-B (375 MB): the first two are unlikely to be used in an app, maybe the last one but still pretty large.

Looking online, I found [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) as a viable alternative to be used in an app. It ensembles a smaller image encoder that has been trained using distillation from ViT-H, while retaining the same mask decoder and prompt encoder, weighing just 40 MB in total.

## Model Architecture

If you look at how the `SamPredictor` is implemented in [MobileSAM](https://github.com/ChaoningZhang/MobileSAM/blob/master/mobile_sam/predictor.py), we can isolate 5 stages:

### 1. Preprocessing

- Downsampling of the source image to fit the **Image Encoder** input size of 1024x1024.
    
    ```python
    # source: https://github.com/ChaoningZhang/MobileSAM/blob/c12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed/mobile_sam/utils/transforms.py#L63
    
    class ResizeLongestSize:
    	.....
    	def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
    	  target_size = self.get_preprocess_shape(
    			image.shape[2], 
    			image.shape[3], 
    			1024
    		)
    	  return F.interpolate(
    		  image, 
    			target_size, 
    			mode="bilinear", 
    			align_corners=False, 
    			antialias=True
    	  )
    	.....
    
    # source: https://github.com/ChaoningZhang/MobileSAM/blob/c12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed/mobile_sam/predictor.py#L56
    class SamPredictor:
    	.....
    	def __init__(
        self,
        sam_model: Sam,
      ) -> None:
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(1024)
    
      def set_image(
          self,
          image: np.ndarray,
      ) -> None:
    		.....
        input_image = self.transform.apply_image(image)
    		.....
    ```
    
- Permute the dimensions of the source image from HxWxC to CxHxW, by splitting each RGB pixel into three separate planes.
    
    ```python
    # source: https://github.com/ChaoningZhang/MobileSAM/blob/c12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed/mobile_sam/predictor.py#L58
    class SamPredictor:
    	.....
      def set_image(
          self,
          image: np.ndarray,
      ) -> None:
    		.....
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    		.....
    ```
    
- Normalize each pixel by subtracting a mean pixel value and dividing by a std pixel value
    
    ```python
    # source: https://github.com/ChaoningZhang/MobileSAM/blob/c12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed/mobile_sam/modeling/sam.py#L165
    
    class SAM(nn.Module):
    	.....
    	def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std
    	.....
    ```
    
- Pad the image to fill 1024x1024
    
    ```python
    # source: https://github.com/ChaoningZhang/MobileSAM/blob/c12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed/mobile_sam/modeling/sam.py#L165
    
    class SAM(nn.Module):
    	.....
    	def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        .....
    		# Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
    	.....
    ```
    

### 2. Image Encoder - Vision Transformer ViT

**Input: 3 x 1024 x 1024 (CxHxW) RGB image**
**Output: 1 x 256 x 64 x 64 image embeddings**

- [Usage](https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2FChaoningZhang%2FMobileSAM%2Fblob%2Fc12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed%2Fmobile_sam%2Fpredictor.py%23L62-L91&style=default&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on)
    
- [PyTorch Module Definition](https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2FChaoningZhang%2FMobileSAM%2Fblob%2Fc12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed%2Fmobile_sam%2Fmodeling%2Ftiny_vit_sam.py%23L462-L620&style=default&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on)
    

### 3. Prompt Encoder

**Input: N x 2**, where N is the number of points and two refers to X and Y dimensions.
**Output: N x 256 and 1 x 256 x 64 x 64**

The former is the s**parse positional embeddings of points**, while the latter is a learned **dense embedding** representing no mask, as reported in the [paper](https://arxiv.org/abs/2304.02643).

- [Usage](https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2FChaoningZhang%2FMobileSAM%2Fblob%2Fc12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed%2Fmobile_sam%2Fpredictor.py%23L170-L227&style=default&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on)
    
- [PyTorch Module Definition](https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2FChaoningZhang%2FMobileSAM%2Fblob%2Fc12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed%2Fmobile_sam%2Fmodeling%2Fprompt_encoder.py%23L7-L214&style=default&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on)
    

### 4. Mask Decoder

Input:

- Image Embeddings: 1 x 256 x 64 x 64
- Sparse point embeddings: N x 256
- Dense embeddings: 1 x 256 x 64 x 64
- Image Point Embeddings: 1 x 256 x 64 x 64, positional encoding with the shape of the Image Embeddings.

Output: 1 x 4 x 256 x 256 low-resolution masks and 1 x 4 IOU scores.

- [Usage](https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2FChaoningZhang%2FMobileSAM%2Fblob%2Fc12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed%2Fmobile_sam%2Fpredictor.py%23L170-L236&style=default&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on)
    
- [PyTorch Module Definition](https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2FChaoningZhang%2FMobileSAM%2Fblob%2Fc12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed%2Fmobile_sam%2Fmodeling%2Fmask_decoder.py%23L7-L149&style=default&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on)
    

### 5. Postprocessing

- [Upsample the masks to match the original image size using a bilinear filter.](https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2FChaoningZhang%2FMobileSAM%2Fblob%2Fc12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed%2Fmobile_sam%2Fmodeling%2Fsam.py%23L134-L163&style=default&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on)
    
- [Clamp the mask values into the [0, 1] range.](https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2FChaoningZhang%2FMobileSAM%2Fblob%2Fc12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed%2Fmobile_sam%2Fpredictor.py%23L239-L244&style=default&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on)
    

# Porting to CoreML

Converting a PyTorch model to CoreML might not be straightforward, especially if you’re using [**coremltools**](https://github.com/apple/coremltools) for the first time. First, you need to keep in mind that only Torch models can be converted, so all the Python codes before or after the model inference have to be removed and rewritten in Swift. Thus, the preprocessing and postprocessing operations have been written in Swift (Metal) and will be detailed in the next section.

I decided to extract three CoreML packages: the image encoder, the prompt encoder, and the mask decoder, using the following script:

```python
import torch
import numpy as np
import mobile_sam as msam
import coremltools as ct

def convert_image_encoder(model: msam.Sam, device: torch.device):
    input_tensor = torch.randn(1, 3, 1024, 1024).to(device)
    traced_image_encoder = torch.jit.trace(model.image_encoder, input_tensor)
    traced_image_encoder(input_tensor)
    
    coreml_model = ct.convert(
        traced_image_encoder,
        inputs=[ct.TensorType(name="image", shape=input_tensor.shape)],
        outputs=[ct.TensorType(name="imageEmbeddings")],
    )
    coreml_model.save("./coreml/ImageEncoder.mlpackage")

def convert_prompt_encoder(model: msam.Sam, device: torch.device):
    n = 2
    points = torch.randn(1, n, 2).to(device=device)
    labels = torch.ones(1, n).to(device=device)

    model.prompt_encoder.forward = model.prompt_encoder.coreml_forward

    traced_model = torch.jit.trace(model.prompt_encoder, (points, labels))
    traced_model(points, labels)

    r = ct.RangeDim(1, 100)

    coreml_model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="points", shape=(1, r, 2)),
            ct.TensorType(name="labels", shape=(1, r))
        ],
        outputs=[
            ct.TensorType(name="sparsePromptEmbeddings"),
            ct.TensorType(name="densePromptEmbeddings")
        ]
    )
    coreml_model.save(f"./coreml/PromptEncoder.mlpackage")

def convert_mask_decoder(model: msam.Sam, device: torch.device):
    n = 3
    t1 = torch.randn(1, 256, 64, 64).to(device=device)
    t2 = torch.randn(1, 256, 64, 64).to(device=device)
    t3 = torch.randn(1, n, 256).to(device=device)
    t4 = torch.randn(1, 256, 64, 64).to(device=device)

    traced_mask_decoder = torch.jit.trace(model.mask_decoder, (t1, t2, t3, t4))
    traced_mask_decoder(t1, t2, t3, t4)

    coreml_model = ct.convert(
        traced_mask_decoder,
        inputs=[
            ct.TensorType(name="imageEmbeddings", shape=t1.shape),
            ct.TensorType(name="imagePositionalEncoding", shape=t2.shape),
            ct.TensorType(name="sparsePromptEmbeddings", shape=(1, ct.RangeDim(1, 100), 256)),
            ct.TensorType(name="densePromptEmbeddings", shape=t4.shape),
        ],
        outputs=[
            ct.TensorType(name="masks"),
            ct.TensorType(name="iouPredictions"),
        ],
    )
    coreml_model.save("./coreml/MaskDecoder.mlpackage")

def main():
    if not torch.backends.mps.is_available():
        raise SystemExit("PyTorch MPS backend is not available.")
    
    device = torch.device("mps")

    model_type = "vit_t"
    model_checkpoint = "./weights/mobile_sam.pt"

    mobile_sam = msam.sam_model_registry[model_type](checkpoint=model_checkpoint)
    mobile_sam.to(device)
    mobile_sam.eval()

    convert_image_encoder(mobile_sam, device)
    convert_prompt_encoder(mobile_sam, device)
    convert_mask_decoder(mobile_sam, device)
```

To convert the model, I had to implement some minor changes to this model because some operations and operators were not available/broken during the conversion, you can refer to this [commit](https://github.com/AlessandroToschi/MobileSAM/commit/f0ca9bbc0f256afdc681894ea536421f678ace77#diff-dfb8d6b38ffdb06d7f95213488ac04ee1f9abe7ac61683a5e37d6464b1045274).

### Packages

- [ImageEncoder](https://github.com/AlessandroToschi/MobileSAM/tree/master/coreml/ImageEncoder.mlpackage): 14,1 MB
- [PromptEncoder](https://github.com/AlessandroToschi/MobileSAM/tree/master/coreml/PromptEncoder.mlpackage): 4,2 MB
- [MaskDecoder](https://github.com/AlessandroToschi/MobileSAM/tree/master/coreml/MaskDecoder.mlpackage): 8,2 MB
- [ImagePointEmbeddings](https://github.com/AlessandroToschi/MobileSAM/blob/master/coreml/image_points_embeddings.bin): 1 x 256 x 64 x 64 F32 tensor to be loaded and fed into the Mask Decoder.

# Model In Action

### Preprocessing

Given an RGB image texture, the [preprocessing](https://www.notion.so/Segment-Anything-iOS-fa414648e2da4ad6b8a63c48ae094db2?pvs=21) is performed using a Metal compute kernel.

The output is made of three planes, one per color channel, to be later fed into the image encoder.

- Metal
    
    ```cpp
    #include <metal_stdlib>
    #include "ImageProcessor-Bridging-Header.h"
    
    using namespace metal;
    
    constexpr sampler linear_sampler = sampler(filter::linear);
    
    struct PreprocessingInput {
      simd_float3 mean;
      simd_float3 std;
      simd_uint2 size;
      simd_uint2 padding;
    };
    
    kernel void preprocessing_kernel(texture2d<float, access::sample> texture [[ texture(0) ]],
                                     constant PreprocessingInput& input [[ buffer(0) ]],
                                     device float* rBuffer [[ buffer(1) ]],
                                     device float* gBuffer [[ buffer(2) ]],
                                     device float* bBuffer [[ buffer(3) ]],
                                     uint2 xy [[ thread_position_in_grid ]]
                                     ) {
      if (xy.x >= input.size.x || xy.y >= input.size.y) {
        return;
      }
      
      const float2 uv = float2(xy) / float2(input.size);
      float4 color = texture.sample(linear_sampler, uv);
      color.rgb = (color.rgb - input.mean) / input.std;
      
      const int index = xy.y * (input.size.x + input.padding.x) + xy.x + input.padding.y;
      
      rBuffer[index] = color.r;
      gBuffer[index] = color.g;
      bBuffer[index] = color.b;
    }
    ```
    
- Swift
    
    ```swift
    func preprocess(
        image: MTLTexture,
        commandQueue: MTLCommandQueue
      ) -> MLMultiArray {
        self.originalWidth = image.width
        self.originalHeight = image.height
        
        let scale = Double(self.inputSize) / Double(max(self.originalWidth, self.originalHeight))
        self.resizedWidth = Int(Double(self.originalWidth) * scale + 0.5)
        self.resizedHeight = Int(Double(self.originalHeight) * scale + 0.5)
        let paddingX = self.inputSize - self.resizedWidth
        let paddingY = self.inputSize - self.resizedHeight
        
        let channels = 3
        let bytesPerChannel = MemoryLayout<Float>.stride * self.inputSize * self.inputSize
        let bytesCount = channels * bytesPerChannel
        
        if self.inputBuffer == nil || self.inputBuffer.length != bytesCount {
          self.inputBuffer = self.device.makeBuffer(
            length: bytesCount,
            options: .storageModeShared
          )!
        }
        
        var preprocessingInput = PreprocessingInput(
          mean: self.mean,
          std: self.std,
          size: SIMD2<UInt32>(UInt32(self.resizedWidth), UInt32(self.resizedHeight)),
          padding: SIMD2<UInt32>(UInt32(paddingX), UInt32(paddingY))
        )
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        
        let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
        computeCommandEncoder.setComputePipelineState(self.preprocessComputePipelineState)
        computeCommandEncoder.setTexture(image, index: 0)
        computeCommandEncoder.setBytes(
          &preprocessingInput,
          length: MemoryLayout<PreprocessingInput>.stride,
          index: 0
        )
        computeCommandEncoder.setBuffer(
          self.inputBuffer,
          offset: 0,
          attributeStride: MemoryLayout<Float>.stride,
          index: 1
        )
        computeCommandEncoder.setBuffer(
          self.inputBuffer,
          offset: bytesPerChannel,
          attributeStride: MemoryLayout<Float>.stride,
          index: 2
        )
        computeCommandEncoder.setBuffer(
          self.inputBuffer,
          offset: 2 * bytesPerChannel,
          attributeStride: MemoryLayout<Float>.stride,
          index: 3
        )
        
        let threadgroupSize = MTLSize(
          width: self.preprocessComputePipelineState.threadExecutionWidth,
          height: self.preprocessComputePipelineState.maxTotalThreadsPerThreadgroup / self.preprocessComputePipelineState.threadExecutionWidth,
          depth: 1
        )
        
        if self.device.supportsFamily(.common3) {
          computeCommandEncoder.dispatchThreads(
            MTLSize(
              width: self.resizedWidth,
              height: self.resizedHeight,
              depth: 1
            ),
            threadsPerThreadgroup: threadgroupSize
          )
        } else {
          let gridSize = MTLSize(
            width: (self.resizedWidth + threadgroupSize.width - 1) / threadgroupSize.width,
            height: (self.resizedHeight + threadgroupSize.height - 1) / threadgroupSize.height,
            depth: 1
          )
          computeCommandEncoder.dispatchThreadgroups(
            gridSize,
            threadsPerThreadgroup: threadgroupSize
          )
        }
        
        computeCommandEncoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return try! MLMultiArray(
          dataPointer: self.inputBuffer.contents(),
          shape: [
            1,
            channels as NSNumber,
            self.inputSize as NSNumber,
            self.inputSize as NSNumber
          ],
          dataType: .float32,
          strides: [
            (channels * inputSize * inputSize) as NSNumber,
            (inputSize * inputSize) as NSNumber,
            inputSize as NSNumber,
            1
          ]
        )
      }
    ```
    

### Postprocessing

Given an 256 x 256 float mask, we’re upsampling using a linear filter up to the original H x W and clamped the value between 0 and 1.

In the end, I apply a Gaussian Blur to the scaled mask

- Metal
    
    ```cpp
    constexpr sampler linear_sampler = sampler(filter::linear);
    
    struct PostprocessingInput {
      simd_float2 scaleSizeFactor;
    };
    
    kernel void postprocessing_kernel(texture2d<float, access::sample> mask [[ texture(0) ]],
                                      texture2d<float, access::read_write> output [[ texture (1) ]],
                                      constant PostprocessingInput& input [[ buffer(0) ]],
                                      uint2 xy [[ thread_position_in_grid] ]) {
      
      if (xy.x >= output.get_width() || xy.y >= output.get_height()) {
        return;
      }
      
      const float2 uv = float2(xy) / float2(output.get_width(), output.get_height());
      const float2 mask_uv = uv * input.scaleSizeFactor;
      
      const float4 mask_value = 1.0f - clamp(mask.sample(linear_sampler, mask_uv), float4(0.0), float4(1.0));
      
      output.write(mask_value, xy);
    }
    ```
    
- Swift
    
    ```swift
    func postprocess(masks: MLMultiArray, commandQueue: MTLCommandQueue) -> [MTLTexture] {
        let scale = Float(self.outputSize) / Float(max(self.originalWidth, self.originalHeight))
        let scaledWidth = (Float(self.originalWidth) * scale).rounded()
        let scaledHeight = (Float(self.originalHeight) * scale).rounded()
        let scaleSizeFactor = SIMD2<Float>(
          scaledWidth / Float(self.outputSize),
          scaledHeight / Float(self.outputSize)
        )
        
        var outputMasks = [MTLTexture]()
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        
        let threadgroupSize = MTLSize(
          width: self.preprocessComputePipelineState.threadExecutionWidth,
          height: self.preprocessComputePipelineState.maxTotalThreadsPerThreadgroup / self.preprocessComputePipelineState.threadExecutionWidth,
          depth: 1
        )
        let gridSizeOrThreads: MTLSize
        
        if self.device.supportsFamily(.common3) {
          gridSizeOrThreads = MTLSize(
            width: self.originalWidth,
            height: self.originalHeight,
            depth: 1
          )
        } else {
          gridSizeOrThreads = MTLSize(
            width: (self.originalWidth + threadgroupSize.width - 1) / threadgroupSize.width,
            height: (self.originalHeight + threadgroupSize.height - 1) / threadgroupSize.height,
            depth: 1
          )
        }
        
        let outputMaskTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
          pixelFormat: .r32Float,
          width: self.originalWidth,
          height: self.originalHeight,
          mipmapped: false
        )
        outputMaskTextureDescriptor.usage = [.shaderRead, .shaderWrite]
        outputMaskTextureDescriptor.storageMode = .shared
        
        let maskTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
          pixelFormat: .r32Float,
          width: masks.shape[3].intValue,
          height: masks.shape[2].intValue,
          mipmapped: false
        )
        maskTextureDescriptor.storageMode = .shared
        
        var postprocessingInput = PostprocessingInput(scaleSizeFactor: scaleSizeFactor)
        
        let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
        computeCommandEncoder.setComputePipelineState(self.postprocessComputePipelineState)
        computeCommandEncoder.setBytes(
          &postprocessingInput,
          length: MemoryLayout<PostprocessingInput>.stride,
          index: 0
        )
        
        masks.withUnsafeMutableBytes {
          pointer,
          strides in
          
          let maskPointer = pointer.bindMemory(to: Float.self).baseAddress!
          let maskStride = strides[1]
          
          for maskIndex in 0 ..< masks.shape[1].intValue {
            let maskBuffer = self.device.makeBuffer(
              bytesNoCopy: maskPointer + maskIndex * maskStride,
              length: maskStride * MemoryLayout<Float>.stride
            )!
            let maskTexture = maskBuffer.makeTexture(
              descriptor: maskTextureDescriptor,
              offset: 0,
              bytesPerRow: strides[2] * MemoryLayout<Float>.stride
            )!
            let outputMaskTexture = self.device.makeTexture(descriptor: outputMaskTextureDescriptor)!
            
            computeCommandEncoder.setTexture(maskTexture, index: 0)
            computeCommandEncoder.setTexture(outputMaskTexture, index: 1)
            
            if self.device.supportsFamily(.common3) {
              computeCommandEncoder.dispatchThreads(
                gridSizeOrThreads,
                threadsPerThreadgroup: threadgroupSize
              )
            } else {
              computeCommandEncoder.dispatchThreadgroups(
                gridSizeOrThreads,
                threadsPerThreadgroup: threadgroupSize
              )
            }
            
            outputMasks.append(outputMaskTexture)
          }
        }
        
        computeCommandEncoder.endEncoding()
        
        let gaussianFilter = MPSImageGaussianBlur(device: self.device, sigma: 5)
        
        for maskIndex in 0 ..< outputMasks.count {
          gaussianFilter.encode(commandBuffer: commandBuffer, inPlaceTexture: &outputMasks[maskIndex])
        }
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
            
        return outputMasks
      }
    ```
    

### Everything Combined

You can find and use the code at the following [GitHub repo](https://github.com/AlessandroToschi/SegmentAnythingMobile):

```swift
import Foundation
import Metal
import CoreML

public struct Point: Equatable {
  public var x: Float
  public var y: Float
  public var label: Int
  
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

public class SegmentAnything {
  public let device: MTLDevice
  
  private let commandQueue: MTLCommandQueue
  private let imageProcessor: ImageProcessor
  
  private var width: Int!
  private var height: Int!
  
  private var imageEmbeddings: MLMultiArray!
  private var imageEncoder: ImageEncoder!
  
  private var promptEncoder: PromptEncoder!
  private var imagePointEmbeddings: MLMultiArray!
  private var denseEmbeddings: MLMultiArray!
  
  private var maskDecoder: MaskDecoder!
    
  public init(device: MTLDevice) {
    self.device = device
    self.commandQueue = device.makeCommandQueue()!
    self.imageProcessor = ImageProcessor(
      device: device,
      mean: SIMD3<Float>(123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0),
      std: SIMD3<Float>(58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0)
    )
  }
  
  public func load() {
    self.imageProcessor.load()
    
    let modelConfiguration = MLModelConfiguration()
    modelConfiguration.computeUnits = .cpuAndGPU
    self.imageEncoder = try! ImageEncoder(configuration: modelConfiguration)
    
    self.imagePointEmbeddings = self.imageProcessor.loadTensor(
      tensorName: "image_embeddings",
      shape: [1, 256, 64, 64]
    )
    
    self.promptEncoder = try! PromptEncoder()
    
    modelConfiguration.computeUnits = .all
    self.maskDecoder = try! MaskDecoder(configuration: modelConfiguration)
  }
  
  public func preprocess(image: MTLTexture) {
    self.width = image.width
    self.height = image.height
    
    let resizedImage = self.imageProcessor.preprocess(
      image: image,
      commandQueue: self.commandQueue
    )
    
    let imageEncoderInput = ImageEncoderInput(input: resizedImage)
    let imageEncoderOutput = try! self.imageEncoder.prediction(input: imageEncoderInput)
    
    self.imageEmbeddings = imageEncoderOutput.output
  }
  
  public func predictMasks(points: [Point]) -> [(MTLTexture, Float)] {
    let pEncoderOutput = try! self.promptEncoder.prediction(
      input: self.imageProcessor.mapPoints(points: points)
    )
    
    let maskDecoderInput = MaskDecoderInput(
      imageEmbeddings: self.imageEmbeddings,
      imagePointEmbeddings: self.imagePointEmbeddings,
      sparsePromptEmbeddings: pEncoderOutput.sparseEmbeddings,
      densePromptEmbeddings: pEncoderOutput.denseEmbeddings
    )
    let maskDecoderOutput = try! self.maskDecoder.prediction(input: maskDecoderInput)
    
    let masks = self.imageProcessor.postprocess(
      masks: maskDecoderOutput.masks,
      commandQueue: self.commandQueue
    )
    
    return zip(
      masks,
      maskDecoderOutput.iou_predictionsShapedArray.scalars
    ).reduce(into: [], { $0.append(($1.0, $1.1))})
  }
}
```



https://github.com/AlessandroToschi/SegmentAnythingMobile/assets/6044244/7fb6472d-0253-400e-b9af-0f77b25f6aa7
