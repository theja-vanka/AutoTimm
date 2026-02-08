# Mobile Deployment

Deploy AutoTimm models to iOS and Android devices using PyTorch Mobile.

## Overview

AutoTimm models can be deployed to mobile devices for on-device inference:

- **iOS** - iPhone, iPad applications
- **Android** - Phones, tablets, wearables
- **Privacy** - All processing on-device, no cloud required
- **Offline** - Works without internet connection
- **Low Latency** - Fast inference without network overhead
- **Cost Effective** - No server costs

## Prerequisites

### Model Export

First, export your model to TorchScript with mobile optimization:

```python
from autotimm import ImageClassifier
import torch

# Use lightweight backbone for mobile
model = ImageClassifier(
    backbone="mobilenet_v3_small",  # Or efficientnet_b0
    num_classes=1000,
    compile_model=False  # Disable torch.compile for mobile export
)
model.eval()

# Export
example_input = torch.randn(1, 3, 224, 224)
model.to_torchscript("mobile_model.pt", example_input=example_input)
```

## iOS Deployment

### Setup

1. **Install PyTorch Mobile** via CocoaPods:

Create `Podfile`:

```ruby
platform :ios, '12.0'

target 'YourApp' do
  use_frameworks!
  pod 'LibTorch-Lite', '~>1.13.0'
end
```

Install:

```bash
pod install
```

2. **Add Model to Xcode**:
   - Drag `mobile_model.pt` into your Xcode project
   - Ensure "Copy items if needed" is checked
   - Add to target

### Swift Implementation

#### Classifier.swift

```swift
import UIKit
import LibTorch_Lite

class ImageClassifier {
    private var module: TorchModule
    private let inputSize: Int = 224

    // ImageNet normalization
    private let mean: [Float] = [0.485, 0.456, 0.406]
    private let std: [Float] = [0.229, 0.224, 0.225]

    init(modelPath: String) throws {
        guard let module = TorchModule(fileAtPath: modelPath) else {
            throw ClassifierError.modelLoadFailed
        }
        self.module = module
    }

    func predict(image: UIImage) throws -> Prediction {
        // Preprocess
        guard let tensor = preprocess(image: image) else {
            throw ClassifierError.preprocessingFailed
        }

        // Run inference
        guard let output = module.predict(image: tensor) else {
            throw ClassifierError.inferenceFailed
        }

        // Parse output
        let scores = output.floatArray
        guard let (classId, confidence) = topPrediction(scores: scores) else {
            throw ClassifierError.invalidOutput
        }

        return Prediction(classId: classId, confidence: confidence)
    }

    func predictTop5(image: UIImage) throws -> [Prediction] {
        guard let tensor = preprocess(image: image) else {
            throw ClassifierError.preprocessingFailed
        }

        guard let output = module.predict(image: tensor) else {
            throw ClassifierError.inferenceFailed
        }

        let scores = output.floatArray
        return topKPredictions(scores: scores, k: 5)
    }

    private func preprocess(image: UIImage) -> UnsafeMutableRawPointer? {
        // Resize to 224x224
        guard let resized = image.resized(to: CGSize(
            width: inputSize,
            height: inputSize
        )) else {
            return nil
        }

        // Convert to pixel buffer
        guard let pixelBuffer = resized.pixelBuffer() else {
            return nil
        }

        // Normalize and convert to tensor format
        return normalize(pixelBuffer: pixelBuffer)
    }

    private func normalize(pixelBuffer: CVPixelBuffer) -> UnsafeMutableRawPointer? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return nil
        }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        // Allocate tensor memory [1, 3, 224, 224]
        let tensorSize = 1 * 3 * height * width
        let tensorData = UnsafeMutablePointer<Float>.allocate(
            capacity: tensorSize
        )

        // Convert RGBA to RGB and normalize
        for y in 0..<height {
            for x in 0..<width {
                let pixelOffset = y * bytesPerRow + x * 4
                let pixel = baseAddress.advanced(by: pixelOffset)
                    .assumingMemoryBound(to: UInt8.self)

                // Extract RGB values
                let r = Float(pixel[0]) / 255.0
                let g = Float(pixel[1]) / 255.0
                let b = Float(pixel[2]) / 255.0

                // Normalize and store in CHW format
                let idx = y * width + x
                tensorData[idx] = (r - mean[0]) / std[0]
                tensorData[height * width + idx] = (g - mean[1]) / std[1]
                tensorData[2 * height * width + idx] = (b - mean[2]) / std[2]
            }
        }

        return UnsafeMutableRawPointer(tensorData)
    }

    private func topPrediction(scores: [Float]) -> (Int, Float)? {
        guard let maxIndex = scores.enumerated().max(
            by: { $0.element < $1.element }
        ) else {
            return nil
        }

        return (maxIndex.offset, maxIndex.element)
    }

    private func topKPredictions(scores: [Float], k: Int) -> [Prediction] {
        let indexed = scores.enumerated().map { ($0.offset, $0.element) }
        let sorted = indexed.sorted { $0.1 > $1.1 }

        return sorted.prefix(k).map { idx, score in
            Prediction(classId: idx, confidence: score)
        }
    }
}

struct Prediction {
    let classId: Int
    let confidence: Float
}

enum ClassifierError: Error {
    case modelLoadFailed
    case preprocessingFailed
    case inferenceFailed
    case invalidOutput
}

// UIImage extensions
extension UIImage {
    func resized(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        defer { UIGraphicsEndImageContext() }
        draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }

    func pixelBuffer() -> CVPixelBuffer? {
        let width = Int(size.width)
        let height = Int(size.height)

        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32ARGB,
            attrs,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        )

        guard let ctx = context, let cgImage = self.cgImage else {
            return nil
        }

        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        return buffer
    }
}
```

#### ViewController.swift

```swift
import UIKit

class ViewController: UIViewController, UIImagePickerControllerDelegate,
                      UINavigationControllerDelegate {

    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var resultLabel: UILabel!
    @IBOutlet weak var confidenceLabel: UILabel!

    private var classifier: ImageClassifier?
    private let classNames = loadClassNames() // Load from file

    override func viewDidLoad() {
        super.viewDidLoad()

        // Load model
        guard let modelPath = Bundle.main.path(
            forResource: "mobile_model",
            ofType: "pt"
        ) else {
            showError("Model file not found")
            return
        }

        do {
            classifier = try ImageClassifier(modelPath: modelPath)
        } catch {
            showError("Failed to load model: \(error)")
        }
    }

    @IBAction func selectImageTapped(_ sender: UIButton) {
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = .photoLibrary
        present(picker, animated: true)
    }

    @IBAction func takePictureTapped(_ sender: UIButton) {
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
            showError("Camera not available")
            return
        }

        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = .camera
        present(picker, animated: true)
    }

    func imagePickerController(
        _ picker: UIImagePickerController,
        didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
    ) {
        picker.dismiss(animated: true)

        guard let image = info[.originalImage] as? UIImage else {
            return
        }

        imageView.image = image
        classifyImage(image)
    }

    private func classifyImage(_ image: UIImage) {
        guard let classifier = classifier else {
            showError("Classifier not loaded")
            return
        }

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let prediction = try classifier.predict(image: image)

                DispatchQueue.main.async {
                    self.showResult(prediction: prediction)
                }
            } catch {
                DispatchQueue.main.async {
                    self.showError("Classification failed: \(error)")
                }
            }
        }
    }

    private func showResult(prediction: Prediction) {
        let className = classNames[prediction.classId]
        let confidence = prediction.confidence * 100

        resultLabel.text = "Class: \(className)"
        confidenceLabel.text = String(format: "Confidence: %.1f%%", confidence)
    }

    private func showError(_ message: String) {
        let alert = UIAlertController(
            title: "Error",
            message: message,
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }

    private static func loadClassNames() -> [String] {
        // Load from bundled file or return defaults
        return (0..<1000).map { "Class \($0)" }
    }
}
```

## Android Deployment

### Setup

1. **Add PyTorch Android dependency** in `app/build.gradle`:

```gradle
dependencies {
    implementation 'org.pytorch:pytorch_android:1.13.1'
    implementation 'org.pytorch:pytorch_android_torchvision:1.13.1'
}
```

2. **Add model to assets**:
   - Place `mobile_model.pt` in `app/src/main/assets/`

3. **Add permissions** in `AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
```

### Java/Kotlin Implementation

#### ImageClassifier.kt

```kotlin
package com.example.autotimm

import android.content.Context
import android.graphics.Bitmap
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream

class ImageClassifier(context: Context, modelName: String = "mobile_model.pt") {

    private val module: Module

    // ImageNet normalization
    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std = floatArrayOf(0.229f, 0.224f, 0.225f)

    init {
        // Load model from assets
        module = Module.load(assetFilePath(context, modelName))
    }

    fun predict(bitmap: Bitmap): Prediction {
        // Preprocess
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            bitmap,
            mean,
            std
        )

        // Run inference
        val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()

        // Get scores
        val scores = outputTensor.dataAsFloatArray

        // Find max
        var maxIdx = 0
        var maxScore = scores[0]
        for (i in scores.indices) {
            if (scores[i] > maxScore) {
                maxScore = scores[i]
                maxIdx = i
            }
        }

        return Prediction(maxIdx, maxScore)
    }

    fun predictTopK(bitmap: Bitmap, k: Int = 5): List<Prediction> {
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            bitmap,
            mean,
            std
        )

        val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
        val scores = outputTensor.dataAsFloatArray

        // Get top-k
        val indexed = scores.mapIndexed { idx, score -> Prediction(idx, score) }
        return indexed.sortedByDescending { it.confidence }.take(k)
    }

    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)

        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }

        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                inputStream.copyTo(outputStream)
            }
        }

        return file.absolutePath
    }

    data class Prediction(val classId: Int, val confidence: Float)
}
```

#### MainActivity.kt

```kotlin
package com.example.autotimm

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var resultText: TextView
    private lateinit var confidenceText: TextView
    private lateinit var classifier: ImageClassifier

    private val scope = CoroutineScope(Dispatchers.Main + Job())

    private val selectImageLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            result.data?.data?.let { uri ->
                val inputStream = contentResolver.openInputStream(uri)
                val bitmap = BitmapFactory.decodeStream(inputStream)
                processImage(bitmap)
            }
        }
    }

    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            val bitmap = result.data?.extras?.get("data") as? Bitmap
            bitmap?.let { processImage(it) }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        resultText = findViewById(R.id.resultText)
        confidenceText = findViewById(R.id.confidenceText)

        // Load model
        try {
            classifier = ImageClassifier(this)
        } catch (e: Exception) {
            Toast.makeText(this, "Failed to load model: ${e.message}",
                          Toast.LENGTH_LONG).show()
            return
        }

        // Setup buttons
        findViewById<Button>(R.id.selectImageButton).setOnClickListener {
            selectImage()
        }

        findViewById<Button>(R.id.takePictureButton).setOnClickListener {
            takePicture()
        }
    }

    private fun selectImage() {
        if (checkPermission(Manifest.permission.READ_EXTERNAL_STORAGE)) {
            val intent = Intent(Intent.ACTION_PICK,
                              MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            selectImageLauncher.launch(intent)
        }
    }

    private fun takePicture() {
        if (checkPermission(Manifest.permission.CAMERA)) {
            val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            takePictureLauncher.launch(intent)
        }
    }

    private fun processImage(bitmap: Bitmap) {
        imageView.setImageBitmap(bitmap)

        // Resize to 224x224
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

        // Run inference in background
        scope.launch {
            val prediction = withContext(Dispatchers.Default) {
                classifier.predict(resized)
            }

            showResult(prediction)
        }
    }

    private fun showResult(prediction: ImageClassifier.Prediction) {
        resultText.text = "Class: ${prediction.classId}"
        confidenceText.text = String.format("Confidence: %.1f%%",
                                           prediction.confidence * 100)
    }

    private fun checkPermission(permission: String): Boolean {
        return if (ContextCompat.checkSelfPermission(this, permission)
                  != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(permission), 1)
            false
        } else {
            true
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
    }
}
```

#### activity_main.xml

```xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:padding="16dp">

    <ImageView
        android:id="@+id/imageView"
        android:layout_width="match_parent"
        android:layout_height="0dp"
        android:layout_weight="1"
        android:scaleType="centerInside"
        android:contentDescription="Selected image" />

    <TextView
        android:id="@+id/resultText"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:textSize="18sp"
        android:padding="8dp" />

    <TextView
        android:id="@+id/confidenceText"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:textSize="16sp"
        android:padding="8dp" />

    <Button
        android:id="@+id/selectImageButton"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="Select Image" />

    <Button
        android:id="@+id/takePictureButton"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="Take Picture" />
</LinearLayout>
```

## Model Optimization

### 1. Quantization

Reduce model size and improve inference speed:

```python
import torch
from autotimm import ImageClassifier

# Load model
model = ImageClassifier.load_from_checkpoint("model.ckpt")
model.eval()

# Export with quantization
from torch.quantization import quantize_dynamic

quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)

# Export quantized model
example_input = torch.randn(1, 3, 224, 224)
traced = torch.jit.trace(quantized_model, example_input)
traced.save("mobile_quantized.pt")
```

### 2. Lightweight Backbones

Use mobile-optimized architectures:

```python
# MobileNetV3 - Fastest
model = ImageClassifier(
    backbone="mobilenet_v3_small",
    num_classes=1000
)

# EfficientNet B0 - Good balance
model = ImageClassifier(
    backbone="efficientnet_b0",
    num_classes=1000
)
```

### 3. Input Size Optimization

Use smaller input sizes:

```python
# 224x224 - Standard
example_input = torch.randn(1, 3, 224, 224)

# 128x128 - Faster (if acceptable for your use case)
example_input = torch.randn(1, 3, 128, 128)
```

## Performance Benchmarks

Typical inference times on mobile devices:

| Model | Input Size | iPhone 12 | Pixel 5 |
|-------|-----------|-----------|---------|
| MobileNetV3-Small | 224x224 | 15ms | 25ms |
| EfficientNet-B0 | 224x224 | 30ms | 50ms |
| ResNet18 | 224x224 | 45ms | 75ms |
| ResNet50 | 224x224 | 120ms | 200ms |

## Best Practices

1. **Use Mobile-Optimized Backbones**
   - MobileNet, EfficientNet for best performance
   - Avoid large models (ResNet50+) on mobile

2. **Test on Target Devices**
   - Performance varies significantly across devices
   - Test on oldest device you plan to support

3. **Optimize Model Size**
   - Use quantization when possible
   - Consider pruning for smaller models

4. **Handle Battery Usage**
   - Run inference on background threads
   - Throttle inference rate for continuous detection

5. **Provide Feedback**
   - Show loading indicators during inference
   - Display confidence scores to users

## Troubleshooting

For mobile deployment issues, see the [Troubleshooting - Production Deployment](../../troubleshooting/deployment/production.md) including:

- iOS: Module file doesn't exist
- Android: Native method not found
- Slow inference on mobile devices

## Examples

See complete working examples in the repository:

- `examples/deployment/deploy_torchscript_cpp.py` - Mobile deployment examples

## See Also

- [TorchScript Export](torchscript-export.md) - Export models to TorchScript
- [C++ Deployment](cpp-deployment.md) - Deploy to C++ applications
- [Model Export Guide](../inference/model-export.md) - Overview of all export options
