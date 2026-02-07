# C++ Deployment

Deploy AutoTimm models to C++ applications using TorchScript and LibTorch.

## Overview

AutoTimm models can be exported to TorchScript and deployed in C++ applications without Python dependencies. This enables:

- **High Performance** - Native C++ speed without Python overhead
- **Production Ready** - Stable, type-safe deployment
- **Cross-Platform** - Windows, Linux, macOS support
- **Server Deployment** - REST APIs, microservices, edge servers
- **Embedded Systems** - Raspberry Pi, Jetson Nano, custom hardware

## Prerequisites

### 1. LibTorch

Download and install LibTorch (C++ distribution of PyTorch):

**CPU Version:**
```bash
# Linux
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip

# macOS
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-latest.zip
unzip libtorch-macos-latest.zip
```

**CUDA Version:**
```bash
# Linux with CUDA 11.8
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
```

### 2. OpenCV (Optional)

For image preprocessing:

```bash
# Ubuntu/Debian
sudo apt-get install libopencv-dev

# macOS
brew install opencv

# Verify installation
pkg-config --modversion opencv4
```

### 3. CMake

```bash
# Ubuntu/Debian
sudo apt-get install cmake

# macOS
brew install cmake
```

## Quick Start

### Step 1: Export Model

First, export your trained AutoTimm model to TorchScript:

```python
from autotimm import ImageClassifier
import torch

# Load trained model
model = ImageClassifier.load_from_checkpoint("model.ckpt")

# Export to TorchScript
model.eval()
example_input = torch.randn(1, 3, 224, 224)
model.to_torchscript("model.pt", example_input=example_input)
```

### Step 2: Create C++ Application

Create `inference.cpp`:

```cpp
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: inference <model.pt>\n";
        return 1;
    }

    // Load the model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
        module.eval();
        std::cout << "Model loaded successfully\n";
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        return 1;
    }

    // Create example input
    auto input = torch::randn({1, 3, 224, 224});

    // Run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    torch::NoGradGuard no_grad;
    auto output = module.forward(inputs).toTensor();

    std::cout << "Output shape: " << output.sizes() << "\n";
    std::cout << "Prediction: " << output.argmax(1).item<int>() << "\n";

    return 0;
}
```

### Step 3: Build

Create `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.18)
project(autotimm_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Torch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add executable
add_executable(inference inference.cpp)

# Link libraries
target_link_libraries(inference "${TORCH_LIBRARIES}")
```

Build and run:

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release

# Run
./inference ../model.pt
```

## Complete Example: Image Classifier

Here's a complete example with image preprocessing using OpenCV:

### classifier.hpp

```cpp
#pragma once
#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class ImageClassifier {
public:
    ImageClassifier(const std::string& model_path,
                   const std::vector<std::string>& class_names = {});

    // Single image classification
    int predict(const cv::Mat& image, float& confidence);

    // Batch classification
    std::vector<std::pair<int, float>> predict_batch(
        const std::vector<cv::Mat>& images);

    // Get top-k predictions
    std::vector<std::pair<int, float>> predict_topk(
        const cv::Mat& image, int k = 5);

private:
    torch::jit::script::Module module_;
    std::vector<std::string> class_names_;
    torch::Device device_;

    // Preprocessing
    torch::Tensor preprocess(const cv::Mat& image);
    std::vector<torch::Tensor> preprocess_batch(
        const std::vector<cv::Mat>& images);
};
```

### classifier.cpp

```cpp
#include "classifier.hpp"
#include <algorithm>
#include <stdexcept>

ImageClassifier::ImageClassifier(
    const std::string& model_path,
    const std::vector<std::string>& class_names)
    : class_names_(class_names),
      device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
{
    try {
        module_ = torch::jit::load(model_path);
        module_.eval();
        module_.to(device_);
        std::cout << "Model loaded on "
                  << (device_.is_cuda() ? "CUDA" : "CPU") << "\n";
    } catch (const c10::Error& e) {
        throw std::runtime_error("Failed to load model: " +
                               std::string(e.what()));
    }
}

torch::Tensor ImageClassifier::preprocess(const cv::Mat& image) {
    // Resize
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(224, 224));

    // BGR to RGB
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    // Convert to float [0, 1]
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

    // Normalize with ImageNet stats
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};

    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < resized.rows; i++) {
            for (int j = 0; j < resized.cols; j++) {
                resized.at<cv::Vec3f>(i, j)[c] =
                    (resized.at<cv::Vec3f>(i, j)[c] - mean[c]) / std[c];
            }
        }
    }

    // Convert to tensor [1, 3, 224, 224]
    auto tensor = torch::from_blob(
        resized.data,
        {1, resized.rows, resized.cols, 3},
        torch::kFloat32
    );
    tensor = tensor.permute({0, 3, 1, 2}).contiguous();

    return tensor.to(device_);
}

int ImageClassifier::predict(const cv::Mat& image, float& confidence) {
    auto input = preprocess(image);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    torch::NoGradGuard no_grad;
    auto output = module_.forward(inputs).toTensor();

    // Softmax and get prediction
    auto probs = torch::softmax(output, 1);
    auto max_result = probs.max(1);

    confidence = std::get<0>(max_result).item<float>();
    int class_id = std::get<1>(max_result).item<int>();

    return class_id;
}

std::vector<std::pair<int, float>> ImageClassifier::predict_batch(
    const std::vector<cv::Mat>& images)
{
    auto batch = preprocess_batch(images);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::stack(batch));

    torch::NoGradGuard no_grad;
    auto output = module_.forward(inputs).toTensor();
    auto probs = torch::softmax(output, 1);

    std::vector<std::pair<int, float>> results;
    for (int i = 0; i < images.size(); i++) {
        auto max_result = probs[i].max(0);
        float confidence = std::get<0>(max_result).item<float>();
        int class_id = std::get<1>(max_result).item<int>();
        results.push_back({class_id, confidence});
    }

    return results;
}

std::vector<std::pair<int, float>> ImageClassifier::predict_topk(
    const cv::Mat& image, int k)
{
    auto input = preprocess(image);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    torch::NoGradGuard no_grad;
    auto output = module_.forward(inputs).toTensor();
    auto probs = torch::softmax(output, 1).squeeze(0);

    // Get top-k
    auto topk_result = probs.topk(k);
    auto values = std::get<0>(topk_result);
    auto indices = std::get<1>(topk_result);

    std::vector<std::pair<int, float>> results;
    for (int i = 0; i < k; i++) {
        results.push_back({
            indices[i].item<int>(),
            values[i].item<float>()
        });
    }

    return results;
}

std::vector<torch::Tensor> ImageClassifier::preprocess_batch(
    const std::vector<cv::Mat>& images)
{
    std::vector<torch::Tensor> tensors;
    for (const auto& image : images) {
        tensors.push_back(preprocess(image).squeeze(0));
    }
    return tensors;
}
```

### main.cpp

```cpp
#include "classifier.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.pt> <image1.jpg> [image2.jpg ...]\n";
        return 1;
    }

    try {
        // Load classifier
        ImageClassifier classifier(argv[1]);

        // Process each image
        for (int i = 2; i < argc; i++) {
            cv::Mat image = cv::imread(argv[i]);
            if (image.empty()) {
                std::cerr << "Failed to load: " << argv[i] << "\n";
                continue;
            }

            // Single prediction
            float confidence;
            int class_id = classifier.predict(image, confidence);

            std::cout << "\n" << argv[i] << ":\n";
            std::cout << "  Class: " << class_id << "\n";
            std::cout << "  Confidence: " << confidence * 100 << "%\n";

            // Top-5 predictions
            auto topk = classifier.predict_topk(image, 5);
            std::cout << "  Top-5:\n";
            for (const auto& [id, conf] : topk) {
                std::cout << "    " << id << ": "
                         << conf * 100 << "%\n";
            }
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
```

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.18)
project(autotimm_classifier)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add executable
add_executable(classifier
    main.cpp
    classifier.cpp
)

# Link libraries
target_link_libraries(classifier
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)
```

### Build and Run

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release

./classifier model.pt image1.jpg image2.jpg
```

## REST API Server

Create a simple REST API using [cpp-httplib](https://github.com/yhirose/cpp-httplib):

```cpp
#include "classifier.hpp"
#include "httplib.h"
#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    // Load classifier
    ImageClassifier classifier("model.pt");

    httplib::Server svr;

    // Health check
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("OK", "text/plain");
    });

    // Predict endpoint
    svr.Post("/predict", [&classifier](const httplib::Request& req,
                                        httplib::Response& res) {
        // Decode image from request body
        std::vector<uchar> buffer(req.body.begin(), req.body.end());
        cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);

        if (image.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"Invalid image\"}",
                          "application/json");
            return;
        }

        // Predict
        float confidence;
        int class_id = classifier.predict(image, confidence);

        // Return JSON response
        std::string json = "{\"class\": " + std::to_string(class_id) +
                          ", \"confidence\": " + std::to_string(confidence) +
                          "}";
        res.set_content(json, "application/json");
    });

    std::cout << "Server running on http://localhost:8080\n";
    svr.listen("0.0.0.0", 8080);

    return 0;
}
```

Test with curl:

```bash
curl -X POST http://localhost:8080/predict \
  --data-binary @image.jpg \
  -H "Content-Type: application/octet-stream"
```

## Performance Optimization

### 1. GPU Acceleration

```cpp
// Check CUDA availability
if (torch::cuda::is_available()) {
    std::cout << "CUDA available, using GPU\n";
    device = torch::kCUDA;
    module.to(device);
} else {
    std::cout << "CUDA not available, using CPU\n";
}
```

### 2. Thread Pool

For batch processing:

```cpp
#include <thread>
#include <queue>
#include <mutex>

class InferenceThreadPool {
public:
    InferenceThreadPool(ImageClassifier* classifier, int num_threads)
        : classifier_(classifier), num_threads_(num_threads) {}

    void process_images(const std::vector<std::string>& image_paths) {
        std::vector<std::thread> threads;

        for (int i = 0; i < num_threads_; i++) {
            threads.emplace_back([this, &image_paths, i]() {
                for (size_t j = i; j < image_paths.size();
                     j += num_threads_) {
                    process_single(image_paths[j]);
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }

private:
    void process_single(const std::string& path) {
        cv::Mat image = cv::imread(path);
        float confidence;
        int class_id = classifier_->predict(image, confidence);
        // Handle result...
    }

    ImageClassifier* classifier_;
    int num_threads_;
};
```

### 3. Batch Inference

Process multiple images at once:

```cpp
// Load multiple images
std::vector<cv::Mat> images = {
    cv::imread("img1.jpg"),
    cv::imread("img2.jpg"),
    cv::imread("img3.jpg")
};

// Batch prediction (much faster than individual)
auto results = classifier.predict_batch(images);
```

## Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    cmake \
    g++ \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install LibTorch
WORKDIR /opt
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip \
    && unzip libtorch-cxx11-abi-shared-with-deps-latest.zip \
    && rm libtorch-cxx11-abi-shared-with-deps-latest.zip

# Copy application
WORKDIR /app
COPY . .

# Build
RUN mkdir build && cd build \
    && cmake -DCMAKE_PREFIX_PATH=/opt/libtorch .. \
    && cmake --build . --config Release

# Run
CMD ["./build/classifier", "model.pt"]
```

Build and run:

```bash
docker build -t autotimm-inference .
docker run -v $(pwd)/model.pt:/app/model.pt autotimm-inference
```

## Troubleshooting

### Issue: "undefined symbol" errors

**Solution:** Ensure you're using the correct LibTorch ABI:

```bash
# Check your compiler's ABI
g++ --version

# Use cxx11-abi version for modern compilers
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
```

### Issue: "CUDA error: no kernel image available"

**Solution:** Match LibTorch CUDA version with your system:

```bash
# Check CUDA version
nvidia-smi

# Download matching LibTorch (e.g., CUDA 11.8)
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-latest.zip
```

### Issue: Slow CPU inference

**Solution:** Enable optimizations:

```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
```

## Examples

See complete working examples in the repository:

- `examples/deployment/deploy_torchscript_cpp.py` - Complete C++ deployment examples

## See Also

- [TorchScript Export](torchscript-export.md) - Export models to TorchScript
- [Mobile Deployment](mobile-deployment.md) - Deploy to iOS/Android
- [Model Export Guide](../inference/model-export.md) - Overview of all export options
