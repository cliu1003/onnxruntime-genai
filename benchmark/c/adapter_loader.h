// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

struct OgaGenerator;
struct OgaTensor;

namespace benchmark {

struct LoadedAdapterTensor {
  std::string name;
  std::vector<int8_t> data;
  std::vector<uint8_t> u8_data;
  bool is_uint8{false};
  std::vector<int64_t> shape;
};

struct LoadedAdapter {
  std::vector<LoadedAdapterTensor> tensors;
};

struct DequantizedAdapterTensor {
  std::string name;
  std::vector<uint16_t> data;
  std::vector<int64_t> shape;
};

struct BoundAdapter {
  const LoadedAdapter* loaded{nullptr};
  std::vector<std::unique_ptr<OgaTensor>> ort_tensors;
  std::vector<DequantizedAdapterTensor> dequantized_tensors;
};

struct LoraDequantEntry {
  std::string onnx_input;
  float scale{};
  int32_t zero_point{};
};

using LoraDequantMap = std::unordered_map<std::string, LoraDequantEntry>;

LoadedAdapter LoadSafetensors(const std::string& path);

std::optional<LoraDequantMap> LoadLoraDequantMap(const std::string& model_path);

void BindAdapterToGenerator(OgaGenerator& generator, BoundAdapter& bound_adapter, const std::string& model_path);

}  // namespace benchmark
