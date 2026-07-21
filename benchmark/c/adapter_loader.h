// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct OgaGenerator;
struct OgaTensor;

namespace benchmark {

struct LoadedAdapterTensor {
  std::string name;
  std::vector<int8_t> data;
  std::vector<int64_t> shape;
};

struct LoadedAdapter {
  std::vector<LoadedAdapterTensor> tensors;
};

struct BoundAdapter {
  const LoadedAdapter* loaded{nullptr};
  std::vector<std::unique_ptr<OgaTensor>> ort_tensors;
};

LoadedAdapter LoadSafetensors(const std::string& path);

void BindAdapterToGenerator(OgaGenerator& generator, BoundAdapter& bound_adapter);

}  // namespace benchmark
