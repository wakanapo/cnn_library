#include "set_gene.hpp"
#include <memory>


namespace {
static std::unique_ptr<GlobalParams> g_global_params;
}  // namespace

GlobalParams* GlobalParams::getInstance() {
  return g_global_params.get();
}

void GlobalParams::setParams(std::vector<float> partition) {
  g_global_params.reset(new GlobalParams(std::move(partition)));
}
