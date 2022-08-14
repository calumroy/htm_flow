#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cudaflow.hpp>

// This requires installing CUDA compiler nvcc 
// See https://taskflow.github.io/taskflow/CompileTaskflowWithCUDA.html

__global__ void saxpy(size_t N, float alpha, float* dx, float* dy) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a*x[i] + y[i];
  }
}
tf::Task cudaflow = taskflow.emplace([&](tf::cudaFlow& cf) {

  // data copy tasks
  tf::cudaTask h2d_x = cf.copy(dx, hx.data(), N).name("h2d_x");
  tf::cudaTask h2d_y = cf.copy(dy, hy.data(), N).name("h2d_y");
  tf::cudaTask d2h_x = cf.copy(hx.data(), dx, N).name("d2h_x");
  tf::cudaTask d2h_y = cf.copy(hy.data(), dy, N).name("d2h_y");
  
  // kernel task with parameters to launch the saxpy kernel
  tf::cudaTask saxpy = cf.kernel(
    (N+255)/256, 256, 0, saxpy, N, 2.0f, dx, dy
  ).name("saxpy");

  saxpy.succeed(h2d_x, h2d_y)
       .precede(d2h_x, d2h_y);
}).name("cudaFlow");