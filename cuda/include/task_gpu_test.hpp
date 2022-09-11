#ifndef TASK_GPU_TEST_HP
#define TASK_GPU_TEST_HP

namespace task_gpu_test{

class TaskGPUTest {
public:

    TaskGPUTest();

    std::vector<int> gpu(int M, int N, int K);
    auto cpu(int M, int N, int K);

private:
    bool test1;

};

}

#endif //TASK_GPU_TEST_HP