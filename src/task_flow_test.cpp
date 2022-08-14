#include <taskflow/taskflow.hpp>  // Taskflow is header-only

#include <taskflow/taskflow.hpp>

// Procedure: for_each
void for_each(int N) {

  tf::Executor executor;
  tf::Taskflow taskflow;

  std::vector<int> range(N);
  std::iota(range.begin(), range.end(), 0);

  taskflow.for_each(range.begin(), range.end(), [&] (int i) {
    printf("for_each on container item: %d\n", i);
  });

  executor.run(taskflow).get();
}

// Procedure: for_each_index
void for_each_index(int N) {

  tf::Executor executor;
  tf::Taskflow taskflow;

  // [0, N) with step size 2
  taskflow.for_each_index(0, N, 2, [] (int i) {
    printf("for_each_index on index: %d\n", i);
  });

  executor.run(taskflow).get();
}


// Function: main
int main(int argc, char* argv[]) {

  if(argc != 2) {
    std::cerr << "Usage: ./parallel_for num_iterations" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  tf::Executor executor;
  tf::Taskflow taskflow;

  auto [A, B, C, D] = taskflow.emplace(  // create four tasks
    [] () { std::cout << "TaskA\n"; },
    [] () { std::cout << "TaskB\n"; },
    [] () { std::cout << "TaskC\n"; },
    [] () { std::cout << "TaskD\n"; } 
  );                                  
                                      
  A.precede(B, C);  // A runs before B and C
  D.succeed(B, C);  // D runs after  B and C
                                      
  executor.run(taskflow).wait(); 
  
  // Run some parallel for loops
  for_each(std::atoi(argv[1]));
  for_each_index(std::atoi(argv[1]));


  return 0;
}