# Encountered issues

## Taskflow issues
I had an issue where taskflow tasks kept either not outputting anything or was crashing due a segmentation fault.
e.g
```
./build/htm_flow_tests --gtest_filter=parallel_Images2Neibs_1D.test1_small
Running main() from /home/calum/Documents/projects/htm_flow/lib/googletest/googletest/src/gtest_main.cc
Note: Google Test filter = parallel_Images2Neibs_1D.test1_small
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from parallel_Images2Neibs_1D
[ RUN      ] parallel_Images2Neibs_1D.test1_small
Segmentation fault
```

The issue was that in the parallel_IMages2Neibs_1D function that was called by the test it would use the input parameters
create some temporary variables inside the function and then create a taskflow lambda task that would use the temporary variables.
The issue was that the temporary variables would go out of scope before the taskflow task was executed. This would cause the taskflow task to segfault.
The solution was to make sure any variables required by the taskflow task were passed in by reference and additional temp variables where created in the lambda function so they would not go out of scope. 
See the definition of parallel_Images2Neibs_1D for an example of this.
