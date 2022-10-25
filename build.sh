nvcc -O3 msm.cu cuda_src/*.cu;
if [ $? -eq 0 ]; then
    echo "Compilation successful";
    ./a.out;
    rm a.out;
fi
