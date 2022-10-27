nvcc -O3 -o cu_msm.out cuda_src/*.cu;
if [ $? -eq 0 ]; then
    echo "Compilation successful";
    ./cu_msm.out;
    rm cu_msm.out;
fi
