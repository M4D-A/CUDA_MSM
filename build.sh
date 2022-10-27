gcc -O3 cpu_src/*.cpp -lstdc++;
if [ $? -eq 0 ]; then
    echo "Compilation successful";
    ./a.out;
    rm a.out;
fi
