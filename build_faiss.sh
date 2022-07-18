# In cases where pip isntall fais-gpu does not work or gpu is  not used you can build Faiss from source using the following commands

export CUDA_HOME=/usr/local/cuda-11.6
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.6/lib64:/usr/local/cuda-11.6/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin

https://github.com/facebookresearch/faiss.git
rm -rf build

cmake -B build -DCUDAToolkit_ROOT=/usr/local/cuda-11.6/ -DCMAKE_CUDA_ARCHITECTURES="86" -DPython_EXECUTABLE=/usr/bin/python3 .

make -C build -j faiss

make -C build -j swigfaiss
(cd build/faiss/python && python3 setup.py install)