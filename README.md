# Bachelor Project

### Setup instructions:
```bash
git clone --recurse-submodules https://github.com/PaulUrsulean/bachelor-project.git
cd bachelor-project/trident/
mkdir build
cd build
cmake .. -DPYTHON=1
make

# Only needed if unpacking the .nt file
cd ../..
./trident/build/trident load -f lubm1.nt -i ./lubm1/ --relsOwnIDs 1

# Make sure to add the trident build directory to the environment PYTHONPATH variable
python3 train.py ./lubm1 ./models
```
