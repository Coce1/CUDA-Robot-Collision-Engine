NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_75

TARGET = collision_engine
SRCS = src/main.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) -I./src $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)
