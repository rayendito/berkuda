query: src/query.cu
	nvcc src/query.cu -o bin/query
	./bin/query

add: src/add.cu
	nvcc src/add.cu -o bin/add
	./bin/add

dot: src/dot.cu
	nvcc src/dot.cu -o bin/dot
	./bin/dot

sgemm_naive: src/sgemm_naive.cu
	nvcc src/sgemm_naive.cu -o bin/sgemm_naive
	./bin/sgemm_naive