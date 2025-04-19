query: src/query.cu
	nvcc src/query.cu -o bin/query
	./bin/query

hello: src/hello.cu
	nvcc src/hello.cu -o bin/hello
	./bin/hello