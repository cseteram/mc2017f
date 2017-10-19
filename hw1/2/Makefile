all: flops_float_add flops_float_mul flops_double_add flops_double_mul

flops_float_add: flops_float_add.c
	gcc -o flops_float_add flops_float_add.c -Wall -std=c99 -O2

flops_double_add: flops_double_add.c
	gcc -o flops_double_add flops_double_add.c -Wall -std=c99 -O2

flops_float_mul: flops_float_mul.c
	gcc -o flops_float_mul flops_float_mul.c -Wall -std=c99 -O2

flops_double_mul: flops_double_mul.c
	gcc -o flops_double_mul flops_double_mul.c -Wall -std=c99 -O2

clean:
	rm flops_float_add flops_float_mul flops_double_add flops_double_mul
