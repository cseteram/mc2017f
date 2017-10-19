all: fma no_fma

fma: fma.c
	gcc -o fma fma.c -Wall -std=c99 -mfma -O2

no_fma: no_fma.c
	gcc -o no_fma no_fma.c -Wall -std=c99 -O2

clean:
	rm fma no_fma
