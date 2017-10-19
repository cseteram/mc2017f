#include <stdio.h>

int main()
{
    float x;
    scanf("%f", &x);
    unsigned int n = *(unsigned int*)&x;

    int bin[32];
    for (int i = 0; i < 32; i++) {
        bin[i] = n % 2;
        n /= 2;
    }
    for (int i = 31; i >= 0; i--)
        printf("%d", bin[i]);

    printf("\n%f\n", x);
    return 0;
}

