#include <stdio.h>

int main()
{
    double x;
    scanf("%lf", &x);
    unsigned long long int n = *(unsigned long long int*)&x;

    int bin[64];
    for (int i = 0; i < 64; i++) {
        bin[i] = n % 2;
        n /= 2;
    }
    for (int i = 63; i >= 0; i--)
        printf("%d", bin[i]);

    printf("\n%f\n", x);
    return 0;
}

