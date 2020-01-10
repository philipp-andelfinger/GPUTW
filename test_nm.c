#include <stdio.h>
#include "nelder_mead.c"

float func(float x, float y) {
	return -(x*x - 4*x + y*y - y - x*y);
}

int main() {
	nm_start(0,   0,   0, func(  0,   0));
	nm_start(1, 1.2,   0, func(1.2,   0));
	nm_start(2,   0, 0.8, func(  0, 0.8));

	float x, y, f;

	int i;
	for (i = 0; i < 100; i++) {
		nm_get_next_point(f, &x, &y);
		f = func(x, y);

		printf("%d %f\n",i , f);
	}

	return 0;
}
		
