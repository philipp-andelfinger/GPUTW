/*  This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#include "nelder_mead.h"

static float x[3];
static float y[3];
static float f[3];

static float mx, my, rx, ry, ex, ey, cx ,cy, sx, sy;
static int phase;

static int best, good, worst;

void nm_start(int start, float start_x, float start_y, float start_f) {
	x[start] = start_x;
	y[start] = start_y;
	f[start] = start_f;

	phase = 0;
}

void determin_bgw() {
	if (f[1] > f[0]) {
		best = 1; good = 0;
	} else {
		best = 0; good = 1;
	}

	if (f[2] > f[best]) {
		worst = good; good = best; best = 2;
	} else if (f[2] > f[good]) {
		worst = good; good = 2;
	} else {
		worst = 2;
	}
}

void calc_m() {
	mx = (x[best] + x[good]) / 2;
	my = (y[best] + y[good]) / 2;
}

void calc_r() {
	rx = 2 * mx - x[worst];
	ry = 2 * my - y[worst];
}

void calc_e() {
	ex = 2 * rx - mx;
 	ey = 2 * ry - my;
}

void calc_c1() {
	cx = (x[worst] + mx) / 2;
	cy = (y[worst] + my) / 2;
}

void calc_c2() {
	cx = (mx + rx) / 2;
	cy = (my + ry) / 2;
}

void calc_s() {
	sx = (x[worst] + x[best]) / 2;
	sy = (y[worst] + y[best]) / 2;
}

void nm_get_next_point(float last_f, float *next_x, float *next_y) {
	if (phase == 1) {
		if (last_f > f[best]) {		// Try expansion
			calc_e();
			f[worst] = last_f;

			*next_x = ex;
			*next_y = ey;

			phase = 2;
			return;
		} else if (last_f > f[good]) {	// Refelction
		// Replace W with R
			x[worst] = rx;
			y[worst] = ry;
			f[worst] = last_f;

//			printf("RELECTION\n");
		} else {			// Try contraction
			if (last_f > f[worst]) {
				calc_c2();
				f[worst] = last_f;
			} else {
				calc_c1();
			}

			*next_x = cx;
			*next_y = cy;

			phase = 3;
			return;
		}
	} else if (phase == 2) {		// Try expansion
		if (last_f > f[best]) {		// Expansion: yes
			// Replace W with E
			x[worst] = ex;
			y[worst] = ey;
			f[worst] = last_f;

//			printf("EXPANSION\n");
		} else {			// Expansion: no
			// Replace W with R
			x[worst] = rx;
			y[worst] = ry;

//			printf("REFLECTION\n");
		}
	} else if (phase == 3) {		// Try contraction
		if (last_f > f[worst]) {	// Contraction
			// Replace W with C
			x[worst] = cx;
			y[worst] = cy;
			f[worst] = last_f;

//			printf("CONTRACTION\n");
		} else {			// Shrink
			calc_s();

			*next_x = sx;
			*next_y = sy;

//			printf("SHRINK\n");

			phase = 4;
			return;
		}
	} else if (phase == 4) {		// Shrink W to S
		// Replace W with S
		x[worst] = sx;
		y[worst] = sy;
		f[worst] = last_f;

		*next_x = mx;
		*next_y = my;

		phase = 5;
		return;
	} else if (phase == 5) {		// Shrink G to M
		// Replace G with M
		x[good] = mx;
		y[good] = my;
		f[good] = last_f;
	}

	determin_bgw();
	calc_m(); calc_r();

	*next_x = rx;
	*next_y = ry;

	phase = 1;
}
