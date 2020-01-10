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

#include "nelder_mead_3d.h"

static float x[4];
static float y[4];
static float z[4];
static float f[4];

static float mx, my, mz, rx, ry, rz, ex, ey, ez, cx ,cy, cz;
static int phase;

static int best, good, worst;

void nm_start(
int start, float start_x, float start_y, float start_z, float start_f) {
	x[start] = start_x;
	y[start] = start_y;
	z[start] = start_z;
	f[start] = start_f;

	phase = 0;
}

void determin_bgw() {
	best = 0; worst = 0;

	for (int i = 1; i < 4; i++) {
		if (f[i] > f[best]) {
			best = i;
		} else if (f[i] < f[worst]) {
			worst = i;
		}
	}

	good = -1;

	for (int i = 0; i < 4; i++) {
		if (i == best || i == worst) { continue; }

		if (good == -1) { good = i; }
		else if (f[i] < f[good]) { good = i; }
	}
}

void calc_m() {
	mx = my = mz = 0;

	for (int i = 0; i < 4; i++) {
		if (i == worst) { continue; }

		mx += x[i];
		my += y[i];
		mz += z[i];
	}

	mx /= 3;
	my /= 3;
	mz /= 3;
}

void calc_r() {
	rx = 2 * mx - x[worst];
	ry = 2 * my - y[worst];
	rz = 2 * mz - z[worst];
}

void calc_e() {
	ex = 2 * rx - mx;
 	ey = 2 * ry - my;
	ez = 2 * rz - mz;
}

void calc_c1() {
	cx = (x[worst] + mx) / 2;
	cy = (y[worst] + my) / 2;
	cz = (z[worst] + mz) / 2;
}

void calc_c2() {
	cx = (mx + rx) / 2;
	cy = (my + ry) / 2;
	cz = (mz + rz) / 2;
}

void nm_get_next_point(
float last_f, float *next_x, float *next_y, float *next_z) {
	if (phase == 1) {
		if (last_f > f[best]) {		// Try expansion
			calc_e();
			f[worst] = last_f;

			*next_x = ex;
			*next_y = ey;
			*next_z = ez;

			phase = 2;
			return;
		} else if (last_f > f[good]) {	// Refelction
		// Replace W with R
			x[worst] = rx;
			y[worst] = ry;
			z[worst] = rz;
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
			*next_z = cz;

			phase = 3;
			return;
		}
	} else if (phase == 2) {		// Try expansion
		if (last_f > f[worst]) {	// Expansion: yes
			// Replace W with E
			x[worst] = ex;
			y[worst] = ey;
			z[worst] = ez;
			f[worst] = last_f;

//			printf("EXPANSION\n");
		} else {			// Expansion: no
			// Replace W with R
			x[worst] = rx;
			y[worst] = ry;
			z[worst] = rz;

//			printf("REFLECTION\n");
		}
	} else if (phase == 3) {		// Try contraction
		if (last_f > f[worst]) {	// Contraction
			// Replace W with C
			x[worst] = cx;
			y[worst] = cy;
			z[worst] = cz;
			f[worst] = last_f;

//			printf("CONTRACTION\n");
		} else {			// Shrink
//			calc_s();

			int i = (best + 1) % 4;

			x[i] = (x[best] + x[i]) / 2;
			y[i] = (y[best] + y[i]) / 2;
			z[i] = (z[best] + z[i]) / 2;

			*next_x = x[i];
			*next_y = y[i];
			*next_z = z[i];

//			printf("SHRINK\n");

			phase = 4;
			return;
		}
	} else if (phase == 4) {		// Shrink point 1
		int i = (best + 1) % 4;
		f[i] = last_f;

		i = (i + 1) % 4;
		x[i] = (x[best] + x[i]) / 2;
		y[i] = (y[best] + y[i]) / 2;
		z[i] = (z[best] + z[i]) / 2;

		*next_x = x[i];
		*next_y = y[i];
		*next_z = z[i];

		phase = 5;
		return;
	} else if (phase == 5) {		// Shrink point 2
		int i = (best + 2) % 4;
		f[i] = last_f;

		i = (i + 1) % 4;
		x[i] = (x[best] + x[i]) / 2;
		y[i] = (y[best] + y[i]) / 2;
		z[i] = (z[best] + z[i]) / 2;

		*next_x = x[i];
		*next_y = y[i];
		*next_z = z[i];

		phase = 6;
		return;
	} else if (phase == 6) {		// Shrink point 3
		int i = (best + 3) % 4;
		f[i] = last_f;
	}

	determin_bgw();
	calc_m(); calc_r();

	*next_x = rx;
	*next_y = ry;
	*next_z = rz;

	phase = 1;
}
