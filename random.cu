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

#include "random.h"

__device__
uint random(curandState_t *state, uint max) {
	return curand(state) % max;
}

__device__
uint random_exp(curandState_t *state, uint mean) {
	float ru = curand_uniform(state);
	return -(mean * logf(ru));
}

__device__ // private
uint calc_v0(uint v3, uint v4) {
	uint r = v4 ^ v3 ^ (v3 << 4);
	
	uint t = r;
	for (int i = 0; i < 31; i++) { t = t ^ (t >> 1); }
	t = t & 1;
	t = t | (t << 1);
	t = t << 30;

	t = (r >> 1) ^ (r >> 2) ^ t;
	for (int i = 0; i < 7; i++) { t = t ^ (t >> 4); }

	return t;
}

__device__
uint calc_v0_2(uint v3, uint v4) {
	uint r = v4 ^ v3 ^ (v3 << 4);

	uint t = 0;
	for (int i = 0; i < 32; i++) {
		t = t ^ r;
		r = r << 1;
	}

	uint v = 0;
	for (int i = 0; i < 32; i += 2) {
		v = v ^ t;
		t = t >> 2;
	}

	return v;
}

__device__
void reverse_state(curandState_t *state) {
	uint v0 = calc_v0_2(state->v[3], state->v[4]);
	state->v[4] = state->v[3];
	state->v[3] = state->v[2];
	state->v[2] = state->v[1];
	state->v[1] = state->v[0];
	state->v[0] = v0;
	state->d -= 362437;
}
