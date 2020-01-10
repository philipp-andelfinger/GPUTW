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

#ifndef random_h
#define random_h

#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

__device__
uint random(curandState_t *state, uint max);

__device__
uint random_exp(curandState_t *state, uint mean);

__device__
void reverse_state(curandState_t *state);

#endif
