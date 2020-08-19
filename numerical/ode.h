
// dxdt: given vector and time, calculate the derivative
//       void dxdt(const double* x, double t, double* dx_dt);
// x: given vector at time t0, update it to time t1
// N: length of vector
// t0: start time
// dt: last time (end time is t0+dt)
// temp: initialize to size N, use to avoid memory allocation

template<typename Fun>
void EulersMethod(Fun dxdt, double* x, int N, double t0, double dt, double* temp) {
	dxdt(x, t0, temp);
	for (int i = 0; i < N; i++) {
		x[i] += temp[i] * dt;
	}
}

template<typename Fun>
void MidpointMethod(Fun dxdt, double* x, int N, double t0, double dt, double* temp1, double* temp2) {
	dxdt(x, t0, temp1);
	for (int i = 0; i < N; i++) {
		temp1[i] = temp1[i] * (.5*dt) + x[i];
	}
	dxdt(temp1, t0 + .5*dt, temp2);
	for (int i = 0; i < N; i++) {
		x[i] += temp2[i] * dt;
	}
}

template<typename Fun>
void RungeKuttaMethod(Fun dxdt, double* x, int N, double t0, double dt, double* temp0, double* temp1, double* temp2) {
	dxdt(x, t0, temp1);
	for (int i = 0; i < N; i++) {
		temp1[i] *= dt;
		temp0[i] = x[i] + (1. / 6.)*temp1[i];
		temp1[i] = x[i] + .5*temp1[i];
	}
	dxdt(temp1, t0 + .5*dt, temp2);
	for (int i = 0; i < N; i++) {
		temp2[i] *= dt;
		temp0[i] += (1. / 3.)*temp2[i];
		temp2[i] = x[i] + .5*temp2[i];
	}
	dxdt(temp2, t0 + .5*dt, temp1);
	for (int i = 0; i < N; i++) {
		temp1[i] *= dt;
		temp0[i] += (1. / 3.)*temp1[i];
		temp1[i] += x[i];
	}
	dxdt(temp1, t0 + dt, temp2);
	for (int i = 0; i < N; i++) {
		temp2[i] *= dt;
		x[i] = temp0[i] + (1. / 6.)*temp2[i];
	}
}


// To-do: implicit method; adaptive step length

