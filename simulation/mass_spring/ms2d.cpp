// mass-spring simulation in 2d

// implicit method: my formula is incorrect??


// ========================================= Win32 GUI =========================================

#pragma region Windows

#include <Windows.h>
#include <windowsx.h>
#include <tchar.h>

#define WIN_NAME "mass-spring simulation in 2d"
#define WinW_Default 600
#define WinH_Default 400
#define WinW_Min 400
#define WinH_Min 300
#define WinW_Max 3840
#define WinH_Max 2160

void render();
void WindowCreate(int _W, int _H);
void WindowResize(int _oldW, int _oldH, int _W, int _H);
void WindowClose();
void MouseMove(int _X, int _Y);
void MouseWheel(int _DELTA);
void MouseDownL(int _X, int _Y);
void MouseUpL(int _X, int _Y);
void MouseDownR(int _X, int _Y);
void MouseUpR(int _X, int _Y);
void KeyDown(WPARAM _KEY);
void KeyUp(WPARAM _KEY);

HWND _HWND; int _WIN_W, _WIN_H;
HBITMAP _HIMG; COLORREF *_WINIMG;
#define Canvas(x,y) _WINIMG[(y)*_WIN_W+(x)]
#define setColor(x,y,col) do{if((x)>=0&&(x)<_WIN_W&&(y)>=0&&(y)<_WIN_H)Canvas(x,y)=col;}while(0)

double _DEPTHBUF[WinW_Max][WinH_Max];  // how you use this depends on you

bool Render_Needed = true;


LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
#define _RDBK { if (!Render_Needed) break; HDC hdc = GetDC(hWnd), HImgMem = CreateCompatibleDC(hdc); HBITMAP hbmOld = (HBITMAP)SelectObject(HImgMem, _HIMG); render(); BitBlt(hdc, 0, 0, _WIN_W, _WIN_H, HImgMem, 0, 0, SRCCOPY); SelectObject(HImgMem, hbmOld), DeleteDC(HImgMem), DeleteDC(hdc); Render_Needed = false; break; }
	switch (message) {
	case WM_NULL: { _RDBK }
	case WM_CREATE: { RECT Client; GetClientRect(hWnd, &Client); _WIN_W = Client.right, _WIN_H = Client.bottom; WindowCreate(_WIN_W, _WIN_H); break; }
	case WM_CLOSE: { DestroyWindow(hWnd); WindowClose(); return 0; } case WM_DESTROY: { PostQuitMessage(0); return 0; }
	case WM_MOVE:; case WM_SIZE: {
		RECT Client; GetClientRect(hWnd, &Client); WindowResize(_WIN_W, _WIN_H, Client.right, Client.bottom); _WIN_W = Client.right, _WIN_H = Client.bottom;
		BITMAPINFO bmi; bmi.bmiHeader.biSize = sizeof(BITMAPINFO), bmi.bmiHeader.biWidth = Client.right, bmi.bmiHeader.biHeight = Client.bottom, bmi.bmiHeader.biPlanes = 1, bmi.bmiHeader.biBitCount = 32; bmi.bmiHeader.biCompression = BI_RGB, bmi.bmiHeader.biSizeImage = 0, bmi.bmiHeader.biXPelsPerMeter = bmi.bmiHeader.biYPelsPerMeter = 0, bmi.bmiHeader.biClrUsed = bmi.bmiHeader.biClrImportant = 0; bmi.bmiColors[0].rgbBlue = bmi.bmiColors[0].rgbGreen = bmi.bmiColors[0].rgbRed = bmi.bmiColors[0].rgbReserved = 0;
		if (_HIMG != NULL) DeleteObject(_HIMG); HDC hdc = GetDC(hWnd); _HIMG = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, (void**)&_WINIMG, NULL, 0); DeleteDC(hdc); _RDBK }
	case WM_GETMINMAXINFO: { LPMINMAXINFO lpMMI = (LPMINMAXINFO)lParam; lpMMI->ptMinTrackSize.x = WinW_Min, lpMMI->ptMinTrackSize.y = WinH_Min, lpMMI->ptMaxTrackSize.x = WinW_Max, lpMMI->ptMaxTrackSize.y = WinH_Max; break; }
	case WM_PAINT: { PAINTSTRUCT ps; HDC hdc = BeginPaint(hWnd, &ps), HMem = CreateCompatibleDC(hdc); HBITMAP hbmOld = (HBITMAP)SelectObject(HMem, _HIMG); BitBlt(hdc, 0, 0, _WIN_W, _WIN_H, HMem, 0, 0, SRCCOPY); SelectObject(HMem, hbmOld); EndPaint(hWnd, &ps); DeleteDC(HMem), DeleteDC(hdc); break; }
#define _USER_FUNC_PARAMS GET_X_LPARAM(lParam), _WIN_H - 1 - GET_Y_LPARAM(lParam)
	case WM_MOUSEMOVE: { MouseMove(_USER_FUNC_PARAMS); _RDBK }
	case WM_MOUSEWHEEL: { MouseWheel(GET_WHEEL_DELTA_WPARAM(wParam)); _RDBK }
	case WM_LBUTTONDOWN: { SetCapture(hWnd); MouseDownL(_USER_FUNC_PARAMS); _RDBK }
	case WM_LBUTTONUP: { ReleaseCapture(); MouseUpL(_USER_FUNC_PARAMS); _RDBK }
	case WM_RBUTTONDOWN: { MouseDownR(_USER_FUNC_PARAMS); _RDBK }
	case WM_RBUTTONUP: { MouseUpR(_USER_FUNC_PARAMS); _RDBK }
	case WM_SYSKEYDOWN:; case WM_KEYDOWN: { KeyDown(wParam); _RDBK }
	case WM_SYSKEYUP:; case WM_KEYUP: { KeyUp(wParam); _RDBK }
	} return DefWindowProc(hWnd, message, wParam, lParam);
}
int main() {
	HINSTANCE hInstance = NULL; int nCmdShow = SW_RESTORE;
	WNDCLASSEX wc; wc.cbSize = sizeof(WNDCLASSEX), wc.style = 0, wc.lpfnWndProc = WndProc, wc.cbClsExtra = wc.cbWndExtra = 0, wc.hInstance = hInstance; wc.hIcon = wc.hIconSm = 0, wc.hCursor = LoadCursor(NULL, IDC_ARROW), wc.hbrBackground = CreateSolidBrush(RGB(0, 0, 0)), wc.lpszMenuName = NULL, wc.lpszClassName = _T(WIN_NAME); if (!RegisterClassEx(&wc)) return -1;
	_HWND = CreateWindow(_T(WIN_NAME), _T(WIN_NAME), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, WinW_Default, WinH_Default, NULL, NULL, hInstance, NULL); ShowWindow(_HWND, nCmdShow); UpdateWindow(_HWND);
	MSG message; while (GetMessage(&message, 0, 0, 0)) { TranslateMessage(&message); DispatchMessage(&message); } return (int)message.wParam;
}

#pragma endregion  // Windows





// ======================================== Data / Parameters ========================================

#include "numerical/geometry.h"
#include <stdio.h>

// window parameters
char text[64];	// window title
vec2 Center = vec2(0, 0);	// origin in screen coordinate
double Unit = 100.0;		// screen unit to object unit
#define fromInt(p) (((p) - Center) * (1.0 / Unit))
#define fromFloat(p) ((p) * Unit + Center)

// user parameters
vec2 Cursor = vec2(0, 0);
bool mouse_down = false;
bool Ctrl = false, Shift = false, Alt = false;

// forward declarations of rendering functions
void drawLine(vec2 p, vec2 q, COLORREF col);
void drawBox(double x0, double x1, double y0, double y1, COLORREF col);
void drawDot(vec2 c, double r, COLORREF col);
#define drawLineF(p,q,col) drawLine(fromFloat(p),fromFloat(q),col)
#define drawDotF(p,r,col) drawDot(fromFloat(p),r,col)




// ======================================== Objects / Scene ========================================

#include <vector>

struct obj_mass {
	double inv_m;  // reciprocal of mass, can be 0
	double vd;  // air resistance coefficient, f=-vd*v
	vec2 p;  // position
	vec2 v;  // velocity
	vec2 F;  // net force acting on it
	obj_mass() {}
	obj_mass(vec2 p, double m = 1., double vd = 0., vec2 v = vec2(0.)) :p(p), inv_m(1. / m), vd(vd), v(v) {}
};
struct obj_spring {
	double l0;  // rest length
	double ks, kd;  // spring constant and damping constant, |F|=ks*(length(Δx)-r)+kd*dot(Δv,normalize(Δx))
	int m1, m2;  // ID of two connected masses
	obj_spring() {}
	obj_spring(int m1, int m2, double l0 = 0., double ks = 1., double kd = 0.) :m1(m1), m2(m2), l0(l0), ks(ks), kd(kd) {}
};

struct object {
	std::vector<obj_mass> masses;
	std::vector<obj_spring> springs;
	double time = 0.;
	int drag_id = -1;  // the index of the mass that is being dragged by the user

	// conversion between state and float vector for communicating with the solver
	int vector_N() const {
		return 4 * masses.size();  // size of the vector in the differential equation
	}
	void toVector(double v[]) const {
		for (int i = 0, mn = masses.size(); i < mn; i++) {
			vec2* s = (vec2*)&v[4 * i];
			s[0] = masses[i].p;
			s[1] = masses[i].v;
		}
	};
	void toVectorDerivative(double v[]) const {
		for (int i = 0, mn = masses.size(); i < mn; i++) {
			vec2* s = (vec2*)&v[4 * i];
			s[0] = masses[i].v;
			s[1] = masses[i].F * masses[i].inv_m;
		}
		if (drag_id != -1) {
			vec2* s = (vec2*)&v[4 * drag_id];
			s[0] = s[1] = vec2(0);
		}
	}
	void fromVector(const double v[]) {
		for (int i = 0, mn = masses.size(); i < mn; i++) {
			const vec2* s = (vec2*)&v[4 * i];
			masses[i].p = s[0];
			masses[i].v = s[1];
			masses[i].F = vec2(0.);
		}
	}

	// force calculation
	void calcForce_nonstiff();
	void calcForce(bool stiff_only);

	// force derivatives (2Nx2N matrix)
	void calcForce_dfdx_stiff(double A[]);
	void calcForce_dfdv_stiff(double A[]);

	// recalculate drag_id
	bool update_drag_id(vec2 p, double epsilon) {
		if (!(epsilon >= 0.)) {
			drag_id = -1;
			return false;
		}
		drag_id = -1;
		double md2 = epsilon * epsilon;
		for (int i = 0, mn = masses.size(); i < mn; i++) {
			if (masses[i].inv_m == 0) continue;
			double d2 = (p - masses[i].p).sqr();
			if (d2 < md2) md2 = d2, drag_id = i;
		}
		return drag_id != -1;
	}
};


// scene
object Scene;
object Scene_ref;


// acceleration due to gravity
const vec2 g = vec2(0, -1);
// ground contact spring constant
const double k_ground = 1000.;

void object::calcForce_nonstiff() {
	int mn = masses.size();
	for (int i = 0; i < mn; i++) masses[i].F = vec2(0.);
	// gravity
	for (int i = 0; i < mn; i++) {
		masses[i].F += g / max(masses[i].inv_m, 1e-6);
	}
	// ground contact
	for (int i = 0; i < mn; i++) {
		double Fn = k_ground * max(-masses[i].p.y, 0.);
		masses[i].F += vec2(0, Fn);  // no friction, no damping??
	}
}

void object::calcForce(bool stiff_only = false) {
	int mn = masses.size(), sn = springs.size();
	for (int i = 0; i < mn; i++) masses[i].F = vec2(0.);

	// non-stiff forces
	if (!stiff_only) calcForce_nonstiff();

	// viscous drag
	for (int i = 0; i < mn; i++) {
		masses[i].F += -masses[i].vd * masses[i].v;
	}

	// spring force
	for (int i = 0; i < sn; i++) {
		obj_spring s = springs[i];
		obj_mass *m1 = &masses[s.m1], *m2 = &masses[s.m2];
		vec2 dp = m2->p - m1->p;
		vec2 dv = m2->v - m1->v;
		double dpl = length(dp); vec2 edp = dp * (1. / dpl);
		double fm = s.ks * (dpl - s.l0) + s.kd * dot(dv, edp);
		vec2 f = fm * edp;
		m1->F += f, m2->F -= f;
	}

	// dragging
	if (drag_id != -1) masses[drag_id].F = vec2(0.);
}

void object::calcForce_dfdx_stiff(double dfdx[]) {
	int mn = masses.size(), sn = springs.size();
	auto A = [&](int i, int j)->double& { return dfdx[i*(2 * mn) + j]; };
	for (int i = 0; i < 4 * (mn*mn); i++) dfdx[i] = 0.0;

	// spring force
	for (int i = 0; i < sn; i++) {
		obj_spring s = springs[i];
		vec2 p1 = masses[s.m1].p, p2 = masses[s.m2].p;
		vec2 v1 = masses[s.m1].v, v2 = masses[s.m2].v;
		auto F = [&](vec2 p1, vec2 p2, vec2 v1, vec2 v2) {
			vec2 dp = p2 - p1, dv = v2 - v1;
			double dpl = length(dp); vec2 edp = dp * (1. / dpl);
			double fm = s.ks * (dpl - s.l0) + s.kd * dot(dv, edp);
			return fm * edp;
		};
		double eps = 1e-4;
		vec2 dfdp1x = (F(p1 + vec2(eps, 0), p2, v1, v2) -
			F(p1 - vec2(eps, 0), p2, v1, v2)) * (.5 / eps);
		vec2 dfdp1y = (F(p1 + vec2(0, eps), p2, v1, v2) -
			F(p1 - vec2(0, eps), p2, v1, v2)) * (.5 / eps);
		vec2 dfdp2x = (F(p1, p2 + vec2(eps, 0), v1, v2) -
			F(p1, p2 - vec2(eps, 0), v1, v2)) * (.5 / eps);
		vec2 dfdp2y = (F(p1, p2 + vec2(0, eps), v1, v2) -
			F(p1, p2 - vec2(0, eps), v1, v2)) * (.5 / eps);
		double dfdp1[4] = { dfdp1x.x, dfdp1x.y, dfdp1y.x, dfdp1y.y };
		for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
			A(2 * s.m1 + i, 2 * s.m1 + j) += dfdp1[2 * i + j];
			A(2 * s.m1 + i, 2 * s.m2 + j) -= dfdp1[2 * i + j];
		}
		double dfdp2[4] = { dfdp2x.x, dfdp2x.y, dfdp2y.x, dfdp2y.y };
		for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
			A(2 * s.m2 + i, 2 * s.m1 + j) += dfdp2[2 * i + j];
			A(2 * s.m2 + i, 2 * s.m2 + j) -= dfdp2[2 * i + j];
		}
	}

	for (int i = 0; i < 2 * mn; i++) for (int j = 0; j < i; j++) std::swap(A(i, j), A(j, i));
}

void object::calcForce_dfdv_stiff(double dfdv[]) {
	int mn = masses.size(), sn = springs.size();
	auto A = [&](int i, int j)->double& { return dfdv[i*(2 * mn) + j]; };
	for (int i = 0; i < 4 * (mn*mn); i++) dfdv[i] = 0.0;

	// viscous drag
	for (int i = 0; i < mn; i++) {
		A(2 * i, 2 * i) -= masses[i].vd;
		A(2 * i + 1, 2 * i + 1) -= masses[i].vd;
	}

	// spring force
	for (int i = 0; i < sn; i++) {
		obj_spring s = springs[i];
		vec2 p1 = masses[s.m1].p, p2 = masses[s.m2].p;
		vec2 v1 = masses[s.m1].v, v2 = masses[s.m2].v;
		auto F = [&](vec2 p1, vec2 p2, vec2 v1, vec2 v2) {
			vec2 dp = p2 - p1, dv = v2 - v1;
			double dpl = length(dp); vec2 edp = dp * (1. / dpl);
			double fm = s.ks * (dpl - s.l0) + s.kd * dot(dv, edp);
			return fm * edp;
		};
		double eps = 1e-4;
		vec2 dfdv1x = (F(p1, p2, v1 + vec2(eps, 0), v2) -
			F(p1, p2, v1 - vec2(eps, 0), v2)) * (.5 / eps);
		vec2 dfdv1y = (F(p1, p2, v1 + vec2(0, eps), v2) -
			F(p1, p2, v1 - vec2(0, eps), v2)) * (.5 / eps);
		vec2 dfdv2x = (F(p1, p2, v1, v2 + vec2(eps, 0)) -
			F(p1, p2, v1, v2 - vec2(eps, 0))) * (.5 / eps);
		vec2 dfdv2y = (F(p1, p2, v1, v2 + vec2(0, eps)) -
			F(p1, p2, v1, v2 - vec2(0, eps))) * (.5 / eps);
		double dfdv1[4] = { dfdv1x.x, dfdv1x.y, dfdv1y.x, dfdv1y.y };
		for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
			A(2 * s.m1 + i, 2 * s.m1 + j) += dfdv1[2 * i + j];
			A(2 * s.m1 + i, 2 * s.m2 + j) -= dfdv1[2 * i + j];
		}
		double dfdv2[4] = { dfdv2x.x, dfdv2x.y, dfdv2y.x, dfdv2y.y };
		for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
			A(2 * s.m2 + i, 2 * s.m1 + j) += dfdv2[2 * i + j];
			A(2 * s.m2 + i, 2 * s.m2 + j) -= dfdv2[2 * i + j];
		}
	}
}




// ODE solvers

// standard Runge Kutta method
#include "numerical/ode.h"
void step_RungeKutta(object &Scene, double dt) {
	int N = Scene.vector_N();
	double *x = new double[N];
	Scene.toVector(x);
	double t0 = Scene.time;
	auto dxdt = [&](const double* x, double t, double* dx_dt) {
		Scene.fromVector(x);
		Scene.time = t;
		Scene.calcForce();
		Scene.toVectorDerivative(dx_dt);
	};
	double *temp0 = new double[N], *temp1 = new double[N], *temp2 = new double[N];
	RungeKuttaMethod(dxdt, x, N, t0, dt, temp0, temp1, temp2);
	Scene.fromVector(x);
	Scene.time = t0 + dt;
	delete x; delete temp0; delete temp1; delete temp2;
}

// standard Euler method
void step_Euler(object &Scene, double dt) {
	Scene.calcForce();
	for (int i = 0, N = Scene.masses.size(); i < N; i++) {
		if (i != Scene.drag_id) {
			Scene.masses[i].p += Scene.masses[i].v * dt;
			Scene.masses[i].v += Scene.masses[i].inv_m * Scene.masses[i].F * dt;
		}
		else {
			Scene.masses[i].v = vec2(0.);
		}
	}
	Scene.time += dt;
}

// implicit Euler method (brute force, elimination)
#include "numerical/linearsystem.h"
void step_impEuler_elim(object &Scene, double h) {
	// Δv = (I - h⋅M⁻¹⋅(∂f/∂v + h⋅∂f/∂x))⁻¹ [h⋅M⁻¹⋅(f(x₀,v₀) + h⋅∂f/∂x⋅v₀)]
	// Δx = h⋅(v₀ + Δv)

	int mn = Scene.masses.size(), N = 2 * mn;

	// calculate M⁻¹
	double *inv_m = new double[N*N];
	for (int i = 0; i < N*N; i++) inv_m[i] = 0.0;
	for (int i = 0; i < mn; i++) {
		inv_m[(2 * i)*(N + 1)] = inv_m[(2 * i + 1)*(N + 1)] = Scene.masses[i].inv_m;
	}

	// calculate f(x₀,v₀)
	Scene.calcForce(true);
	double *f0 = new double[N], *x0 = new double[N], *v0 = new double[N];
	for (int i = 0; i < mn; i++) {
		f0[2 * i] = Scene.masses[i].F.x;
		f0[2 * i + 1] = Scene.masses[i].F.y;
		x0[2 * i] = Scene.masses[i].p.x;
		x0[2 * i + 1] = Scene.masses[i].p.y;
		v0[2 * i] = Scene.masses[i].v.x;
		v0[2 * i + 1] = Scene.masses[i].v.y;
	}

	// calculate ∂f/∂x and ∂f/∂v
	double *dFdx = new double[N*N], *dFdv = new double[N*N];
	Scene.calcForce_dfdx_stiff(dFdx);
	Scene.calcForce_dfdv_stiff(dFdv);

	// multiply ∂f/∂x, ∂f/∂v, and f(x₀,v₀) by M⁻¹
	double *A = new double[N*N];
#if 0
	matmul(N, inv_m, dFdx, A); matcpy(N, A, dFdx);
	matmul(N, inv_m, dFdv, A); matcpy(N, A, dFdv);
#else
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			dFdx[i*N + j] *= inv_m[i*N + i];
			dFdv[i*N + j] *= inv_m[i*N + i];
		}
	}
#endif
	for (int i = 0; i < N; i++) f0[i] *= inv_m[i*N + i];

	// left of the linear system: I-h⋅M⁻¹⋅∂f/∂v-h²⋅M⁻¹⋅∂f/∂x
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			A[i*N + j] = (i == j ? 1.0 : 0.0) - h * dFdv[i*N + j] - h * h*dFdx[i*N + j];
		}
	}

	// right of the linear system: h⋅M⁻¹⋅(f(x₀,v₀) + h⋅∂f/∂x⋅v₀)
	double *dv = new double[N];
	matvecmul(N, dFdx, v0, dv);
	for (int i = 0; i < N; i++) {
		dv[i] = h * (f0[i] + h * dv[i]);
	}

	// solve the linear system
	solveLinear(N, A, dv);

	// update
	Scene.calcForce_nonstiff();
	for (int i = 0; i < mn; i++) {
		Scene.masses[i].v += *(vec2*)&dv[2 * i];
		Scene.masses[i].v += Scene.masses[i].F*Scene.masses[i].inv_m * h;
		Scene.masses[i].p += h * Scene.masses[i].v;
	}
	Scene.time += h;
	delete inv_m; delete dFdx; delete dFdv; delete A;
	delete f0; delete x0; delete v0; delete dv;
}



void updateScene(double dt, vec2 cursor) {
	// split large steps
	double max_step = 0.04;
	if (dt > max_step) {
		int N = (int)(dt / max_step + 1);
		for (int i = 0; i < N; i++) updateScene(dt / N, cursor);
		return;
	}

	// solve ODE

	//step_Euler(Scene, dt);
	step_impEuler_elim(Scene, dt);
	step_RungeKutta(Scene_ref, dt);

	// dragging + collision detection (to be implemented)
	if (Scene.drag_id != -1) {
		Scene.masses[Scene.drag_id].p = cursor;
	}
}





// ============================================ Rendering ============================================

#include <chrono>
auto t0 = std::chrono::high_resolution_clock::now();

double V[1920][1080];

void drawLine(vec2 p, vec2 q, COLORREF col) {
	vec2 d = q - p;
	double slope = d.y / d.x;
	if (abs(slope) <= 1.0) {
		if (p.x > q.x) std::swap(p, q);
		int x0 = max(0, int(p.x)), x1 = min(_WIN_W - 1, int(q.x)), y;
		double yf = slope * x0 + (p.y - slope * p.x);
		for (int x = x0; x <= x1; x++) {
			y = (int)yf;
			if (y >= 0 && y < _WIN_H) Canvas(x, y) = col;
			yf += slope;
		}
	}
	else {
		slope = d.x / d.y;
		if (p.y > q.y) std::swap(p, q);
		int y0 = max(0, int(p.y)), y1 = min(_WIN_H - 1, int(q.y)), x;
		double xf = slope * y0 + (p.x - slope * p.y);
		for (int y = y0; y <= y1; y++) {
			x = (int)xf;
			if (x >= 0 && x < _WIN_W) Canvas(x, y) = col;
			xf += slope;
		}
	}
};
void drawBox(double x0, double x1, double y0, double y1, COLORREF col) {
	drawLineF(vec2(x0, y0), vec2(x1, y0), col);
	drawLineF(vec2(x0, y1), vec2(x1, y1), col);
	drawLineF(vec2(x0, y0), vec2(x0, y1), col);
	drawLineF(vec2(x1, y0), vec2(x1, y1), col);
};
void drawDot(vec2 c, double r, COLORREF col) {
	int i0 = max(0, (int)floor(c.x - r - 1)), i1 = min(_WIN_W - 1, (int)ceil(c.x + r + 1));
	int j0 = max(0, (int)floor(c.y - r - 1)), j1 = min(_WIN_H - 1, (int)ceil(c.y + r + 1));
	for (int j = j0; j <= j1; j++) for (int i = i0; i <= i1; i++) {
		double d = length(vec2(i, j) - c) - r;
		if (d < 0.) Canvas(i, j) = col;
	}
}


void render() {
	// debug
	auto t1 = std::chrono::high_resolution_clock::now();
	double dt = std::chrono::duration<double>(t1 - t0).count();
	printf("[%d×%d] time elapsed: %.1fms (%.1ffps)\n", _WIN_W, _WIN_H, 1000.0*dt, 1. / dt);
	t0 = t1;

	// initialize window
	for (int i = 0, l = _WIN_W * _WIN_H; i < l; i++) _WINIMG[i] = 0;

	// axis and grid
	{
		vec2 LB = fromInt(vec2(0, 0)), RT = fromInt(vec2(_WIN_W, _WIN_H));
		COLORREF GridCol = clamp(0x10, (int)sqrt(10.0*Unit), 0x20); GridCol = GridCol | (GridCol << 8) | (GridCol) << 16;  // adaptive grid color
		for (int y = (int)round(LB.y), y1 = (int)round(RT.y); y <= y1; y++) drawLine(fromFloat(vec2(LB.x, y)), fromFloat(vec2(RT.x, y)), GridCol);  // horizontal gridlines
		for (int x = (int)round(LB.x), x1 = (int)round(RT.x); x <= x1; x++) drawLine(fromFloat(vec2(x, LB.y)), fromFloat(vec2(x, RT.y)), GridCol);  // vertical gridlines
		drawLine(vec2(0, Center.y), vec2(_WIN_W, Center.y), 0x404060);  // x-axis
		drawLine(vec2(Center.x, 0), vec2(Center.x, _WIN_H), 0x404060);  // y-axis
	}

	// scene
	{
		if (1) {
			for (int i = 0, sn = Scene_ref.springs.size(); i < sn; i++) {
				drawLineF(Scene_ref.masses[Scene_ref.springs[i].m1].p, Scene_ref.masses[Scene_ref.springs[i].m2].p, 0x404040);
			}
			for (int i = 0, mn = Scene_ref.masses.size(); i < mn; i++) {
				drawDotF(Scene_ref.masses[i].p, 2, 0x400000);
			}
		}
		for (int i = 0, sn = Scene.springs.size(); i < sn; i++) {
			drawLineF(Scene.masses[Scene.springs[i].m1].p, Scene.masses[Scene.springs[i].m2].p, 0xFFFFFF);
		}
		for (int i = 0, mn = Scene.masses.size(); i < mn; i++) {
			drawDotF(Scene.masses[i].p, 2, 0xFF0000);
		}
	}


	vec2 cursor = fromInt(Cursor);
	sprintf(text, "(%.2f,%.2f)  %.2fs", cursor.x, cursor.y, Scene.time);
	SetWindowTextA(_HWND, text);
}




// ============================================== Tests ==============================================


class presetScenes {
	typedef obj_mass om;
	typedef obj_spring os;
public:
	static void box_X() {
		object s;
		s.masses.assign({ om(vec2(0,1),.8,.1), om(vec2(1,1),1.,.1), om(vec2(1,2),1.,.1), om(vec2(0,2),1.,.1) });
		const double rt2 = sqrt(2.) - 0.1;
		s.springs.assign({ os(0,1,1.0), os(1,2,1.0), os(2,3,1.0), os(3,0,1.0), os(0,2,rt2), os(1,3,rt2) });
		for (int i = 0, sn = s.springs.size(); i < sn; i++) {
			s.springs[i].ks = 100.;
			s.springs[i].kd = 2.;
		}
		Scene = s;
	}
	static void box_XX(int xD, int yD, double lx, double ly, double ang) {
		object s;
		// masses
		for (int y = 0; y <= yD; y++) for (int x = 0; x <= xD; x++) {
			//double m = 1.0 / ((xD + 1)*(yD + 1));
			double m = 1.0;
			s.masses.push_back(om(vec2(x, y), m, .1));
		}
		// positions
		for (int i = 0, mn = s.masses.size(); i < mn; i++) {
			s.masses[i].p = rotationMatrix2d(ang)*(s.masses[i].p*vec2(lx / xD, ly / yD) - 0.5*vec2(lx, ly)) + vec2(0, 2);
		}
		// horizontal springs
		double ks = 100., kd = 5.;
		double dx = lx / xD, dy = ly / yD, dl = hypot(dx, dy);
		for (int y = 0; y <= yD; y++) for (int x = 0; x < xD; x++) {
			s.springs.push_back(os(y*(xD + 1) + x, y*(xD + 1) + x + 1, dx, ks / dx, kd / dx));
		}
		// vertical springs
		for (int x = 0; x <= xD; x++) for (int y = 0; y < yD; y++) {
			s.springs.push_back(os(y*(xD + 1) + x, (y + 1)*(xD + 1) + x, dy, ks / dy, kd / dy));
		}
		// cross springs
		for (int x = 0; x < xD; x++) for (int y = 0; y < yD; y++) {
			s.springs.push_back(os(y*(xD + 1) + x, (y + 1)*(xD + 1) + x + 1, dl, ks / dl, kd / dl));
			s.springs.push_back(os(y*(xD + 1) + x + 1, (y + 1)*(xD + 1) + x, dl, ks / dl, kd / dl));
		}
		Scene = s;
	}
	static void polygon_S(int N, double r1, double r2) {
		object s;
		// masses
		for (int i = 0; i < N; i++)
			s.masses.push_back(om(r1*cossin(i*2.*PI / N), 1. + 0.1*sin(1234.56*i + 78.9), .1));
		for (int i = 0; i < N; i++)
			s.masses.push_back(om(r2*cossin(i*2.*PI / N), 1. + 0.1*sin(3456.78*i + 90.1), .1));
		for (int i = 0, mn = s.masses.size(); i < mn; i++)
			s.masses[i].p += vec2(0, 2);
		// springs
		double ks = 500., kd = 5.;
		for (int i = 0; i < N; i++) {
			// longitude springs
			s.springs.push_back(os(i, (i + 1) % N, 2.*r1*sin(PI / N), ks, kd));
			s.springs.push_back(os(i + N, (i + 1) % N + N, 2.*r2*sin(PI / N), ks, kd));
			// latitude springs
			s.springs.push_back(os(i, i + N, abs(r2 - r1), ks, kd));
			// cross springs
			s.springs.push_back(os(i, (i + 1) % N + N, NAN, ks, kd));
			s.springs.push_back(os(i + N, (i + 1) % N, NAN, ks, kd));
		}
		// spring length
		for (int i = 0, sn = s.springs.size(); i < sn; i++) {
			s.springs[i].l0 = length(s.masses[s.springs[i].m2].p - s.masses[s.springs[i].m1].p);
		}
		Scene = s;
	}
	static void sheet_hang_2(int xD, int yD, double lx, double ly, bool hasCross = false, bool allHang = false) {
		object s;
		// masses
		for (int y = 0; y <= yD; y++) for (int x = 0; x <= xD; x++) {
			double m = 10.0 / ((xD + 1)*(yD + 1));
			s.masses.push_back(om(vec2(x, y)*vec2(lx / xD, ly / yD) - vec2(0.5*lx, ly) + vec2(0, 2.5), m, .1));
		}
		// hang
		s.masses[yD*(xD + 1)].inv_m = 0;
		s.masses[yD*(xD + 1) + xD].inv_m = 0;
		if (allHang) for (int i = 0; i <= xD; i++) s.masses[yD*(xD + 1) + i].inv_m = 0;
		// horizontal springs
		double ks = 10., kd = 5.;
		double dx = lx / xD, dy = ly / yD, dl = hypot(dx, dy);
		for (int y = 0; y <= yD; y++) for (int x = 0; x < xD; x++) {
			s.springs.push_back(os(y*(xD + 1) + x, y*(xD + 1) + x + 1, dx, ks / dx, kd / dx));
		}
		// vertical springs
		for (int x = 0; x <= xD; x++) for (int y = 0; y < yD; y++) {
			s.springs.push_back(os(y*(xD + 1) + x, (y + 1)*(xD + 1) + x, dy, ks / dy, kd / dy));
		}
		// cross springs
		if (hasCross) for (int x = 0; x < xD; x++) for (int y = 0; y < yD; y++) {
			s.springs.push_back(os(y*(xD + 1) + x, (y + 1)*(xD + 1) + x + 1, dl, ks / dl, kd / dl));
			s.springs.push_back(os(y*(xD + 1) + x + 1, (y + 1)*(xD + 1) + x, dl, ks / dl, kd / dl));
		}
		Scene = s;
	}
};

void initScene() {
	//presetScenes::box_X();
	//presetScenes::box_XX(5, 4, 1.0, 0.8, 0.2);
	//presetScenes::box_XX(10, 8, 2.0, 1.6, 1.5);
	//presetScenes::polygon_S(8, 0.5, 0.8);
	//presetScenes::polygon_S(16, 0.5, 1.0);
	presetScenes::sheet_hang_2(16, 8, 3.0, 1.5, false, false);
	//presetScenes::sheet_hang_2(24, 12, 3.0, 1.5, true, false);
	Scene_ref = Scene;
}




// ============================================== User ==============================================


#include <thread>

HANDLE animationThread;


void WindowCreate(int _W, int _H) {
	Center = vec2(_W, 0.2*_H) * 0.5;
	initScene();

	// simulation thread
	new std::thread([]() {
		auto t0 = std::chrono::high_resolution_clock::now();
		const double frame_delay = 0.01;
		for (;;) {
			double time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
			double time_next = ceil(time / frame_delay)*frame_delay;
			double dt = time_next - Scene.time;
			updateScene(dt, fromInt(Cursor));
			double t_remain = time_next - std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
			//if (t_remain > 0.) std::this_thread::sleep_for(std::chrono::milliseconds(int(1000.*t_remain)));
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
	});

	// rendering thread
	new std::thread([]() {
		for (;;) {
			if (_WINIMG) {
				Render_Needed = true;
				SendMessage(_HWND, WM_NULL, NULL, NULL);
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
			}
		}
	});
}

void WindowResize(int _oldW, int _oldH, int _W, int _H) {
	if (_W*_H == 0 || _oldW * _oldH == 0) return;
	double pw = _oldW, ph = _oldH, w = _W, h = _H;
	double s = sqrt((w * h) / (pw * ph));
	Unit *= s;
	Center.x *= w / pw, Center.y *= h / ph;
	Render_Needed = true;
}
void WindowClose() {}

void MouseMove(int _X, int _Y) {
	Render_Needed = true;
	Cursor = vec2(_X, _Y);
	return;

	vec2 P0 = Cursor, P = vec2(_X, _Y);
	vec2 p0 = fromInt(P0), p = fromInt(P), d = p - p0;

	// click and drag
	if (mouse_down) {
		Center = Center + d * Unit;
	}
}

void MouseWheel(int _DELTA) {
	Render_Needed = true;
	return;

	double s = exp((Alt ? 0.0001 : 0.001)*_DELTA);
	double D = length(vec2(_WIN_W, _WIN_H)), Max = 100.0*D, Min = 0.0001*D;
	if (Unit * s > Max) s = Max / Unit;
	else if (Unit * s < Min) s = Min / Unit;
	Center = mix(Cursor, Center, s);
	Unit *= s;
}

void MouseDownL(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	mouse_down = true;
	vec2 p = fromInt(Cursor);
	Scene.update_drag_id(p, 10. / Unit);
	Render_Needed = true;
}

void MouseUpL(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	mouse_down = false;
	vec2 p = fromInt(Cursor);
	Scene.update_drag_id(vec2(NAN), -1);
}

void MouseDownR(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
}

void MouseUpR(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	vec2 p = fromInt(Cursor);
}

void KeyDown(WPARAM _KEY) {
	if (_KEY == VK_CONTROL) Ctrl = true;
	else if (_KEY == VK_SHIFT) Shift = true;
	else if (_KEY == VK_MENU) Alt = true;
}

void KeyUp(WPARAM _KEY) {
	if (_KEY == VK_CONTROL) Ctrl = false;
	else if (_KEY == VK_SHIFT) Shift = false;
	else if (_KEY == VK_MENU) Alt = false;
}

