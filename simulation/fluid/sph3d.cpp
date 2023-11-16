// Win32 3D GUI template

// ========================================= Win32 Standard =========================================

#pragma region Windows

#ifndef UNICODE
#define UNICODE
#endif

#include <Windows.h>
#include <windowsx.h>
#include <tchar.h>



#pragma region Window Macros / Forward Declarations


#define WIN_NAME "3D GUI Template"
#define WinW_Padding 100
#define WinH_Padding 100
#define WinW_Default 640
#define WinH_Default 400
#define WinW_Min 400
#define WinH_Min 300
#define WinW_Max 3840
#define WinH_Max 2160

void Init();  // called before window is created
void render();
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

#pragma endregion  // Windows global variables and forward declarations


// Win32 Entry

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
#define _RDBK { if (!Render_Needed) break; HDC hdc = GetDC(hWnd), HImgMem = CreateCompatibleDC(hdc); HBITMAP hbmOld = (HBITMAP)SelectObject(HImgMem, _HIMG); render(); BitBlt(hdc, 0, 0, _WIN_W, _WIN_H, HImgMem, 0, 0, SRCCOPY); SelectObject(HImgMem, hbmOld), DeleteDC(HImgMem), DeleteDC(hdc); Render_Needed = false; break; }
	switch (message) {
	case WM_NULL: { _RDBK }
	case WM_CREATE: { if (!_HWND) Init(); break; } case WM_CLOSE: { WindowClose(); DestroyWindow(hWnd); return 0; } case WM_DESTROY: { PostQuitMessage(0); return 0; }
	case WM_MOVE:; case WM_SIZE: { RECT Client; GetClientRect(hWnd, &Client); WindowResize(_WIN_W, _WIN_H, Client.right, Client.bottom); _WIN_W = Client.right, _WIN_H = Client.bottom; BITMAPINFO bmi; bmi.bmiHeader.biSize = sizeof(BITMAPINFO), bmi.bmiHeader.biWidth = Client.right, bmi.bmiHeader.biHeight = Client.bottom, bmi.bmiHeader.biPlanes = 1, bmi.bmiHeader.biBitCount = 32; bmi.bmiHeader.biCompression = BI_RGB, bmi.bmiHeader.biSizeImage = 0, bmi.bmiHeader.biXPelsPerMeter = bmi.bmiHeader.biYPelsPerMeter = 0, bmi.bmiHeader.biClrUsed = bmi.bmiHeader.biClrImportant = 0; bmi.bmiColors[0].rgbBlue = bmi.bmiColors[0].rgbGreen = bmi.bmiColors[0].rgbRed = bmi.bmiColors[0].rgbReserved = 0; if (_HIMG != NULL) DeleteObject(_HIMG); HDC hdc = GetDC(hWnd); _HIMG = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, (void**)&_WINIMG, NULL, 0); DeleteDC(hdc); _RDBK }
	case WM_GETMINMAXINFO: { LPMINMAXINFO lpMMI = (LPMINMAXINFO)lParam; lpMMI->ptMinTrackSize.x = WinW_Min, lpMMI->ptMinTrackSize.y = WinH_Min, lpMMI->ptMaxTrackSize.x = WinW_Max, lpMMI->ptMaxTrackSize.y = WinH_Max; break; }
	case WM_PAINT: { PAINTSTRUCT ps; HDC hdc = BeginPaint(hWnd, &ps), HMem = CreateCompatibleDC(hdc); HBITMAP hbmOld = (HBITMAP)SelectObject(HMem, _HIMG); BitBlt(hdc, 0, 0, _WIN_W, _WIN_H, HMem, 0, 0, SRCCOPY); SelectObject(HMem, hbmOld); EndPaint(hWnd, &ps); DeleteDC(HMem), DeleteDC(hdc); break; }
#define _USER_FUNC_PARAMS GET_X_LPARAM(lParam), _WIN_H - 1 - GET_Y_LPARAM(lParam)
	case WM_MOUSEMOVE: { MouseMove(_USER_FUNC_PARAMS); _RDBK } case WM_MOUSEWHEEL: { MouseWheel(GET_WHEEL_DELTA_WPARAM(wParam)); _RDBK }
	case WM_LBUTTONDOWN: { SetCapture(hWnd); MouseDownL(_USER_FUNC_PARAMS); _RDBK } case WM_LBUTTONUP: { ReleaseCapture(); MouseUpL(_USER_FUNC_PARAMS); _RDBK }
	case WM_RBUTTONDOWN: { MouseDownR(_USER_FUNC_PARAMS); _RDBK } case WM_RBUTTONUP: { MouseUpR(_USER_FUNC_PARAMS); _RDBK }
	case WM_SYSKEYDOWN:; case WM_KEYDOWN: { if (wParam >= 0x08) KeyDown(wParam); _RDBK } case WM_SYSKEYUP:; case WM_KEYUP: { if (wParam >= 0x08) KeyUp(wParam); _RDBK }
	} return DefWindowProc(hWnd, message, wParam, lParam);
}
int main() {
	HINSTANCE hInstance = NULL; int nCmdShow = SW_RESTORE;
	WNDCLASSEX wc; wc.cbSize = sizeof(WNDCLASSEX), wc.style = 0, wc.lpfnWndProc = WndProc, wc.cbClsExtra = wc.cbWndExtra = 0, wc.hInstance = hInstance; wc.hIcon = wc.hIconSm = 0, wc.hCursor = LoadCursor(NULL, IDC_ARROW), wc.hbrBackground = CreateSolidBrush(RGB(0, 0, 0)), wc.lpszMenuName = NULL, wc.lpszClassName = _T(WIN_NAME); if (!RegisterClassEx(&wc)) return -1;
	_HWND = CreateWindow(_T(WIN_NAME), _T(WIN_NAME), WS_OVERLAPPEDWINDOW, WinW_Padding, WinH_Padding, WinW_Default, WinH_Default, NULL, NULL, hInstance, NULL); ShowWindow(_HWND, nCmdShow); UpdateWindow(_HWND);
	MSG message; while (GetMessage(&message, 0, 0, 0)) { TranslateMessage(&message); DispatchMessage(&message); } return (int)message.wParam;
}

#pragma endregion  // WIN32


// ================================== Vector Classes/Functions ==================================

#include <cmath>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <thread>
#include <chrono>
#include "numerical/geometry.h"
#include "numerical/random.h"
#include "triangulate/octatree.h"
#include "UI/stl_encoder.h"

#pragma region Vector & Matrix

const vec3 vec0(0, 0, 0), veci(1, 0, 0), vecj(0, 1, 0), veck(0, 0, 1);
#define SCRCTR vec2(0.5*_WIN_W,0.5*_WIN_H)

// 4x4 matrix
struct Affine {
	vec3 u, v, w;  // first row, second row, third row
	vec3 t, p;  // translation, perspective
	double s;  // scaling
};
vec3 operator * (Affine T, vec3 p) {
	vec3 q = vec3(dot(T.u, p), dot(T.v, p), dot(T.w, p)) + T.t;
	double d = 1.0 / (dot(T.p, p) + T.s);
	return d < 0.0 ? vec3(NAN) : q * d;
	//return q * d;
}
Affine operator * (const Affine &A, const Affine &B) {
	Affine R;
	R.u = A.u.x*B.u + A.u.y*B.v + A.u.z*B.w + A.t.x*B.p;
	R.v = A.v.x*B.u + A.v.y*B.v + A.v.z*B.w + A.t.y*B.p;
	R.w = A.w.x*B.u + A.w.y*B.v + A.w.z*B.w + A.t.z*B.p;
	R.t = vec3(dot(A.u, B.t), dot(A.v, B.t), dot(A.w, B.t)) + A.t*B.s;
	R.p = vec3(A.p.x*B.u.x + A.p.y*B.v.x + A.p.z*B.w.x, A.p.x*B.u.y + A.p.y*B.v.y + A.p.z*B.w.y, A.p.x*B.u.z + A.p.y*B.v.z + A.p.z*B.w.z) + B.p*A.s;
	R.s = dot(A.p, B.t) + A.s*B.s;
	return R;
}


#pragma endregion  // Vector & Matrix


#pragma region Ray Tracing Functions

// Intersection functions - return the distance, NAN means no intersection
double intHorizon(double z, vec3 p, vec3 d) {
	return (z - p.z) / d.z;
}
double intSphere(vec3 O, double r, vec3 p, vec3 d) {
	p = p - O;
	double b = -dot(p, d), c = dot(p, p) - r * r;  // required to be normalized
	double delta = b * b - c;
	if (delta < 0.0) return NAN;
	delta = sqrt(delta);
	c = b - delta;
	return c > 0. ? c : b + delta;  // usually we want it to be positive
}
double intTriangle(vec3 v0, vec3 v1, vec3 v2, vec3 ro, vec3 rd) {
	vec3 v1v0 = v1 - v0, v2v0 = v2 - v0, rov0 = ro - v0;
	vec3 n = cross(v1v0, v2v0);
	vec3 q = cross(rov0, rd);
	double d = 1.0 / dot(rd, n);
	double u = d * dot(-q, v2v0); if (u<0. || u>1.) return NAN;
	double v = d * dot(q, v1v0); if (v<0. || (u + v)>1.) return NAN;
	return d * dot(-n, rov0);
}


#pragma endregion Raytracing - Intersector, Normal, SDF, Bounding Box


// ======================================== Data / Parameters ========================================

// viewport
vec3 Center(0.0, 0.0, 0.0);  // view center in world coordinate
double rz = 0.2*PI, rx = 0.15*PI, ry = 0.0, dist = 12.0, Unit = 100.0;  // yaw, pitch, roll, camera distance, scale to screen

#pragma region General Global Variables

// window parameters
Affine Tr;  // matrix
vec3 CamP, ScrO, ScrA, ScrB;  // camera and screen
auto scrDir = [](vec2 pixel) { return normalize(ScrO + (pixel.x / _WIN_W)*ScrA + (pixel.y / _WIN_H)*ScrB - CamP); };

// user parameters
vec2 Cursor = vec2(0, 0), clickCursor;  // current cursor and cursor position when mouse down
bool mouse_down = false;
bool Ctrl = false, Shift = false, Alt = false;

#pragma endregion Camera/Screen, Mouse/Key



#pragma region Global Variable Related Functions

// projection - matrix vs camera/screen
Affine axisAngle(vec3 axis, double a) {
	axis = normalize(axis); double ct = cos(a), st = sin(a);
	return Affine{
		vec3(ct + axis.x*axis.x*(1 - ct), axis.x*axis.y*(1 - ct) - axis.z*st, axis.x*axis.z*(1 - ct) + axis.y*st),
		vec3(axis.y*axis.x*(1 - ct) + axis.z*st, ct + axis.y*axis.y*(1 - ct), axis.y*axis.z*(1 - ct) - axis.x*st),
		vec3(axis.z*axis.x*(1 - ct) - axis.y*st, axis.z*axis.y*(1 - ct) + axis.x*st, ct + axis.z*axis.z*(1 - ct)),
		vec3(0), vec3(0), 1.0
	};
}
void calcMat() {
	double cx = cos(rx), sx = sin(rx), cz = cos(rz), sz = sin(rz), cy = cos(ry), sy = sin(ry);
	Affine D{ veci, vecj, veck, -Center, vec3(0), 1.0 };  // world translation
	Affine R{ vec3(-sz, cz, 0), vec3(-cz * sx, -sz * sx, cx), vec3(-cz * cx, -sz * cx, -sx), vec3(0), vec3(0), 1.0 };  // rotation
	R = Affine{ vec3(cy, -sy, 0), vec3(sy, cy, 0), vec3(0, 0, 1), vec3(0), vec3(0), 1.0 } *R;  // camera roll (ry)
	Affine P{ veci, vecj, veck, vec3(0), vec3(0, 0, 1.0 / dist), 1.0 };  // perspective
	Affine S{ veci, vecj, veck, vec3(0), vec3(0), 1.0 / Unit };  // scaling
	Affine T{ veci, vecj, veck, vec3(SCRCTR, 0.0), vec3(0), 1.0 };  // screen translation
	Tr = T * S * P * R * D;
}
void getRay(vec2 Cursor, vec3 &p, vec3 &d) {
	p = CamP;
	d = normalize(ScrO + (Cursor.x / _WIN_W)*ScrA + (Cursor.y / _WIN_H)*ScrB - CamP);
}
void getScreen(vec3 &P, vec3 &O, vec3 &A, vec3 &B) {  // O+uA+vB
	double cx = cos(rx), sx = sin(rx), cz = cos(rz), sz = sin(rz);
	vec3 u(-sz, cz, 0), v(-cz * sx, -sz * sx, cx), w(cz * cx, sz * cx, sx);
	Affine Y = axisAngle(w, -ry); u = Y * u, v = Y * v;
	u *= 0.5*_WIN_W / Unit, v *= 0.5*_WIN_H / Unit, w *= dist;
	P = Center + w;
	O = Center - (u + v), A = u * 2.0, B = v * 2.0;
}
void projRange_Sphere(vec3 P, double r, vec2 &p0, vec2 &p1) {
	vec3 O = ScrO - CamP, k = CamP - P;
	vec3 Ak = cross(ScrA, k), Bk = cross(ScrB, k), Ok = cross(O, k);
	double r2 = r * r;
	// A x² + B y² + C xy + D x + E y + F = 0
	double A = r2 * ScrA.sqr() - Ak.sqr();
	double B = r2 * ScrB.sqr() - Bk.sqr();
	double C = 2.0*(r2*dot(ScrA, ScrB) - dot(Ak, Bk));
	double D = 2.0*(r2*dot(ScrA, O) - dot(Ak, Ok));
	double E = 2.0*(r2*dot(ScrB, O) - dot(Bk, Ok));
	double F = r2 * O.sqr() - Ok.sqr();
	double a, b, c, delta, t0, t1;
	if (abs(C / F) < 1e-6) {  // not sure if I use the right formula
		a = 4 * A*B, b = 4 * A*E, c = 4 * A*F - D * D;
		delta = sqrt(b*b - 4 * a*c);
		t0 = (-b + delta) / (2.0*a), t1 = (-b - delta) / (2.0*a); if (t0 > t1) std::swap(t0, t1);
		p0.y = t0 * _WIN_H, p1.y = t1 * _WIN_H;
		a = 4 * A*B, b = 4 * B*D, c = 4 * B*F - E * E;
		delta = sqrt(b*b - 4 * a*c);
		t0 = (-b + delta) / (2.0*a), t1 = (-b - delta) / (2.0*a); if (t0 > t1) std::swap(t0, t1);
		p0.x = t0 * _WIN_W, p1.x = t1 * _WIN_W;
	}
	else {
		a = 4 * A*A*B - A * C*C, b = 4 * A*B*D - 2 * A*C*E, c = B * D*D - C * D*E + C * C*F;
		delta = sqrt(b*b - 4 * a*c);
		t0 = (-b + delta) / (2.0*a), t1 = (-b - delta) / (2.0*a);
		t0 = (-D - 2 * A*t0) / C, t1 = (-D - 2 * A*t1) / C; if (t0 > t1) std::swap(t0, t1);
		p0.y = t0 * _WIN_H, p1.y = t1 * _WIN_H;
		a = 4 * A*B*B - B * C*C, b = 4 * A*B*E - 2 * B*C*D, c = A * E*E - C * D*E + C * C*F;
		delta = sqrt(b*b - 4 * a*c);
		t0 = (-b + delta) / (2.0*a), t1 = (-b - delta) / (2.0*a);
		t0 = (-E - 2 * B*t0) / C, t1 = (-E - 2 * B*t1) / C; if (t0 > t1) std::swap(t0, t1);
		p0.x = t0 * _WIN_W, p1.x = t1 * _WIN_W;
	}
}
void projRange_Triangle(vec3 A, vec3 B, vec3 C, vec2 &p0, vec2 &p1) {
	vec2 p = (Tr*A).xy(); p0 = p1 = p;
	p = (Tr*B).xy(); p0 = pMin(p0, p), p1 = pMax(p1, p);
	p = (Tr*C).xy(); p0 = pMin(p0, p), p1 = pMax(p1, p);
}

#pragma endregion Get Ray/Screen, projection




// ============================================ Simulation ============================================


// faster when set to true
#define USE_NEIGHBOR 1

// reconstruct surface or draw particles
#define RECONSTRUCT_SURFACE 1

namespace SPH {

	// scene
	int N = 0;  // number of SPH particles
	struct particle {
		vec3 p;  // position
		vec3 v;  // velocity
		double density;  // density
		double pressure;  // pressure
	} *Particles = 0;  // array of particles
	double t = 0.;  // time
	double max_step = INFINITY;  // maximum simulation step

	// particle properties
	double m;  // mass of each particle
	double h;  // smoothing radius
	double W(double d) {  // smoothing kernel, a function of distance
		double x = abs(d) / h;
		double f = x < 1. ? 1. + x * x*(-1.5 + 0.75*x) : x < 2. ? 0.25*(2. - x)*(2. - x)*(2. - x) : 0.;
		return f / (PI*h*h);
	}
	vec3 W_grad(vec3 d) {  // gradient of the smoothing kernel
		double x = length(d) / h;
		if (!(x > 0.)) return vec3(0.);
		double f = x < 1. ? x * (2.25*x - 3.) : x < 2. ? -0.75*(2. - x)*(2. - x) : 0.;
		return normalize(d) * (f / (PI*h*h*h));
	}

	// forces
	double density_0 = 1.0;  // rest density
	double pressure(double density) {  // calculate pressure from fluid density
		//return 10.*(pow(density / density_0, 7.) - 1.);
		return 100.*(density / density_0 - 1.);
	}
	double viscosity = 0.;  // fluid viscosity, unit: m²/s; a=viscosity*∇²v
	vec3 g = vec3(0, 0, -9.8);  // acceleration due to gravity
	vec3(*Boundary)(vec3, vec3) = 0;  // penalty acceleration field that defines the rigid boundary
	vec3 *Accelerations = 0;  // computed acceleration of each particle

	// neighbor finding
	vec3 Grid_LB;  // lower bottom of the grid starting point
	vec3 Grid_dp;  // size of each grid, should be vec2(2h,2h)
	ivec3 Grid_N;  // dimension of the overall grid, top right Grid_LB+Grid_dp*Grid_N
	const int MAX_PN_G = 31;  // maximum number of particles in each grid
	struct gridcell {
		int N = 0;  // number of associated particles
		int Ps[MAX_PN_G];  // index of particles
	} *Grid = 0;  // grids
	bool isValidGrid(ivec3 p) { return p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x < Grid_N.x && p.y < Grid_N.y && p.z < Grid_N.z; }  // if a grid ID is valid
	gridcell* getGrid(ivec3 p) { return &Grid[(p.z*Grid_N.y + p.y)*Grid_N.x + p.x]; }  // access grid cell
	ivec3 getGridId(vec3 p) { return ivec3(floor((p - Grid_LB) / Grid_dp)); }  // calculate grid from position
	void calcGrid();  // update simulation grid
#if USE_NEIGHBOR
	const int MAX_NEIGHBOR_P = 255;  // maximum number of neighbors of each particle
	struct particle_neighbor {
		int N = 0;  // number of neighbors
		int Ns[MAX_NEIGHBOR_P];  // index of neighbors
	} *Neighbors = 0;
#endif
	void find_neighbors();  // update neighbors for each particle

	// simulation
	void updateScene(double dt);
	void non_iterative_splitting(double dt);

	// surface reconstruction
	std::vector<triangle_3d> Trigs;
	void reconstructSurface();

};  // namespace SPH


	// re-calculate SPH simulation grid
void SPH::calcGrid() {
	// calculate the bounding box of fluid particles
	vec3 LB(INFINITY), TR(-INFINITY);
	for (int i = 0; i < N; i++) {
		LB = min(LB, Particles[i].p);
		TR = max(TR, Particles[i].p);
	}
#if 1
	LB = pMax(LB, vec3(-2, -2, -2)), TR = pMin(TR, vec3(5, 4, 4));
#endif
	LB -= Grid_dp, TR += Grid_dp;
	// update Grid_LB and Grid_N
	Grid_LB = LB;
	ivec3 Grid_N_old = Grid_N;
	Grid_N = (ivec3)ceil((TR - LB) / Grid_dp);
	// create grid
	gridcell *g_del = Grid;
#if 0
	if (Grid_N != Grid_N_old || Grid == 0) Grid = new gridcell[Grid_N.x*Grid_N.y*Grid_N.z];
	else g_del = nullptr;
#else
	Grid = new gridcell[Grid_N.x*Grid_N.y*Grid_N.z];
#endif
	for (int i = 0; i < N; i++) {
		ivec3 gi = getGridId(Particles[i].p);
		if (isValidGrid(gi)) {
			gridcell* g = getGrid(gi);
			if (g->N < MAX_PN_G) g->Ps[g->N++] = i;
		}
	}
	if (g_del) delete g_del;
}

// find neighbors of each particle
void SPH::find_neighbors() {
	calcGrid();

#if USE_NEIGHBOR
	// resolve multithreading issues
	particle_neighbor *Neighbors_del = Neighbors;
	particle_neighbor *Neighbors_new = new particle_neighbor[N];
	Neighbors = Neighbors_new;
	if (Neighbors_del) delete Neighbors_del;

	// find neighbors for each particle
	for (int i = 0; i < N; i++) {
		particle_neighbor *nb = &Neighbors[i];
		nb->N = 0;
		auto addGrid = [&](ivec3 gi) {
			if (!isValidGrid(gi)) return;
			gridcell *g = getGrid(gi);
			for (int ju = 0; ju < g->N; ju++) {
				int j = g->Ps[ju];
				if (//j != i &&
					length(Particles[j].p - Particles[i].p) < 2.*h &&
					nb->N < MAX_NEIGHBOR_P) {
					nb->Ns[nb->N++] = j;
				}
			}
		};
		ivec3 g = getGridId(Particles[i].p);
		for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) for (int k = -1; k <= 1; k++)
			addGrid(g + ivec3(i, j, k));
	}
#endif
}


void SPH::updateScene(double dt) {
	// split large steps
	if (dt > max_step) {
		int N = (int)(dt / max_step + 1);
		for (int i = 0; i < N; i++) updateScene(dt / N);
		return;
	}

	if (!Accelerations) Accelerations = new vec3[N];

	non_iterative_splitting(dt);  // slower but less "particle effect"

	SPH::t += dt;

}

// SPH with state equation and splitting
void SPH::non_iterative_splitting(double dt) {
	find_neighbors();

	// calculate the acceleration caused by other forces
	for (int i = 0; i < N; i++)
		Accelerations[i] = g + Boundary(Particles[i].p, Particles[i].v);

	// calculate the density at each particle
	for (int i = 0; i < N; i++) {
		double density = 1e-12;
#if USE_NEIGHBOR
		for (int _ = 0; _ < Neighbors[i].N; _++) {
			int j = Neighbors[i].Ns[_];
			density += m * W(length(Particles[i].p - Particles[j].p));
		}
#else
		ivec3 g = getGridId(Particles[i].p);
		for (int gi = g.x - 1; gi <= g.x + 1; gi++) for (int gj = g.y - 1; gj <= g.y + 1; gj++) for (int gk = g.z - 1; gk <= g.z + 1; gk++) {
			if (isValidGrid(ivec3(gi, gj, gk))) {
				gridcell *g1 = getGrid(ivec3(gi, gj, gk));
				for (int j = 0; j < g1->N; j++) {
					density += m * W(length(Particles[g1->Ps[j]].p - Particles[i].p));
				}
			}
		}
#endif
		Particles[i].density = density;
	}

	// calculate the acceleration caused by viscosity and update velocity
	for (int i = 0; i < N; i++) {
		vec3 lap_v(0.);
#if USE_NEIGHBOR
		for (int u = 0; u < Neighbors[i].N; u++) {
			int j = Neighbors[i].Ns[u];
			vec3 xij = Particles[i].p - Particles[j].p, vij = Particles[i].v - Particles[j].v;
			double rhoj = Particles[j].density;
			vec3 W_grad_ij = W_grad(xij);
			lap_v += (2.*m / rhoj) * vij * (dot(xij, W_grad_ij) / (xij.sqr() + 0.01*h*h));
		}
#else
		ivec3 g = getGridId(Particles[i].p);
		for (int gi = g.x - 1; gi <= g.x + 1; gi++) for (int gj = g.y - 1; gj <= g.y + 1; gj++) for (int gk = g.z - 1; gk <= g.z + 1; gk++) {
			if (isValidGrid(ivec3(gi, gj, gk))) {
				gridcell *g1 = getGrid(ivec3(gi, gj, gk));
				for (int ji = 0; ji < g1->N; ji++) {
					int j = g1->Ps[ji];
					vec3 xij = Particles[i].p - Particles[j].p, vij = Particles[i].v - Particles[j].v;
					double rhoj = Particles[j].density;
					vec3 W_grad_ij = W_grad(xij);
					lap_v += (2.*m / rhoj) * vij * (dot(xij, W_grad_ij) / (xij.sqr() + 0.01*h*h));
				}
			}
		}
#endif
		Accelerations[i] += viscosity * lap_v;
		Particles[i].v += Accelerations[i] * dt;
	}

	// calculate the temp density and pressure
	double *rho_temp = new double[N];
	for (int i = 0; i < N; i++) {
		double density = 1e-12;
#if USE_NEIGHBOR
		for (int _ = 0; _ < Neighbors[i].N; _++) {
			int j = Neighbors[i].Ns[_];
			density += m * W(length(Particles[i].p - Particles[j].p));
			density += m * dot((Particles[i].v - Particles[j].v) * dt, W_grad(Particles[i].p - Particles[j].p));
		}
#else
		ivec3 g = getGridId(Particles[i].p);
		for (int gi = g.x - 1; gi <= g.x + 1; gi++) for (int gj = g.y - 1; gj <= g.y + 1; gj++) for (int gk = g.z - 1; gk <= g.z + 1; gk++) {
			if (isValidGrid(ivec3(gi, gj, gk))) {
				gridcell *g1 = getGrid(ivec3(gi, gj, gk));
				for (int j = 0; j < g1->N; j++) {
					density += m * W(length(Particles[g1->Ps[j]].p - Particles[i].p));
					density += m * dot((Particles[i].v - Particles[g1->Ps[j]].v) * dt, W_grad(Particles[i].p - Particles[g1->Ps[j]].p));
				}
			}
		}
#endif
		rho_temp[i] = density;
		Particles[i].pressure = pressure(density);
	}

	// calculate the acceleration due to the gradient of pressure
	for (int i = 0; i < N; i++) {
		vec3 grad_P(0.);
#if USE_NEIGHBOR
		for (int u = 0; u < Neighbors[i].N; u++) {
			int j = Neighbors[i].Ns[u];
			vec3 W_grad_ij = W_grad(Particles[i].p - Particles[j].p);
			double rhoi = Particles[i].density, rhoj = Particles[j].density;
			//double rhoi = rho_temp[i], rhoj = rho_temp[j];
			double pi = Particles[i].pressure, pj = Particles[j].pressure;
			grad_P += rhoi * m * (pi / (rhoi*rhoi) + pj / (rhoj*rhoj)) * W_grad_ij;
		}
#else
		ivec3 g = getGridId(Particles[i].p);
		for (int gi = g.x - 1; gi <= g.x + 1; gi++) for (int gj = g.y - 1; gj <= g.y + 1; gj++) for (int gk = g.z - 1; gk <= g.z + 1; gk++) {
			if (isValidGrid(ivec3(gi, gj, gk))) {
				gridcell *g1 = getGrid(ivec3(gi, gj, gk));
				for (int ji = 0; ji < g1->N; ji++) {
					int j = g1->Ps[ji];
					vec3 W_grad_ij = W_grad(Particles[i].p - Particles[j].p);
					double rhoi = Particles[i].density, rhoj = Particles[j].density;
					double pi = Particles[i].pressure, pj = Particles[j].pressure;
					grad_P += rhoi * m * (pi / (rhoi*rhoi) + pj / (rhoj*rhoj)) * W_grad_ij;
				}
			}
		}
#endif
		Accelerations[i] = -grad_P / rho_temp[i];
	}

	// update position and velocity
	for (int i = 0; i < N; i++) {
		Particles[i].v += Accelerations[i] * dt;
		Particles[i].p += Particles[i].v * dt;
	}
	delete rho_temp;
}



// surface reconstruction
void SPH::reconstructSurface() {

	// calculate the bounding box of fluid particles
	vec3 p0(INFINITY), p1(-INFINITY);
	for (int i = 0; i < N; i++) {
		p0 = min(p0, Particles[i].p);
		p1 = max(p1, Particles[i].p);
	}
	p0 -= vec3(2.0*h), p1 += vec3(2.0*h);

	// initialize a grid for density field
	const vec3 dp = vec3(0.5*h);
	ivec3 pn = ivec3((p1 - p0) / dp) + ivec3(1);
	p1 = p0 + vec3(pn) * dp;
	double*** dens = new double**[pn.z];
	for (int k = 0; k < pn.z; k++) {
		dens[k] = new double*[pn.y];
		for (int j = 0; j < pn.y; j++) {
			dens[k][j] = new double[pn.x];
			for (int i = 0; i < pn.x; i++) dens[k][j][i] = 1.0;
		}
	}

	// calculate the density field
	for (int i = 0; i < N; i++) {
		vec3 pf = Particles[i].p, p = (pf - p0) / dp;
		vec3 r = vec3(2.0*h) / dp;
		ivec3 q0 = max(ivec3(0), ivec3(floor(p - r - vec3(1.))));
		ivec3 q1 = min(pn - ivec3(1), ivec3(ceil(p + r + vec3(1.))));
		//printf("%d %d %d  %d %d %d\n", q0.x, q0.y, q0.z, q1.x, q1.y, q1.z);
		for (int k = q0.z; k <= q1.z; k++) for (int j = q0.y; j <= q1.y; j++) for (int i = q0.x; i <= q1.x; i++) {
			vec3 qf = p0 + vec3(i, j, k) * dp;
			dens[k][j][i] -= m * W(length(qf - pf));
		}
	}

	// surface reconstruction
	std::vector<triangle_3d> ts = ScalarFieldTriangulator_octatree::marching_cube<double, vec3, triangle_3d>(dens, 0.5*density_0, pn.z, pn.y, pn.x);
	for (int i = 0; i < (int)ts.size(); i++) for (int u = 0; u < 3; u++)
		ts[i][u] = p0 + ts[i][u] * dp;
	Trigs = ts;

	// clean up
	for (int k = 0; k < pn.z; k++) {
		for (int j = 0; j < pn.y; j++) {
			delete dens[k][j];
		}
		delete dens[k];
	}
	delete dens;
}



// ============================================ Rendering ============================================

const vec3 light = normalize(vec3(0.2, 0.2, 1.0));

#pragma region Rasterization functions

typedef unsigned char byte;
COLORREF toCOLORREF(vec3 c) {
	COLORREF r = 0; byte *k = (byte*)&r;
	k[0] = byte(255 * clamp(c.z, 0, 1));
	k[1] = byte(255 * clamp(c.y, 0, 1));
	k[2] = byte(255 * clamp(c.x, 0, 1));
	return r;
}

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
}
void drawCross(vec2 p, double r, COLORREF Color = 0xFFFFFF) {
	drawLine(p - vec2(r, 0), p + vec2(r, 0), Color);
	drawLine(p - vec2(0, r), p + vec2(0, r), Color);
}
void drawTriangle(vec2 A, vec2 B, vec2 C, COLORREF col, bool stroke = false, COLORREF strokecol = 0xFFFFFF) {
	int x0 = max((int)min(min(A.x, B.x), C.x), 0), x1 = min((int)max(max(A.x, B.x), C.x), _WIN_W - 1);
	int y0 = max((int)min(min(A.y, B.y), C.y), 0), y1 = min((int)max(max(A.y, B.y), C.y), _WIN_H - 1);
	for (int i = y0; i <= y1; i++) for (int j = x0; j <= x1; j++) {
		// the slow way
		vec2 P(j, i);
		if (((det(P - A, P - B) < 0) + (det(P - B, P - C) < 0) + (det(P - C, P - A) < 0)) % 3 == 0)
			Canvas(j, i) = col;
	}
	if (stroke) {
		drawLine(A, B, strokecol); drawLine(A, C, strokecol); drawLine(B, C, strokecol);
	}
	// To-do: write a nasty-but-quick triangle rasterizer
}
void drawBox(vec2 Min, vec2 Max, COLORREF col = 0xFF0000) {
	drawLine(vec2(Min.x, Min.y), vec2(Max.x, Min.y), col);
	drawLine(vec2(Max.x, Min.y), vec2(Max.x, Max.y), col);
	drawLine(vec2(Max.x, Max.y), vec2(Min.x, Max.y), col);
	drawLine(vec2(Min.x, Max.y), vec2(Min.x, Min.y), col);
}
void fillBox(vec2 Min, vec2 Max, COLORREF col = 0xFF0000) {
	int x0 = max((int)Min.x, 0), x1 = min((int)Max.x, _WIN_W - 1);
	int y0 = max((int)Min.y, 0), y1 = min((int)Max.y, _WIN_H - 1);
	for (int x = x0; x <= x1; x++) for (int y = y0; y <= y1; y++) Canvas(x, y) = col;
}
void drawSquare(vec2 C, double r, COLORREF col = 0xFFA500) {
	drawBox(C - vec2(r, r), C + vec2(r, r), col);
}
void fillSquare(vec2 C, double r, COLORREF col = 0xFFA500) {
	fillBox(C - vec2(r, r), C + vec2(r, r), col);
}

void drawLine_F(vec3 A, vec3 B, COLORREF col = 0xFFFFFF) {
	double u = dot(Tr.p, A) + Tr.s, v = dot(Tr.p, B) + Tr.s;
	if (u > 0 && v > 0) { drawLine((Tr*A).xy(), (Tr*B).xy(), col); return; }
	if (u < 0 && v < 0) return;
	if (u < v) std::swap(A, B), std::swap(u, v);
	double t = u / (u - v) - 1e-6;
	B = A + (B - A)*t;
	drawLine((Tr*A).xy(), (Tr*B).xy(), col);
}


void drawSphere_RT(vec3 P, double r, vec3 col) {
	vec2 p0, p1; projRange_Sphere(P, r, p0, p1);
	int x0 = max((int)p0.x, 0), x1 = min((int)p1.x, _WIN_W - 1), y0 = max((int)p0.y, 0), y1 = min((int)p1.y, _WIN_H - 1);
	double t; for (int i = x0; i <= x1; i++) for (int j = y0; j <= y1; j++) {
		vec3 dir = scrDir(vec2(i, j));
		if ((t = intSphere(P, r, CamP, dir)) > 0 && t < _DEPTHBUF[i][j]) {
			vec3 n = (CamP + t * dir - P) / r;
			vec3 c = col * mix(0.5, 1.0, dot(n, light));
			Canvas(i, j) = toCOLORREF(c), _DEPTHBUF[i][j] = t;
		}
	}
}
void drawTriangle_RT(vec3 A, vec3 B, vec3 C, COLORREF col) {
	vec2 p0, p1; projRange_Triangle(A, B, C, p0, p1);
	int x0 = max((int)p0.x, 0), x1 = min((int)p1.x, _WIN_W - 1), y0 = max((int)p0.y, 0), y1 = min((int)p1.y, _WIN_H - 1);
	double t; for (int i = x0; i <= x1; i++) for (int j = y0; j <= y1; j++)
		if ((t = intTriangle(A, B, C, CamP, scrDir(vec2(i, j)))) > 0 && t < _DEPTHBUF[i][j]) Canvas(i, j) = col, _DEPTHBUF[i][j] = t;
	// can be accelerated by precomputing the triangle edges
}

#pragma endregion


void render() {

	// initialize window
	for (int i = 0, l = _WIN_W * _WIN_H; i < l; i++) _WINIMG[i] = 0;
	for (int i = 0; i < _WIN_W; i++) for (int j = 0; j < _WIN_H; j++) _DEPTHBUF[i][j] = INFINITY;
	calcMat();
	getScreen(CamP, ScrO, ScrA, ScrB);

	// axis and grid
	{
		const double R = 20.0;
		for (int i = -R; i <= R; i++) {
			drawLine_F(vec3(-R, i, 0), vec3(R, i, 0), 0x404040);
			drawLine_F(vec3(i, -R, 0), vec3(i, R, 0), 0x404040);
		}
		drawLine_F(vec3(0, -R, 0), vec3(0, R, 0), 0x409040);
		drawLine_F(vec3(-R, 0, 0), vec3(R, 0, 0), 0xC04040);
		//drawLine_F(vec3(0, 0, -.6*R), vec3(0, 0, .6*R), 0x4040FF);
	}

#if RECONSTRUCT_SURFACE
	// draw surface
	for (int i = 0; i < (int)SPH::Trigs.size(); i++) {
		triangle_3d t = SPH::Trigs[i];
		vec3 n = ncross(t[1] - t[0], t[2] - t[0]);
		vec3 col = mix(vec3(0.2), vec3(0.6, 0.6, 1.0), 0.5*dot(n, light) + 0.5);
		drawTriangle_RT(t[0], t[1], t[2], toCOLORREF(col));
	}
#else
	// draw SPH particles
	for (int i = 0; i < SPH::N; i++) {
		//vec3 col = ColorFunctions::ThermometerColors(clamp(SPH::Particles[i].density, 0., 1.));
		vec3 col = ColorFunctions<vec3, double>::TemperatureMap(tanh(0.1*length(SPH::Particles[i].v)));
		drawSphere_RT(SPH::Particles[i].p, 0.5*SPH::h, col);
	}
#endif

	// momentum and energy calculation
	vec3 P = vec3(0.);  // total momentum
	double Eg = 0., Ek = 0.;  // gravitational potential and kinetic
	for (int i = 0; i < SPH::N; i++) {
		double m = SPH::m;
		Eg += -m * dot(SPH::Particles[i].p, SPH::g);
		Ek += 0.5 * m * SPH::Particles[i].v.sqr();
		P += m * SPH::Particles[i].v;
		// should also add potential energy due to the boundary
	}

	char text[1024];
	sprintf(text, "%d particles  t=%.3lf   E=Eg+Ek=%.3lg+%.3lg=%.3lg", SPH::N, SPH::t, Eg, Ek, Eg + Ek);
	SetWindowTextA(_HWND, text);
}


// ============================================== User ==============================================


// preset scenes
class presetScenes {

public:  // boundary functions

	struct boundary_functions {

		static vec3 box_212(vec3 p, vec3 v) {
			return 1000.*vec3(
				p.x<0. ? -p.x + 1. : p.x>2. ? -(p.x - 2.) - 1. : 0.,
				p.y<0. ? -p.y + 1. : p.y>1. ? -(p.y - 1.) - 1. : 0.,
				p.z<0. ? -p.z + 1. : p.z>2. ? -(p.z - 2.) - 1. : 0.
			);
		}

		static vec3 box_block(vec3 p, vec3 v) {
			auto E = [](double x, double y, double z)->double {
				double E = max(max(
					std::max({ abs(x - 1.5) - 1.5, abs(y - 0.5) - 0.5, abs(z - 1.) - 1. }),  // room
					-std::max({ abs(x - 2.0) - 0.2, y - 0.6, z - 0.2 })  // block
				), 0.0);
				return E * E;
			};
			const double eps = 1e-5;
			return -1e5 * vec3(
				E(p.x + eps, p.y, p.z) - E(p.x - eps, p.y, p.z),
				E(p.x, p.y + eps, p.z) - E(p.x, p.y - eps, p.z),
				E(p.x, p.y, p.z + eps) - E(p.x, p.y, p.z - eps)
			) / (2.0*eps);
		}

		static vec3 box_rod(vec3 p, vec3 v) {
			auto E = [](double x, double y, double z)->double {
				return max(max(
					std::max({ abs(x - 1.0) - 1.0, abs(y - 0.5) - 0.5, abs(z - 0.8) - 0.8 }),  // room
					0.1 - length(vec2(x, z) - vec2(1.5, 0.15))  // rod
				), 0.);
			};
			const double eps = 0.01;
			return -1e4 * vec3(
				E(p.x + eps, p.y, p.z) - E(p.x - eps, p.y, p.z),
				E(p.x, p.y + eps, p.z) - E(p.x, p.y - eps, p.z),
				E(p.x, p.y, p.z + eps) - E(p.x, p.y, p.z - eps)
			) / (2.0*eps);
		}

		static vec3 cylinder(vec3 p, vec3 v) {
			auto E = [](double x, double y, double z)->double {
				return max(max(
					length(vec2(x, y) - vec2(0.5, 0.5)) - sqrt(0.5),
					abs(z - 1.0) - 1.0
				), 0.);
			};
			const double eps = 0.01;
			return -1e4 * vec3(
				E(p.x + eps, p.y, p.z) - E(p.x - eps, p.y, p.z),
				E(p.x, p.y + eps, p.z) - E(p.x, p.y - eps, p.z),
				E(p.x, p.y, p.z + eps) - E(p.x, p.y, p.z - eps)
			) / (2.0*eps);
		}

		static vec3 sphere(vec3 p, vec3 v) {
			p = p - vec3(0.5, 0.5, 0.5);
			return -10000. * max(length(p) - sqrt(0.75), 0.) * normalize(p);
		}

	};

public:  // preset scenes

	static void Scene_random(int N, double h, vec3 v0 = vec3(0.)) {
		SPH::h = h;
		SPH::N = N;
		SPH::m = SPH::density_0 / (6.*SPH::W(h) + 8.*SPH::W(1.732051*h) + SPH::W(0.));
		SPH::Particles = new SPH::particle[N];
		uint32_t seed = 0;
		for (int i = 0; i < N; i++) {
			vec3 p = vec3(2.*rand01(seed), rand01(seed), rand01(seed));
			int j; for (j = 0; j < i; j++) {
				if (length(SPH::Particles[j].p - p) < 2.*h) {
					i--; break;
				}
			}
			if (j == i)
				SPH::Particles[i] = SPH::particle{ p, v0 };
		}
		SPH::Boundary = boundary_functions::box_212;
		SPH::Grid_LB = vec3(0, 0, 0);
		SPH::Grid_dp = vec3(2.*h);
		SPH::Grid_N = ivec3(ceil((vec3(2, 1, 1) - SPH::Grid_LB) / SPH::Grid_dp));
	}

	static void Scene_left(double dist) {
		using namespace SPH;
		h = dist;
		m = SPH::density_0 / (6.*SPH::W(h) + 8.*SPH::W(1.732051*h) + SPH::W(0.));
		int xN = (int)floor(0.4999 / dist);
		int yN = (int)floor(0.6999 / dist);
		int zN = (int)floor(1.1999 / dist);
		Particles = new particle[xN*yN*zN];
		N = 0;
		for (int i = 0; i < xN; i++) {
			for (int j = 0; j < yN; j++) {
				for (int k = 0; k < zN; k++) {
					Particles[N++] = particle{
						vec3((i + 0.5)*dist, (j + 0.5)*dist, (k + 0.5)*dist),
						vec3(0.0)
					};
				}
			}
		}
		viscosity = 1e-4;
		Boundary = boundary_functions::box_212;
		Grid_LB = vec3(0, 0, 0);
		Grid_dp = vec3(2.*h);
		Grid_N = ivec3(ceil((vec3(2, 1, 1) - Grid_LB) / Grid_dp));
	}

	static void Scene_right(double dist) {
		using namespace SPH;
		h = dist;
		m = SPH::density_0 / (6.*SPH::W(h) + 8.*SPH::W(1.732051*h) + SPH::W(0.));
		int xN = (int)floor(0.4999 / dist);
		int yN = (int)floor(0.5999 / dist);
		int zN = (int)floor(0.9999 / dist);
		Particles = new particle[xN*yN*zN];
		N = 0;
		vec3 p0 = vec3(1.2, 0.1, 0);
		for (int k = 0; k < zN; k++) {
			for (int j = 0; j < yN; j++) {
				for (int i = 0; i < xN; i++) {
					Particles[N++] = particle{
						p0 + vec3((i + 0.5)*dist, (j + 0.5)*dist, (k + 0.5)*dist),
						vec3(0.0)
					};
				}
			}
		}
		viscosity = 1e-4;
		Boundary = boundary_functions::box_212;
		Grid_LB = vec3(0, 0, 0);
		Grid_dp = vec3(2.*h);
		Grid_N = ivec3(ceil((vec3(2, 1, 1) - Grid_LB) / Grid_dp));
	}

	static void Scene_fall(double dist) {
		using namespace SPH;
		h = dist;
		m = 0.9 * SPH::density_0 / (6.*SPH::W(h) + 8.*SPH::W(1.732051*h) + SPH::W(0.));
		int xN = (int)floor(2.0 / dist);
		int yN = (int)floor(1.0 / dist);
		int zN = (int)floor(1.0 / dist);
		std::vector<vec3> particles;
		for (int i = 0; i < xN; i++) {
			for (int j = 0; j < yN; j++) {
				for (int k = 0; k < zN; k++) {
					vec3 p = vec3(i, j, k)*dist;
					double dist = length(p - vec3(1, 0.5, 0.8)) - 0.2;
					dist = min(dist, p.z - 0.2);
					if (dist < 0.) particles.push_back(p);
				}
			}
		}
		N = (int)particles.size();
		Particles = new particle[N];
		for (int i = 0; i < N; i++) {
			vec3 p = particles[i];
			Particles[i] = particle{ p, vec3(0.0) };
		}
		viscosity = 1e-4;
		Boundary = boundary_functions::box_212;
		Grid_LB = vec3(0, 0, 0);
		Grid_dp = vec3(2.*h);
		Grid_N = ivec3(ceil((vec3(2, 1, 1) - Grid_LB) / Grid_dp));
	}

};


void Init() {
	//presetScenes::Scene_random(200, 0.1, vec3(1, 0, 0));
	//presetScenes::Scene_left(0.06);
	presetScenes::Scene_left(0.04);
	//presetScenes::Scene_left(0.03);
	//presetScenes::Scene_right(0.04);
	//presetScenes::Scene_fall(0.05);

	SPH::max_step = SPH::h >= 0.04 ? 0.002 : 0.001;
	//SPH::m *= 0.9;

	//SPH::Boundary = presetScenes::boundary_functions::box_block;
	SPH::viscosity = 0.01;

	Center = SPH::Grid_LB + 0.5*vec3(SPH::Grid_N)*SPH::Grid_dp;

	// simulation thread
	new std::thread([]() {
		SPH::reconstructSurface();
		SPH::Trigs.reserve(16 * SPH::Trigs.size());
		auto t0 = std::chrono::high_resolution_clock::now();
		const double dt = 0.04;
		for (int step = 0; ; step++) {
			double time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

			if (RECONSTRUCT_SURFACE) {
				SPH::reconstructSurface();
				if (0) {
					char filename[64]; sprintf(filename, "simulation/fluid/sph3d/%04d.stl", step);
					writeSTL<triangle_3d>(filename, &SPH::Trigs[0], (int)SPH::Trigs.size());
				}
			}
			SPH::updateScene(dt);

			double time_elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() - time;
			printf("(%lf,%lf),", time, time_elapsed / dt);
			if (time_elapsed < dt) std::this_thread::sleep_for(std::chrono::milliseconds((int)(1e3 * (dt - time_elapsed))));
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
	if (_W*_H == 0 || _oldW * _oldH == 0) return;  // window is minimized
	double pw = _oldW, ph = _oldH, w = _W, h = _H;
	double s = sqrt((w * h) / (pw * ph));
	Unit *= s, dist /= s;
	Render_Needed = true;
}
void WindowClose() {
}

void MouseWheel(int _DELTA) {
	Render_Needed = true;
	if (Ctrl) Center.z += 0.1 * _DELTA / Unit;
	else if (Shift) dist *= exp(-0.001*_DELTA);
	else {
		double s = exp(0.001*_DELTA);
		double D = length(vec2(_WIN_W, _WIN_H)), Max = D, Min = 0.015*D;
		//if (Unit * s > Max) s = Max / Unit; else if (Unit * s < Min) s = Min / Unit;
		Unit *= s, dist /= s;
	}
}
void MouseDownL(int _X, int _Y) {
	clickCursor = Cursor = vec2(_X, _Y);
	mouse_down = true;
	Render_Needed = true;

}
void MouseMove(int _X, int _Y) {
	vec2 P0 = Cursor, P = vec2(_X, _Y), D = P - P0;
	Cursor = P;

	// drag to rotate scene
	if (mouse_down) {
		Render_Needed = true;
		if (Ctrl) {
			vec3 d = scrDir(P0);
			vec3 p = CamP.z / d.z * d;
			d = scrDir(P);
			vec3 q = CamP.z / d.z * d;
			Center += q - p;
		}
		else if (Shift) {
			ry += 0.005*D.y;
		}
		else {
			vec2 d = 0.01*D;
			rz -= cos(ry)*d.x + sin(ry)*d.y, rx -= -sin(ry)*d.x + cos(ry)*d.y;  // doesn't work very well
			//rz -= d.x, rx -= d.y;
		}
	}

}
void MouseUpL(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	bool moved = (int)length(clickCursor - Cursor) != 0;   // be careful about coincidence
	mouse_down = false;
	Render_Needed = true;
}
void MouseDownR(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	Render_Needed = true;
}
void MouseUpR(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	bool topmost = GetWindowLong(_HWND, GWL_EXSTYLE) & WS_EX_TOPMOST;
	SetWindowPos(_HWND, topmost ? HWND_NOTOPMOST : HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
	Render_Needed = true;
}
void KeyDown(WPARAM _KEY) {
	if (_KEY == VK_CONTROL) Render_Needed = !Ctrl, Ctrl = true;
	else if (_KEY == VK_SHIFT) Render_Needed = !Shift, Shift = true;
	else if (_KEY == VK_MENU) Render_Needed = !Alt, Alt = true;
}
void KeyUp(WPARAM _KEY) {
	if (_KEY == VK_CONTROL) Ctrl = false;
	else if (_KEY == VK_SHIFT) Shift = false;
	else if (_KEY == VK_MENU) Alt = false;
	Render_Needed = true;

	if (_KEY == VK_HOME || (Ctrl && (_KEY == '0' || _KEY == VK_NUMPAD0))) {
		Center = vec3(0.0, 0.0, 0.0);
		rz = 0.2*PI, rx = 0.15*PI, ry = 0.0, dist = 12.0, Unit = 100.0;
	}
}

