// Win32 GUI 2D template

 // ========================================= Win32 GUI =========================================

#pragma region Windows

#include <Windows.h>
#include <windowsx.h>
#include <tchar.h>

#define WIN_NAME "SPH 2D"
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

double _DEPTHBUF[WinW_Max][WinH_Max];

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
#include <vector>

#pragma region Window Variables

// window parameters
char text[64];	// window title
vec2 Origin = vec2(0, 0);	// origin in screen coordinate
double Unit = 100.0;		// screen unit to object unit
#define fromInt(p) (((p) - Origin) * (1.0 / Unit))
#define fromFloat(p) ((p) * Unit + Origin)

// user parameters
vec2 Cursor = vec2(0, 0);
bool mouse_down = false;
bool Ctrl = false, Shift = false, Alt = false;

// forward declarations of rendering functions
void drawLine(vec2 p, vec2 q, COLORREF col);
void drawBox(vec2 p0, vec2 p1, COLORREF col);
void drawCircle(vec2 c, double r, COLORREF col);
void fillBox(vec2 p0, vec2 p1, COLORREF col);
void fillCircle(vec2 c, double r, COLORREF col);
void drawTriangle(vec2 A, vec2 B, vec2 C, COLORREF col);
void fillTriangle(vec2 A, vec2 B, vec2 C, COLORREF col);
#define drawLineF(p,q,col) drawLine(fromFloat(p),fromFloat(q),col)
#define drawBoxF(p0,p1,col) drawBox(fromFloat(p0),fromFloat(p1),col)
#define drawCircleF(c,r,col) drawCircle(fromFloat(c),(r)*Unit,col)
#define fillBoxF(p0,p1,col) fillBox(fromFloat(p0),fromFloat(p1),col)
#define fillCircleF(c,r,col) fillCircle(fromFloat(c),(r)*Unit,col)
#define drawDotF(c,r,col) fillCircle(fromFloat(c),r,col)  // r: in screen coordinate
#define drawDotSquareF(c,r,col) fillBox(fromFloat(c)-vec2(r),fromFloat(c)+vec2(r),col)  // r: in screen coordinate
#define drawDotHollowF(c,r,col) drawCircle(fromFloat(c),r,col)  // r: in screen coordinate
#define drawSquareHollowF(c,r,col) drawBox(fromFloat(c)-vec2(r),fromFloat(c)+vec2(r),col)  // r: in screen coordinate
#define drawTriangleF(A,B,C,col) drawTriangle(fromFloat(A),fromFloat(B),fromFloat(C),col)
#define fillTriangleF(A,B,C,col) fillTriangle(fromFloat(A),fromFloat(B),fromFloat(C),col)

#pragma endregion  // Window variables


// faster when set to true
#define USE_NEIGHBOR 1

namespace SPH {

	// scene
	int N = 0;  // number of SPH particles
	struct particle {
		vec2 p;  // position
		vec2 v;  // velocity
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
		return f / (PI*h*h) * (10. / 7.);
	}
	vec2 W_grad(vec2 d) {  // gradient of the smoothing kernel
		double x = length(d) / h;
		if (!(x > 0.)) return vec2(0.);
		double f = x < 1. ? x * (2.25*x - 3.) : x < 2. ? -0.75*(2. - x)*(2. - x) : 0.;
		return normalize(d) * (f / (PI*h*h*h) * (10. / 7.));
	}

	// forces
	double density_0 = 1.0;  // rest density
	double pressure(double density) {  // calculate pressure from fluid density
		//return 10.*(pow(density / density_0, 7.) - 1.);
		return 100.*(density / density_0 - 1.);
	}
	double viscosity = 0.;  // fluid viscosity, unit: m²/s; a=viscosity*∇²v
	vec2 g = vec2(0, -9.8);  // acceleration due to gravity
	vec2(*Boundary)(vec2, vec2) = 0;  // penalty acceleration field that defines the rigid boundary
	vec2 *Accelerations = 0;  // computed acceleration of each particle

	// neighbor finding
	vec2 Grid_LB;  // lower bottom of the grid starting point
	vec2 Grid_dp;  // size of each grid, should be vec2(2h,2h)
	ivec2 Grid_N;  // dimension of the overall grid, top right Grid_LB+Grid_dp*Grid_N
	const int MAX_PN_G = 15;  // maximum number of particles in each grid
	struct gridcell {
		int N = 0;  // number of associated particles
		int Ps[MAX_PN_G];  // index of particles
	} *Grid = 0;  // grids
	bool isValidGrid(ivec2 p) { return p.x >= 0 && p.y >= 0 && p.x < Grid_N.x && p.y < Grid_N.y; }  // if a grid ID is valid
	gridcell* getGrid(ivec2 p) { return &Grid[p.y*Grid_N.x + p.x]; }  // access grid cell
	ivec2 getGridId(vec2 p) { return ivec2(floor((p - Grid_LB) / Grid_dp)); }  // calculate grid from position
	void calcGrid();  // update simulation grid
#if USE_NEIGHBOR
	const int MAX_NEIGHBOR_P = 63;  // maximum number of neighbors of each particle
	struct particle_neighbor {
		int N = 0;  // number of neighbors
		int Ns[MAX_NEIGHBOR_P];  // index of neighbors
	} *Neighbors = 0;
#endif
	void find_neighbors();  // update neighbors for each particle

	// simulation
	void updateScene(double dt);
	void non_iterative_state_equation(double dt);
	void non_iterative_splitting(double dt);
	void iterative_splitting(double dt, double eta);

};  // namespace SPH



// ============================================ Simulation ============================================

// re-calculate SPH simulation grid
void SPH::calcGrid() {
	// calculate the bounding box of fluid particles
	vec2 LB(INFINITY), TR(-INFINITY);
	for (int i = 0; i < N; i++) {
		LB = pMin(LB, Particles[i].p);
		TR = pMax(TR, Particles[i].p);
	}
	LB = pMax(LB, vec2(-2, -2)), TR = pMin(TR, vec2(5, 4));
	LB -= Grid_dp, TR += Grid_dp;
	if (0) {
		const double R = 3.;
		TR.x = clamp(TR.x, -R, R); TR.y = clamp(TR.y, -R, R);
		LB.x = clamp(LB.x, -R, R); LB.y = clamp(LB.y, -R, R);
	}
	// update Grid_LB and Grid_N
	Grid_LB = LB;
	ivec2 Grid_N_old = Grid_N;
	Grid_N = (ivec2)ceil((TR - LB) / Grid_dp);
	// create grid
	gridcell *g_del = Grid;
#if 0
	if (Grid_N != Grid_N_old || Grid == 0) Grid = new gridcell[Grid_N.x*Grid_N.y];
	else g_del = nullptr;
#else
	Grid = new gridcell[Grid_N.x*Grid_N.y];
#endif
	for (int i = 0; i < N; i++) {
		ivec2 gi = getGridId(Particles[i].p);
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
		auto addGrid = [&](ivec2 gi) {
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
		ivec2 g = getGridId(Particles[i].p);
		addGrid(g);
		addGrid(g + ivec2(-1, 0));
		addGrid(g + ivec2(1, 0));
		addGrid(g + ivec2(0, -1));
		addGrid(g + ivec2(0, 1));
		addGrid(g + ivec2(-1, -1));
		addGrid(g + ivec2(-1, 1));
		addGrid(g + ivec2(1, -1));
		addGrid(g + ivec2(1, 1));
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

	if (!Accelerations) Accelerations = new vec2[N];

	//non_iterative_state_equation(dt);
	non_iterative_splitting(dt);  // slower but less "particle effect"
	//iterative_splitting(dt, 0.01);  // not sure if I implemented it correctly but it isn't more stable

	SPH::t += dt;

}

// SPH with state equation
void SPH::non_iterative_state_equation(double dt) {
	find_neighbors();

	// calculate the density and pressure at each particle
	for (int i = 0; i < N; i++) {
		double density = 1e-12;
#if USE_NEIGHBOR
		for (int _ = 0; _ < Neighbors[i].N; _++) {
			int j = Neighbors[i].Ns[_];
			density += m * W(length(Particles[j].p - Particles[i].p));
		}
#else
		ivec2 g = getGridId(Particles[i].p);
		for (int gi = g.x - 1; gi <= g.x + 1; gi++) for (int gj = g.y - 1; gj <= g.y + 1; gj++) {
			if (isValidGrid(ivec2(gi, gj))) {
				gridcell *g1 = getGrid(ivec2(gi, gj));
				for (int j = 0; j < g1->N; j++) {
					density += m * W(length(Particles[g1->Ps[j]].p - Particles[i].p));
				}
			}
		}
#endif
		Particles[i].density = density;
		Particles[i].pressure = pressure(density);
	}

	// compute the acceleration of each particle
	for (int i = 0; i < N; i++) {
		Accelerations[i] = g + Boundary(Particles[i].p, Particles[i].v);
	}
	for (int i = 0; i < N; i++) {
		// estimate the gradient of pressure and the element-wise laplacian of velocity
		vec2 grad_P(0.), lap_v(0.);
#if USE_NEIGHBOR
		for (int u = 0; u < Neighbors[i].N; u++) {
			int j = Neighbors[i].Ns[u];
			vec2 xij = Particles[i].p - Particles[j].p, vij = Particles[i].v - Particles[j].v;
			double rhoi = Particles[i].density, rhoj = Particles[j].density,
				pi = Particles[i].pressure, pj = Particles[j].pressure;
			vec2 W_grad_ij = W_grad(xij);
			grad_P += rhoi * m * (pi / (rhoi*rhoi) + pj / (rhoj*rhoj)) * W_grad_ij;
			lap_v += (2.*m / rhoj) * vij * (dot(xij, W_grad_ij) / (xij.sqr() + 0.01*h*h));
		}
#else
		ivec2 g = getGridId(Particles[i].p);
		for (int gi = g.x - 1; gi <= g.x + 1; gi++) for (int gj = g.y - 1; gj <= g.y + 1; gj++) {
			if (isValidGrid(ivec2(gi, gj))) {
				gridcell *g1 = getGrid(ivec2(gi, gj));
				for (int ji = 0; ji < g1->N; ji++) {
					int j = g1->Ps[ji];
					vec2 xij = Particles[i].p - Particles[j].p, vij = Particles[i].v - Particles[j].v;
					double rhoi = Particles[i].density, rhoj = Particles[j].density,
						pi = Particles[i].pressure, pj = Particles[j].pressure;
					vec2 W_grad_ij = W_grad(xij);
					grad_P += rhoi * m * (pi / (rhoi*rhoi) + pj / (rhoj*rhoj)) * W_grad_ij;
					lap_v += (2.*m / rhoj) * vij * (dot(xij, W_grad_ij) / (xij.sqr() + 0.01*h*h));
				}
			}
		}
#endif
		// add acceleration due to pressure difference
		Accelerations[i] -= grad_P / Particles[i].density;
		Accelerations[i] += viscosity * lap_v;
	}

	// update position and velocity of each particle
	for (int i = 0; i < N; i++) {
		Particles[i].v += Accelerations[i] * dt;
		Particles[i].p += Particles[i].v * dt;
	}
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
		ivec2 g = getGridId(Particles[i].p);
		for (int gi = g.x - 1; gi <= g.x + 1; gi++) for (int gj = g.y - 1; gj <= g.y + 1; gj++) {
			if (isValidGrid(ivec2(gi, gj))) {
				gridcell *g1 = getGrid(ivec2(gi, gj));
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
		vec2 lap_v(0.);
#if USE_NEIGHBOR
		for (int u = 0; u < Neighbors[i].N; u++) {
			int j = Neighbors[i].Ns[u];
			vec2 xij = Particles[i].p - Particles[j].p, vij = Particles[i].v - Particles[j].v;
			double rhoj = Particles[j].density;
			vec2 W_grad_ij = W_grad(xij);
			lap_v += (2.*m / rhoj) * vij * (dot(xij, W_grad_ij) / (xij.sqr() + 0.01*h*h));
		}
#else
		ivec2 g = getGridId(Particles[i].p);
		for (int gi = g.x - 1; gi <= g.x + 1; gi++) for (int gj = g.y - 1; gj <= g.y + 1; gj++) {
			if (isValidGrid(ivec2(gi, gj))) {
				gridcell *g1 = getGrid(ivec2(gi, gj));
				for (int ji = 0; ji < g1->N; ji++) {
					int j = g1->Ps[ji];
					vec2 xij = Particles[i].p - Particles[j].p, vij = Particles[i].v - Particles[j].v;
					double rhoj = Particles[j].density;
					vec2 W_grad_ij = W_grad(xij);
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
		ivec2 g = getGridId(Particles[i].p);
		for (int gi = g.x - 1; gi <= g.x + 1; gi++) for (int gj = g.y - 1; gj <= g.y + 1; gj++) {
			if (isValidGrid(ivec2(gi, gj))) {
				gridcell *g1 = getGrid(ivec2(gi, gj));
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
		vec2 grad_P(0.);
#if USE_NEIGHBOR
		for (int u = 0; u < Neighbors[i].N; u++) {
			int j = Neighbors[i].Ns[u];
			vec2 W_grad_ij = W_grad(Particles[i].p - Particles[j].p);
			double rhoi = Particles[i].density, rhoj = Particles[j].density;
			//double rhoi = rho_temp[i], rhoj = rho_temp[j];
			double pi = Particles[i].pressure, pj = Particles[j].pressure;
			grad_P += rhoi * m * (pi / (rhoi*rhoi) + pj / (rhoj*rhoj)) * W_grad_ij;
		}
#else
		ivec2 g = getGridId(Particles[i].p);
		for (int gi = g.x - 1; gi <= g.x + 1; gi++) for (int gj = g.y - 1; gj <= g.y + 1; gj++) {
			if (isValidGrid(ivec2(gi, gj))) {
				gridcell *g1 = getGrid(ivec2(gi, gj));
				for (int ji = 0; ji < g1->N; ji++) {
					int j = g1->Ps[ji];
					vec2 W_grad_ij = W_grad(Particles[i].p - Particles[j].p);
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

// Iterative equation of state solver with splitting
void SPH::iterative_splitting(double dt, double eta) {
	find_neighbors();

	// calculate the acceleration caused by other forces
	for (int i = 0; i < N; i++)
		Accelerations[i] = g + Boundary(Particles[i].p, Particles[i].v);

	// calculate the density at each particle
	for (int i = 0; i < N; i++) {
		double density = 1e-12;
		for (int _ = 0; _ < Neighbors[i].N; _++) {
			int j = Neighbors[i].Ns[_];
			density += m * W(length(Particles[i].p - Particles[j].p));
		}
		Particles[i].density = density;
	}

	// calculate the acceleration caused by viscosity and update velocity/position
	for (int i = 0; i < N; i++) {
		vec2 lap_v(0.);
		for (int u = 0; u < Neighbors[i].N; u++) {
			int j = Neighbors[i].Ns[u];
			vec2 xij = Particles[i].p - Particles[j].p, vij = Particles[i].v - Particles[j].v;
			double rhoj = Particles[j].density;
			vec2 W_grad_ij = W_grad(xij);
			lap_v += (2.*m / rhoj) * vij * (dot(xij, W_grad_ij) / (xij.sqr() + 0.01*h*h));
		}
		Accelerations[i] += viscosity * lap_v;
		Particles[i].v += Accelerations[i] * dt;
		//Particles[i].p += Particles[i].v * dt;
	}

	// iteration
	for (int i = 0; ; i++) {

		// calculate the density and pressure
		for (int i = 0; i < N; i++) {
			double density = 1e-12;
			for (int _ = 0; _ < Neighbors[i].N; _++) {
				int j = Neighbors[i].Ns[_];
				density += m * W(length(Particles[i].p - Particles[j].p));
				//density += m * dot((Particles[i].v - Particles[j].v) * dt, W_grad(Particles[i].p - Particles[j].p));
			}
			Particles[i].density = density;
			Particles[i].pressure = pressure(density);
		}

		// calculate the acceleration due to the gradient of pressure
		for (int i = 0; i < N; i++) {
			vec2 grad_P(0.);
			for (int u = 0; u < Neighbors[i].N; u++) {
				int j = Neighbors[i].Ns[u];
				vec2 W_grad_ij = W_grad(Particles[i].p - Particles[j].p);
				double rhoi = Particles[i].density, rhoj = Particles[j].density;
				double pi = Particles[i].pressure, pj = Particles[j].pressure;
				grad_P += rhoi * m * (pi / (rhoi*rhoi) + pj / (rhoj*rhoj)) * W_grad_ij;
			}
			Accelerations[i] = -grad_P / Particles[i].density;
		}

		// update position and velocity
		for (int i = 0; i < N; i++) {
			Particles[i].v += Accelerations[i] * dt;
			Particles[i].p += Accelerations[i] * dt * dt;
		}

		// check if the error of density is small enough
		double sum_err = 0.;
		for (int i = 0; i < N; i++) {
			double err = Particles[i].density - density_0;
			sum_err += err * err;
		}
		if (sqrt(sum_err / N) < eta*density_0 || i >= 10) {
			break;
		}
		find_neighbors();
	}

	for (int i = 0; i < N; i++) Particles[i].p += Particles[i].v*dt;
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
void drawBox(vec2 p0, vec2 p1, COLORREF col) {
	drawLine(vec2(p0.x, p0.y), vec2(p1.x, p0.y), col);
	drawLine(vec2(p0.x, p1.y), vec2(p1.x, p1.y), col);
	drawLine(vec2(p0.x, p0.y), vec2(p0.x, p1.y), col);
	drawLine(vec2(p1.x, p0.y), vec2(p1.x, p1.y), col);
};
void drawCircle(vec2 c, double r, COLORREF col) {
	int s = int(r / sqrt(2) + 0.5);
	int cx = (int)c.x, cy = (int)c.y;
	for (int i = 0, im = min(s, max(_WIN_W - cx, cx)) + 1; i < im; i++) {
		int u = (int)sqrt(r*r - i * i);
		setColor(cx + i, cy + u, col); setColor(cx + i, cy - u, col); setColor(cx - i, cy + u, col); setColor(cx - i, cy - u, col);
		setColor(cx + u, cy + i, col); setColor(cx + u, cy - i, col); setColor(cx - u, cy + i, col); setColor(cx - u, cy - i, col);
	}
}
void fillBox(vec2 p0, vec2 p1, COLORREF col) {
	int i0 = max(0, (int)p0.x), i1 = min(_WIN_W - 1, (int)p1.x);
	int j0 = max(0, (int)p0.y), j1 = min(_WIN_H - 1, (int)p1.y);
	for (int j = j0; j <= j1; j++) for (int i = i0; i <= i1; i++) {
		Canvas(i, j) = col;
	}
}
void fillCircle(vec2 c, double r, COLORREF col) {
	c -= vec2(0.5);
	int i0 = max(0, (int)floor(c.x - r - 1)), i1 = min(_WIN_W - 1, (int)ceil(c.x + r + 1));
	int j0 = max(0, (int)floor(c.y - r - 1)), j1 = min(_WIN_H - 1, (int)ceil(c.y + r + 1));
	for (int j = j0; j <= j1; j++) for (int i = i0; i <= i1; i++) {
		double d = (vec2(i, j) - c).sqr() - r * r;
		if (d < 0.) Canvas(i, j) = col;
	}
}
void drawTriangle(vec2 A, vec2 B, vec2 C, COLORREF col) {
	drawLine(A, B, col); drawLine(B, C, col); drawLine(C, A, col);
}
void fillTriangle(vec2 A, vec2 B, vec2 C, COLORREF col) {
	int x0 = max((int)min(min(A.x, B.x), C.x), 0), x1 = min((int)max(max(A.x, B.x), C.x), _WIN_W - 1);
	int y0 = max((int)min(min(A.y, B.y), C.y), 0), y1 = min((int)max(max(A.y, B.y), C.y), _WIN_H - 1);
	for (int i = y0; i <= y1; i++) for (int j = x0; j <= x1; j++) {
		vec2 P(j, i);
		if (((det(P - A, P - B) < 0) + (det(P - B, P - C) < 0) + (det(P - C, P - A) < 0)) % 3 == 0)
			Canvas(j, i) = col;
	}
}

#include "ui/color_functions/poly.h"

void render() {
	// debug
	auto t1 = std::chrono::high_resolution_clock::now();
	double dt = std::chrono::duration<double>(t1 - t0).count();
	t0 = t1;

	// initialize window
	for (int i = 0, l = _WIN_W * _WIN_H; i < l; i++) _WINIMG[i] = 0;

	// axis and grid
	{
		vec2 LB = fromInt(vec2(0, 0)), RT = fromInt(vec2(_WIN_W, _WIN_H));
		COLORREF GridCol = clamp(0x10, (int)sqrt(10.0*Unit), 0x20); GridCol = GridCol | (GridCol << 8) | (GridCol) << 16;  // adaptive grid color
		for (int y = (int)round(LB.y), y1 = (int)round(RT.y); y <= y1; y++) drawLine(fromFloat(vec2(LB.x, y)), fromFloat(vec2(RT.x, y)), GridCol);  // horizontal gridlines
		for (int x = (int)round(LB.x), x1 = (int)round(RT.x); x <= x1; x++) drawLine(fromFloat(vec2(x, LB.y)), fromFloat(vec2(x, RT.y)), GridCol);  // vertical gridlines
		vec2 O = fromFloat(vec2(0.));
		drawLine(vec2(0, O.y), vec2(_WIN_W, O.y), 0x404060);  // x-axis
		drawLine(vec2(O.x, 0), vec2(O.x, _WIN_H), 0x404060);  // y-axis
	}

	// visualize SPH grid
	if (0) {
		for (int i = 0; i < SPH::Grid_N.x; i++) {
			for (int j = 0; j < SPH::Grid_N.y; j++) {
				drawBoxF(SPH::Grid_LB + vec2(i, j) * SPH::Grid_dp, SPH::Grid_LB + vec2(i + 1, j + 1)*SPH::Grid_dp, 0x202040);
				if (0) {
					SPH::gridcell* gi = SPH::getGrid(ivec2(i, j) - ivec2(0, 0));
					if (gi->N > 0) {
						COLORREF col = gi->N * 0x604000;
						//col = gi->Ps[0] * 12345679 + 32542778;
						fillBoxF(SPH::Grid_LB + vec2(i, j) * SPH::Grid_dp, SPH::Grid_LB + vec2(i + 1, j + 1)*SPH::Grid_dp, col);
					}
				}
			}
		}
	}

	// visualize density field
	if (1) {
		double *density = new double[_WIN_W*_WIN_H];
		for (int i = 0; i < _WIN_W*_WIN_H; i++) density[i] = 0.;
		for (int i = 0; i < SPH::N; i++) {
			vec2 p = SPH::Particles[i].p, pi = fromFloat(p);
			double r = 2.0 * SPH::h * Unit;
			int i0 = max(0, (int)floor(pi.x - r - 1)), i1 = min(_WIN_W - 1, (int)ceil(pi.x + r + 1));
			int j0 = max(0, (int)floor(pi.y - r - 1)), j1 = min(_WIN_H - 1, (int)ceil(pi.y + r + 1));
			for (int j = j0; j <= j1; j++) for (int i = i0; i <= i1; i++) {
				density[j*_WIN_W + i] += SPH::m * SPH::W(length(fromInt(vec2(i, j)) - p));
			}
		}
		for (int j = 0; j < _WIN_H; j++) for (int i = 0; i < _WIN_W; i++) {
			double d = density[j*_WIN_W + i];
			if (d != 0.) {
				if (1) {
					if (d > 0.5) _WINIMG[j*_WIN_W + i] = 0xffffff;
				}
				else {
					vec3 col = tanh(2.*d) * ColorFunctions<vec3, double>::Rainbow(clamp(0.5*pow(d, 6.), 0., 1.));
					_WINIMG[j*_WIN_W + i] = COLORREF(col.z * 255) | (COLORREF(col.y * 255) << 8) | (COLORREF(col.x * 255) << 16);
				}
			}
		}
		delete density;
	}

	// draw SPH particles
	if (1) {
		for (int i = 0; i < SPH::N; i++) {
			//vec3 col = ColorFunctions::ThermometerColors(clamp(SPH::Particles[i].density, 0., 1.));
			vec3 col = ColorFunctions<vec3, double>::TemperatureMap(tanh(0.1*length(SPH::Particles[i].v)));
			drawDotF(SPH::Particles[i].p, 0.5*SPH::h*Unit,
				COLORREF(col.z * 255) | (COLORREF(col.y * 255) << 8) | (COLORREF(col.x * 255) << 16));
		}
	}

	// momentum and energy calculation
	vec2 P = vec2(0.);  // total momentum
	double Eg = 0., Ek = 0.;  // gravitational potential and kinetic
	for (int i = 0; i < SPH::N; i++) {
		double m = SPH::m;
		Eg += -m * dot(SPH::Particles[i].p, SPH::g);
		Ek += 0.5 * m * SPH::Particles[i].v.sqr();
		P += m * SPH::Particles[i].v;
		// should also add potential energy due to the boundary
	}


	vec2 cursor = fromInt(Cursor);
	sprintf(text, "%d particles  (%.2f,%.2f)  t=%.3lf   E=Eg+Ek=%.3lg+%.3lg=%.3lg", SPH::N, cursor.x, cursor.y, SPH::t, Eg, Ek, Eg + Ek);
	SetWindowTextA(_HWND, text);

	// if this increases, then the simulation is unstable
	//printf("(%lg,%lg),", SPH::t, Eg + Ek);
}




// ============================================== User ==============================================


#include "numerical/random.h"

// preset scenes
class presetScenes {

public:  // boundary functions

	struct boundary_functions {

		static vec2 box_21(vec2 p, vec2 v) {
			return 1000.*vec2(
				p.x<0. ? -p.x + 1. : p.x>2. ? -(p.x - 2.) - 1. : 0.,
				p.y<0. ? -p.y + 1. : p.y>1. ? -(p.y - 1.) - 1. : 0.
			);
		}
		static vec2 circular(vec2 p, vec2 v) {
			p -= vec2(1.0, 0.5);
			return 10000.*max(length(p) - 1., 0.)*normalize(-p);
		}
		static vec2 box_block(vec2 p, vec2 v) {
			auto E = [](double x, double y)->double {
				double E = max(max(
					max(abs(x - 1.5) - 1.5, abs(y - 1.) - 1.),  // room
					-max(abs(x - 2.2) - 0.2, y - 0.2)  // block
				), 0.0);
				return E * E;
			};
			const double eps = 1e-5;
			return -1e5 * vec2(E(p.x + eps, p.y) - E(p.x - eps, p.y), E(p.x, p.y + eps) - E(p.x, p.y - eps)) / (2.0*eps);
		}
		static vec2 box_block_rotating(vec2 p, vec2 v) {
			auto E = [](double x, double y)->double {
				double t = SPH::t;
				vec2 p = mat2(cos(t), -sin(t), sin(t), cos(t)) * (vec2(x, y) - vec2(1, 1)) + vec2(1, 1);
				x = p.x, y = p.y;
				double E = max(max(
					max(abs(x - 1.5) - 1.5, abs(y - 1.) - 1.),  // room
					-max(abs(x - 2.2) - 0.2, y - 0.2)  // block
				), 0.0);
				return E * E;
			};
			const double eps = 1e-5;
			vec2 Fn = -1e5 * vec2(E(p.x + eps, p.y) - E(p.x - eps, p.y), E(p.x, p.y + eps) - E(p.x, p.y - eps)) / (2.0*eps);
			vec2 n = normalize(Fn);
			vec2 Fd = -0.1*length(Fn) * dot(v, n)*n;  // probably not a correct formula because its unit
			vec2 Ff = -0.01*length(Fn) * normalize(v - dot(v, n)*n);
			if (isnan(Fd.x)) Fd = vec2(0.0);
			if (isnan(Ff.x)) Ff = vec2(0.0);
			return Fn + Fd + Ff;
		}
		static vec2 box_block_circ(vec2 p, vec2 v) {
			auto E = [](double x, double y)->double {
				return max(max(
					max(abs(x - 1.) - 1., abs(y - 0.8) - 0.8),  // room
					0.1 - length(vec2(x, y) - vec2(1.5, 0.15))  // block
				), 0.);
			};
			const double eps = 0.01;
			return -10000. * vec2(E(p.x + eps, p.y) - E(p.x - eps, p.y), E(p.x, p.y + eps) - E(p.x, p.y - eps)) / (2.*eps);
		}

		// non-boundary force fields
		static vec2 np_centric(vec2 p, vec2 v) {
			return -50.*p;
		}
		static vec2 np_circular(vec2 p, vec2 v) {
			return 4.5*p.rot() - 50.*p*(p.sqr() - 1.);
		}
		static vec2 np_rouded_quad(vec2 p, vec2 v) {
			vec2 q = vec2(p.x - p.y, p.x + p.y);
			return 4.5*p.rot() - 50.*p*(length(q*q) - 1.);
		}

	};

public:  // preset scenes

	static void Scene_random(int N, double h, vec2 v0 = vec2(0.)) {
		SPH::h = h;
		SPH::N = N;
		SPH::m = SPH::density_0 / (4.*SPH::W(h) + 4.*SPH::W(1.414214*h) + SPH::W(0.));
		SPH::Particles = new SPH::particle[N];
		uint32_t seed = 0;
		for (int i = 0; i < N; i++) {
			vec2 p = vec2(2.* rand01(seed), rand01(seed));
			int j; for (j = 0; j < i; j++) {
				if (length(SPH::Particles[j].p - p) < 2.*h) {
					i--; break;
				}
			}
			if (j == i)
				SPH::Particles[i] = SPH::particle{ p, v0 };
		}
		SPH::Boundary = boundary_functions::box_21;
		SPH::Grid_LB = vec2(0, 0);
		SPH::Grid_dp = vec2(2.*h);
		SPH::Grid_N = ivec2(ceil((vec2(2, 1) - SPH::Grid_LB) / SPH::Grid_dp));
	}

	static void Scene_left(double dist) {
		using namespace SPH;
		h = dist;
		m = density_0 / (4.*W(h) + 4.*W(1.414214*h) + W(0.));
		int xN = (int)floor(0.4999 / dist);
		int yN = (int)floor(0.9999 / dist);
		Particles = new particle[xN*yN];
		N = 0;
		for (int i = 0; i < xN; i++) {
			for (int j = 0; j < yN; j++) {
				Particles[N++] = particle{
					vec2((i + 0.5)*dist, (j + 0.5)*dist),
					0.01*cossin(fmod(12345.67*sin(32.45*i + 98.01*j + 13.25), 2.*PI))
				};
			}
		}
		viscosity = 1e-4;
		Boundary = boundary_functions::box_21;
		Grid_LB = vec2(0, 0);
		Grid_dp = vec2(2.*h);
		Grid_N = ivec2(ceil((vec2(2, 1) - Grid_LB) / Grid_dp));
	}

	static void Scene_right(double dist) {
		using namespace SPH;
		const double hc = 0.5*sqrt(3);
		h = dist;
		m = density_0 / (4.*W(h) + 4.*W(1.414214*h) + W(0.));
		m = density_0 / (6.*W(h) + 6.*W(1.732051*h) + W(0.));
		int xN = (int)floor(0.4999 / dist);
		int yN = (int)floor(0.9999 / (hc*dist));
		Particles = new particle[xN*yN];
		N = 0;
		for (int j = 0; j < yN; j++) {
			vec2 p0 = vec2(j & 1 ? 1.2 - 0.5*dist : 1.2, 0);
			for (int i = j & 1; i < xN; i++) {
				Particles[N++] = particle{
					p0 + vec2((i + 0.5)*dist, (j + 0.5)*dist*hc),
					0.01*cossin(fmod(12345.67*sin(32.45*i + 98.01*j + 13.25), 2.*PI))
				};
			}
		}
		viscosity = 1e-4;
		Boundary = boundary_functions::box_21;
		Grid_LB = vec2(0, 0);
		Grid_dp = vec2(2.*h);
		Grid_N = ivec2(ceil((vec2(2, 1) - Grid_LB) / Grid_dp));
	}

	static void Scene_fall(double dist) {
		using namespace SPH;
		h = dist;
		m = density_0 / (4.*W(h) + 4.*W(1.414214*h) + W(0.));
		int xN = (int)floor(2.0 / dist);
		int yN = (int)floor(1.0 / dist);
		std::vector<vec2> particles;
		for (int i = 0; i < xN; i++) {
			for (int j = 0; j < yN; j++) {
				vec2 p = vec2(i, j)*dist;
				double dist = length(p - vec2(1, 0.8)) - 0.2;
				dist = min(dist, p.y - 0.2);
				if (dist < 0.) particles.push_back(p);
			}
		}
		N = particles.size();
		Particles = new particle[N];
		for (int i = 0; i < N; i++) {
			vec2 p = particles[i];
			Particles[i] = particle{ p, vec2(0.0) };
		}
		viscosity = 1e-4;
		Boundary = boundary_functions::box_21;
		//Boundary = box_block;
		Grid_LB = vec2(0, 0);
		Grid_dp = vec2(2.*h);
		Grid_N = ivec2(ceil((vec2(2, 1) - Grid_LB) / Grid_dp));
	}

};


#include <thread>

void WindowCreate(int _W, int _H) {
	//presetScenes::Scene_random(500, 0.02, vec2(1, 0));
	//presetScenes::Scene_left(0.04);
	//presetScenes::Scene_left(0.02);
	//presetScenes::Scene_left(0.01);
	//presetScenes::Scene_right(0.02);
	presetScenes::Scene_fall(0.02);

	SPH::max_step = SPH::h >= 0.02 ? 0.001 : 0.0005;
	//SPH::max_step *= 0.1;

	SPH::Boundary = presetScenes::boundary_functions::box_block;
	//SPH::viscosity = 0.01;

	vec2 Center = SPH::Grid_LB + 0.5*vec2(SPH::Grid_N)*SPH::Grid_dp;
	Unit = 0.9 * min(_WIN_W / (SPH::Grid_N.x*SPH::Grid_dp.x), _WIN_H / (SPH::Grid_N.y*SPH::Grid_dp.y));
	Origin = (fromInt(0.5*vec2(_WIN_W, _WIN_H)) - Center) * Unit;

	// simulation thread
	new std::thread([]() {
		auto t0 = std::chrono::high_resolution_clock::now();
		const double dt = 0.05;
		for (;;) {
			double time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
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
	if (_W*_H == 0 || _oldW * _oldH == 0) return;
	double pw = _oldW, ph = _oldH, w = _W, h = _H;
	double s = sqrt((w * h) / (pw * ph));
	Unit *= s;
	Origin.x *= w / pw, Origin.y *= h / ph;
	Render_Needed = true;
}
void WindowClose() {}

void MouseMove(int _X, int _Y) {
	vec2 P0 = Cursor, P = vec2(_X, _Y);
	Cursor = P;
	vec2 p0 = fromInt(P0), p = fromInt(P), d = p - p0;

	// click and drag
	if (mouse_down) {
		Origin = Origin + d * Unit;
	}

	Render_Needed = true;
}

void MouseWheel(int _DELTA) {
	Render_Needed = true;
	double s = exp((Alt ? 0.0001 : 0.001)*_DELTA);
	double D = length(vec2(_WIN_W, _WIN_H)), Max = 10.0*D, Min = 0.01*D;
	if (Unit * s > Max) s = Max / Unit;
	else if (Unit * s < Min) s = Min / Unit;
	Origin = mix(Cursor, Origin, s);
	Unit *= s;
}

void MouseDownL(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	mouse_down = true;
	vec2 p = fromInt(Cursor);
	Render_Needed = true;
}

void MouseUpL(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	mouse_down = false;
	vec2 p = fromInt(Cursor);
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

