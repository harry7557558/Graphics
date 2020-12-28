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
void drawBoxF(double x0, double x1, double y0, double y1, COLORREF col);
void drawDot(vec2 c, double r, COLORREF col);
#define drawLineF(p,q,col) drawLine(fromFloat(p),fromFloat(q),col)
#define drawDotF(p,r,col) drawDot(fromFloat(p),r,col)

#pragma endregion  // Window variables


namespace SPH {

	// scene
	int N = 0;  // number of SPH particles
	struct particle {
		vec2 p;  // position
		vec2 v;  // velocity
	} *Particles = 0;  // array of particles
	double t = 0.;  // time

	// particle properties
	double m;  // mass of each particle
	double h;  // smoothing radius
	double W(double d) {  // smoothing kernel, a function of distance
		double x = d / h;
		double f = x < 1. ? 1. - x * x*(1.5 + 0.75*x) : x < 2. ? 0.25*(2. - x)*(2. - x)*(2. - x) : 0.;
		return f / (PI*h*h) * (10. / 7.);
	}
	vec2 W_grad(vec2 d) {  // gradient of the smoothing kernel
		double x = length(d) / h;
		if (!(x > 0.)) return vec2(0.);
		double f = x < 1. ? x * (2.25*x - 3.) : x < 2. ? -0.75*(2. - x)*(2. - x) : 0.;
		return normalize(d) * (f / (PI*h*h*h) * (10. / 7.));
	}

	// forces
	double rho_0 = 1.0;  // rest density
	double pressure(double rho) {  // calculate pressure from fluid density
		return 10.*(pow(rho / rho_0, 7.) - 1.);
	}
	double *Rhos = 0;  // density at each particle
	double *Pressures = 0;  // pressure at each particle
	double viscosity = 0.;  // fluid viscosity, unit: m²/s; a=viscosity*∇²v
	vec2 g = vec2(0, -9.8);  // acceleration due to gravity
	vec2(*Boundary)(vec2) = 0;  // penalty acceleration field that defines the rigid boundary
	vec2 *Accelerations = 0;  // computed acceleration of each particle

	// neighbor finding
	vec2 Grid_LB;  // lower bottom of the grid starting point
	vec2 Grid_dp;  // size of each grid, should be vec2(2h,2h)
	ivec2 Grid_N;  // dimension of the overall grid, top right Grid_LB+Grid_dp*Grid_N
	void calcGrid();  // update simulation grid
	int *Neighbor_N = 0;  // number of neighbors of each particle

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
	LB -= Grid_dp, TR += Grid_dp;
	// update Grid_LB and Grid_N
	Grid_LB = LB;
	Grid_N = (ivec2)ceil((TR - LB) / Grid_dp);
}

void updateScene(double dt) {
	// split large steps
	double max_step = 0.000001;
	if (dt > max_step) {
		int N = (int)(dt / max_step + 1);
		for (int i = 0; i < N; i++) updateScene(dt / N);
		return;
	}

	using namespace SPH;

	// find neighbors of each particle
	calcGrid();
	if (!Neighbor_N) Neighbor_N = new int[N];
	std::vector<int> *Neighbors = new std::vector<int>[N];
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) if (j != i) {
			if (length(Particles[j].p - Particles[i].p) < 2.*h)
				Neighbors[i].push_back(j);
		}
		Neighbor_N[i] = Neighbors[i].size();
	}

	// calculate the density and pressure at each particle
	if (!Rhos) Rhos = new double[N];
	if (!Pressures) Pressures = new double[N];
	for (int i = 0; i < N; i++) {
		double rho = 0.;
		for (int _ = 0; _ < Neighbor_N[i]; _++) {
			int j = Neighbors[i][_];
			rho += m * W(length(Particles[j].p - Particles[i].p));
		}
		Rhos[i] = rho;
		Pressures[i] = pressure(rho);
	}

	// compute the acceleration of each particle
	if (!Accelerations) Accelerations = new vec2[N];
	for (int i = 0; i < N; i++) {
		Accelerations[i] = g + Boundary(Particles[i].p);
		Accelerations[i] -= 0.5*Particles[i].v;
	}
	for (int i = 0; i < N; i++) {
		if (Rhos[i] == 0.) continue;
		// estimate the gradient of pressure
		vec2 grad_P(0.);
		for (int u = 0; u < Neighbor_N[i]; u++) {
			int j = Neighbors[i][u];
			if (Rhos[j] == 0.) throw(0);
			//grad_P += m / Rhos[j] * Pressures[j] * W_grad(Particles[j].p - Particles[i].p);
			grad_P += W_grad(Particles[j].p - Particles[i].p);
		}
		// add acceleration due to pressure difference
		//Accelerations[i] -= grad_P / Rhos[i];
		Accelerations[i] += grad_P;
	}

	// update position and velocity of each particle
	for (int i = 0; i < N; i++) {
		Particles[i].v += Accelerations[i] * dt;
		Particles[i].p += Particles[i].v * dt;
	}

	SPH::t += dt;
	delete[] Neighbors;
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
void drawBoxF(double x0, double x1, double y0, double y1, COLORREF col) {
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
		vec2 O = fromFloat(vec2(0.));
		drawLine(vec2(0, O.y), vec2(_WIN_W, O.y), 0x404060);  // x-axis
		drawLine(vec2(O.x, 0), vec2(O.x, _WIN_H), 0x404060);  // y-axis
	}

	// visualize SPH grid
	if (0) {
		using namespace SPH;
		for (int i = 0; i < Grid_N.x; i++) {
			for (int j = 0; j < Grid_N.y; j++) {
				drawBoxF(Grid_LB.x + i * Grid_dp.x, Grid_LB.x + (i + 1)*Grid_dp.x, Grid_LB.y + j * Grid_dp.y, Grid_LB.y + (j + 1)*Grid_dp.y, 0x202040);
			}
		}
	}

	// draw SPH particles
	{
		for (int i = 0; i < SPH::N; i++) {
			drawDotF(SPH::Particles[i].p, 0.5*SPH::h*Unit, 0x0080FF);
		}
	}


	vec2 cursor = fromInt(Cursor);
	sprintf(text, "(%.2f,%.2f)  t=%.3lf", cursor.x, cursor.y, SPH::t);
	SetWindowTextA(_HWND, text);
}




// ============================================== User ==============================================


// preset scenes
class presetScenes {

private:  // boundary functions

	static vec2 box_21(vec2 p) {
		return 10000.*vec2(
			p.x<0. ? -p.x : p.x>2. ? -(p.x - 2.) : 0.,
			p.y<0. ? -p.y : p.y>1. ? -(p.y - 1.) : 0.
		);
	}

public:  // preset scenes

	static void Scene_0(double dist) {
		using namespace SPH;
		h = dist;
		m = rho_0 / (4.*W(h) + 4.*W(1.414214*h));
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
		Boundary = box_21;
		Grid_LB = vec2(0, 0);
		Grid_dp = vec2(2.*h);
		Grid_N = ivec2(ceil((vec2(2, 1) - Grid_LB) / Grid_dp));
	}

};


#include <thread>

void WindowCreate(int _W, int _H) {
	presetScenes::Scene_0(0.1);

	vec2 Center = SPH::Grid_LB + 0.5*vec2(SPH::Grid_N)*SPH::Grid_dp;
	Unit = min(_WIN_W / (SPH::Grid_N.x*SPH::Grid_dp.x), _WIN_H / (SPH::Grid_N.y*SPH::Grid_dp.y));
	Origin = (fromInt(0.5*vec2(_WIN_W, _WIN_H)) - Center) * Unit;

	// simulation thread
	new std::thread([]() {
		auto t0 = std::chrono::high_resolution_clock::now();
		const double frame_delay = 0.01;
		for (;;) {
			double time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
			double time_next = ceil(time / frame_delay)*frame_delay;
			updateScene(time_next - SPH::t);
			double t_remain = time_next - std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
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

