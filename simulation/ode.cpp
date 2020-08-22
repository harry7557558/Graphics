// Test numerical integration of ODEs: x"(t) = a
// In this experiment, integration variables are 3d vectors

// Win32 GUI
// 2kb simulation code and 30kb debugging code?! oh no...


// To-do:
// * Adaptive step size



// debug
#define _USE_CONSOLE 0

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <functional>
#pragma warning(disable: 4244 4305 4996)

// ========================================= Win32 Standard =========================================

#pragma region Windows

#ifndef UNICODE
#define UNICODE
#endif

#include <Windows.h>
#include <windowsx.h>
#include <tchar.h>

// debug
wchar_t _DEBUG_OUTPUT_BUF[0x1000];
#define dbgprint(format, ...) { if (_USE_CONSOLE) {printf(format, ##__VA_ARGS__);} else {swprintf(_DEBUG_OUTPUT_BUF, 0x1000, _T(format), ##__VA_ARGS__); OutputDebugStringW(_DEBUG_OUTPUT_BUF);} }


#pragma region Window Macros / Forward Declarations

#define WIN_NAME "ode"
#define WinW_Padding 100
#define WinH_Padding 100
#define WinW_Default 640
#define WinH_Default 400
#define WinW_Min 400
#define WinH_Min 300
#define WinW_Max 3840
#define WinH_Max 2160

void Init();
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
double _DEPTHBUF[WinW_Max][WinH_Max];

#pragma endregion


// Win32 Entry

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
#define _RDBK { HDC hdc = GetDC(hWnd), HImgMem = CreateCompatibleDC(hdc); HBITMAP hbmOld = (HBITMAP)SelectObject(HImgMem, _HIMG); render(); BitBlt(hdc, 0, 0, _WIN_W, _WIN_H, HImgMem, 0, 0, SRCCOPY); SelectObject(HImgMem, hbmOld), DeleteDC(HImgMem), DeleteDC(hdc); break; }
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
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow) {
	if (_USE_CONSOLE) if (AttachConsole(ATTACH_PARENT_PROCESS) || AllocConsole()) freopen("CONIN$", "r", stdin), freopen("CONOUT$", "w", stdout), freopen("CONOUT$", "w", stderr);
	WNDCLASSEX wc; wc.cbSize = sizeof(WNDCLASSEX), wc.style = 0, wc.lpfnWndProc = WndProc, wc.cbClsExtra = wc.cbWndExtra = 0, wc.hInstance = hInstance; wc.hIcon = wc.hIconSm = 0, wc.hCursor = LoadCursor(NULL, IDC_ARROW), wc.hbrBackground = CreateSolidBrush(RGB(0, 0, 0)), wc.lpszMenuName = NULL, wc.lpszClassName = _T(WIN_NAME); if (!RegisterClassEx(&wc)) return -1;
	_HWND = CreateWindow(_T(WIN_NAME), _T(WIN_NAME), WS_OVERLAPPEDWINDOW, WinW_Padding, WinH_Padding, WinW_Default, WinH_Default, NULL, NULL, hInstance, NULL); ShowWindow(_HWND, nCmdShow); UpdateWindow(_HWND);
	MSG message; while (GetMessage(&message, 0, 0, 0)) { TranslateMessage(&message); DispatchMessage(&message); } return (int)message.wParam;
}

#pragma endregion


// ================================== Vector Classes/Functions ==================================

#pragma region Vector & Matrix

#include "numerical/geometry.h"

const vec3 vec0(0, 0, 0), veci(1, 0, 0), vecj(0, 1, 0), veck(0, 0, 1);
#define SCRCTR vec2(0.5*_WIN_W,0.5*_WIN_H)

// 4x4 matrix (simple)
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

#pragma endregion


// ======================================== Data / Parameters ========================================

// viewport
// Ctrl/Shift + Drag/Wheel to adjust these variables
vec3 Center(0.0, 0.0, 0.0);  // view center in world coordinate
double rz = -0.8, rx = 0.3, ry = 0.0, dist = 120.0, Unit = 100.0;  // yaw, pitch, row, camera distance, scale to screen

#pragma region General Global Variables and Functions

// window parameters
char text[64];	// window title
Affine Tr;  // matrix
vec3 CamP, ScrO, ScrA, ScrB;  // camera and screen
auto scrDir = [](vec2 pixel) { return normalize(ScrO + (pixel.x / _WIN_W)*ScrA + (pixel.y / _WIN_H)*ScrB - CamP); };

// user parameters
vec2 Cursor = vec2(0, 0), clickCursor;  // current cursor and cursor position when mouse down
bool mouse_down = false;
bool Ctrl = false, Shift = false, Alt = false;  // these variables are shared by both windows

// projection
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
	Affine S{ veci, vecj, veck, vec3(0), vec3(0), 1.0 / Unit };  // scale
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

// rasterization forward declaration
void drawLine(vec2 p, vec2 q, COLORREF col);
void drawCross(vec2 p, double r, COLORREF col);
void drawCircle(vec2 c, double r, COLORREF col);
void fillCircle(vec2 c, double r, COLORREF col);
void drawLine_F(vec3 A, vec3 B, COLORREF col);
void drawCross3D(vec3 P, double r, COLORREF col, bool relative);

#pragma endregion 3D graphics


#pragma region Simulation Test Cases

class Test_Case {
public:
	double dt, t1;
	vec3 P0, V0;
	std::function<vec3(vec3, vec3, double)> Acceleration;
	std::function<void(double)> additional_render;
	template<typename Fun>
	Test_Case(Fun acc, double t1, double dt, vec3 p0, vec3 v0) {
		this->dt = dt, this->t1 = t1; P0 = p0, V0 = v0; Acceleration = acc;
		additional_render = [](double) {};
	}
	template<typename Fun, typename Rd>
	Test_Case(Fun acc, double t1, double dt, vec3 p0, vec3 v0, Rd render = [](double) {}) {
		this->dt = dt, this->t1 = t1; P0 = p0, V0 = v0; Acceleration = acc;
		additional_render = render;
	}
};

// Gravitational Acceleration
const vec3 g = vec3(0, 0, -9.81);

// Acceleration due to gravity
Test_Case Projectile([](vec3 p, vec3 v, double t)->vec3 {
	return g;
}, 2.0, 0.1, vec3(0, 0, 1), vec3(1.5, 1.5, 2));

// Acceleration due to gravity + v³ air resistance
Test_Case Projectile_R([](vec3 p, vec3 v, double t)->vec3 {
	double r = 0.1*dot(v, v);
	return g - r * v;
}, 2.0, 0.06, vec3(0, 0, 1), vec3(2, 2, 3));

// *Drops onto an elastic surface and bounces up
Test_Case Projectile_RB([](vec3 p, vec3 v, double t)->vec3 {
	vec3 r = -0.1*dot(v, v)*v;
	vec3 b = min(p.z, 0.)*vec3(0, 0, -50);
	return g + r + b;
}, 6.0, 0.02, vec3(-2.5, 0, 1), vec3(3, 0, 3));

// **Drops into heavy liquid and bounces up
Test_Case Projectile_RW([](vec3 p, vec3 v, double t)->vec3 {
	vec3 r = -(p.z > 0. ? 0.05 : 0.5)*dot(v, v)*v;
	vec3 b = (p.z < 0. ? 1. : 0.)*vec3(0, 0, 50);
	return g + r + b;
}, 6.0, 0.01, vec3(-3, 0, 1), vec3(3, 0, 3));

// On a one-meter-long non-deformable rod
// An accurate solution should not deviate the unit sphere
Test_Case Pendulum([](vec3 p, vec3 v, double t)->vec3 {
	vec3 u = normalize(cross(p, cross(p, veck))) * (length(g) * length(p.xy()) / length(p));
	vec3 w = -p * dot(v, v);
	return u + w;
}, 6.0, 0.02, vec3(0, 1, 0), vec3(1, 0, 0));

// `On a one-meter-long spring with air resistance
Test_Case Pendulum_S([](vec3 p, vec3 v, double t)->vec3 {
	vec3 d = p - vec3(0, 0, 2);
	vec3 N = -10.0*(length(d) - 1)*normalize(d);
	vec3 r = -0.01*dot(v, v)*v;
	return g + N + r;
}, 6.0, 0.05, vec3(0, 1, 0), vec3(3, 0, 3));

// A "sun" in the center
// The solution is an ellipse
Test_Case NBody_1([](vec3 p, vec3 v, double t)->vec3 {
	double m = length(p);
	return -20.0*p / (m*m*m);
}, 3.0, 0.02, vec3(2, 2, 0), vec3(1, -1, -0.02));

// `Two "suns" with equal mass
Test_Case NBody_2([](vec3 p, vec3 v, double t)->vec3 {
	vec3 q = p - vec3(0, 1, 0);
	double m = length(q);
	vec3 a = q / (m*m*m);
	q = p + vec3(0, 1, 0);
	m = length(q);
	a += q / (m*m*m);
	return -10.0*a;
}, 8.0, 0.01, vec3(2, 2, 0), vec3(1, -0.5, -0.2), [](double t) {
	fillCircle((Tr*vec3(0, 1, 0)).xy(), 4, 0x00A0FF);
	fillCircle((Tr*vec3(0, -1, 0)).xy(), 4, 0x00A0FF);
});

// `One sun and one mobilized planet
Test_Case NBody_m([](vec3 p, vec3 v, double t)->vec3 {
	double m = length(p);
	vec3 F = -9.*p / (m*m*m);
	p -= vec3(cos(3.*t), sin(3.*t), 0);
	m = length(p);
	return F - p / (m*m*m);
}, 4.0, 0.02, vec3(2, 2, 0), vec3(1, -.5, -.2), [](double t) {
	fillCircle((Tr*vec3(0.0)).xy(), 6, 0xC08000);
	fillCircle((Tr*vec3(cos(3.*t), sin(3.*t), 0)).xy(), 4, 0x00FF00);
});

// `Artificial equation #1
Test_Case TimeTest_1([](vec3 p, vec3 v, double t)->vec3 {
	vec3 w = vec3(0, 0, sin(t) + cos(t) + 1);
	return cross(v, w) - w.sqr() * p - 2.*p + vec3(0, 0, -1) - 0.01*v.sqr()*v;
}, 6.0, 0.05, vec3(1, 1, 1), vec3(0, 0, 0));

// **`Artificial equation #2
Test_Case TimeTest_2([](vec3 p, vec3 v, double t)->vec3 {
	double s = floor(2.*t*t + 2.);
	double a = fmod(1000 * sin(100 * floor(t)), 1.0);
	double b = fmod(2000 * cos(200 * floor(t)), 1.0);
	double c = fmod(3000 * sin(300 * floor(t)), 1.0);
	vec3 w = 2.0*vec3(c, b, a) - vec3(1.0);
	return cross(v, w) - (length(w) + 2.) * p + 0.2*w;
}, 6.0, 0.05, vec3(1, 1, 1), vec3(-1, 0, 0));

// `Artificial equation #3 (continuous)
Test_Case TimeTest_3([](vec3 p, vec3 v, double t)->vec3 {
	vec3 a = vec3(cos(p.x + t), sin(v.y - t), sin(p.z + t));
	vec3 b = vec3(cos(v.x + t), exp(p.y), cos(v.z));
	vec3 c = vec3(sin(t), cos(t), sin(t + .25*PI));
	return 4.*(a + b + c - 5.*p);
}, 6.0, 0.05, vec3(0), vec3(0));

// `Artificial equation #4
Test_Case TimeTest_4([](vec3 p, vec3 v, double t)->vec3 {
	p.z -= 2.0;
	vec3 N = -(2.0*exp(sin(t + 0.5)) + 4.0)*(length(p) - cos(t)*cos(t))*p;
	vec3 r = 0.05*(sin(t) + 1.0)*dot(v, v)*v;
	return N - r + vec3(0, 0, -4);
}, 6.0, 0.05, vec3(0, 1, 0), vec3(4, -1, 4));

// `Artificial equation #5
Test_Case TimeTest_5([](vec3 p, vec3 v, double t)->vec3 {
	return vec3(-p.xy()*(exp(cos(2.0*t)) + 1.0) - 0.4*p.z*v.xy(), -2.0*length(v)*p.z + 0.5* exp(sin(2.0*t)));
}, 8.0, 0.1, vec3(1, 1, 1), vec3(-1, 1, 0));


#pragma endregion Projectile, Projectile_R, Projectile_RB, Projectile_RW, Pendulum, Pendulum_S, NBody_1, NBody_2, NBody_m, TimeTest_[1-5]

Test_Case T = NBody_1;



// ============================================ Rendering ============================================

#pragma region Rasterization functions

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
void drawCross(vec2 p, double r, COLORREF col = 0xFFFFFF) {
	drawLine(p - vec2(r, 0), p + vec2(r, 0), col);
	drawLine(p - vec2(0, r), p + vec2(0, r), col);
};
void drawCircle(vec2 c, double r, COLORREF col) {
	int s = int(r / sqrt(2) + 0.5);
	int cx = (int)c.x, cy = (int)c.y;
	for (int i = 0, im = min(s, max(_WIN_W - cx, cx)) + 1; i < im; i++) {
		int u = sqrt(r*r - i * i) + 0.5;
		setColor(cx + i, cy + u, col); setColor(cx + i, cy - u, col); setColor(cx - i, cy + u, col); setColor(cx - i, cy - u, col);
		setColor(cx + u, cy + i, col); setColor(cx + u, cy - i, col); setColor(cx - u, cy + i, col); setColor(cx - u, cy - i, col);
	}
};
void fillCircle(vec2 c, double r, COLORREF col) {
	int x0 = max(0, int(c.x - r)), x1 = min(_WIN_W - 1, int(c.x + r));
	int y0 = max(0, int(c.y - r)), y1 = min(_WIN_H - 1, int(c.y + r));
	int cx = (int)c.x, cy = (int)c.y, r2 = int(r*r);
	for (int x = x0, dx = x - cx; x <= x1; x++, dx++) {
		for (int y = y0, dy = y - cy; y <= y1; y++, dy++) {
			if (dx * dx + dy * dy < r2) Canvas(x, y) = col;
		}
	}
};

void drawLine_F(vec3 A, vec3 B, COLORREF col = 0xFFFFFF) {
	double u = dot(Tr.p, A) + Tr.s, v = dot(Tr.p, B) + Tr.s;
	if (u > 0 && v > 0) { drawLine((Tr*A).xy(), (Tr*B).xy(), col); return; }
	if (u < 0 && v < 0) return;
	if (u < v) std::swap(A, B), std::swap(u, v);
	double t = u / (u - v) - 1e-4;
	B = A + (B - A)*t;
	drawLine((Tr*A).xy(), (Tr*B).xy(), col);
};
void fillCircle_F(vec3 P, double r, COLORREF col) {
	fillCircle((Tr*P).xy(), r, col);
}
void drawCross3D(vec3 P, double r, COLORREF col = 0xFFFFFF, bool relative = true) {
	if (relative) r *= dot(Tr.p, P) + Tr.s;
	drawLine_F(P - vec3(r, 0, 0), P + vec3(r, 0, 0), col);
	drawLine_F(P - vec3(0, r, 0), P + vec3(0, r, 0), col);
	drawLine_F(P - vec3(0, 0, r), P + vec3(0, 0, r), col);
};

#pragma endregion


#pragma region Time
#include <chrono>
typedef std::chrono::high_resolution_clock NTime;
typedef std::chrono::duration<double> fsec;
auto start_time = NTime::now();
double iTime;
#pragma endregion hold alt to "freeze"


#include <vector>
#include "numerical/interpolation.h"
struct keyFrame {
	double t;  // time
	vec3 p;  // position
	vec3 v;  // derivative of position
};
typedef std::vector<keyFrame> Path;
vec3 eval(const Path &P, double t) {  // P should have enough keyframes
	int N = P.size();
	auto s = std::lower_bound(P.begin(), P.end(), t, [](keyFrame f, double t) {return f.t < t; });
	if (s == P.begin()) s++;
	else if (s == P.end()) s--;
	//return lerp((s - 1)->p, s->p, (t - (s - 1)->t) / (s->t - (s - 1)->t));  // linear interpolation
	return intp_d((s - 1)->p, s->p, (s - 1)->v, s->v, (t - (s - 1)->t) / (s->t - (s - 1)->t));  // derivative-based cubic interpolation
}
void plotPath(const Path &P, double t, COLORREF col, bool highlight = false) {
	for (int i = 1, N = P.size(); i < N; i++) {
		if (highlight) drawCross3D(P[i].p, 2, col);
		drawLine_F(P[i - 1].p, P[i].p, col);
	}
	fillCircle_F(eval(P, t), 6, col);
}



#pragma region Simulation

/*
// For a first order ODE with one variable x = x(t):
// x(t+h) = x(t) + x'(t) h + x"(t) h²/2 + x"'(t) h³/6 + x""(t) h⁴/24 + x""'(t) h⁵/120 + O(h⁶)
// x'(t) = v(x,t)
// x"(t) = dv/dx v + dv/dt
// x‴(t) = dv/dx x" + [ d²v/dx² x' + 2 d²v/dxdt ] x' + d²v/dt²
//       = d²v/dx² v² + 2 d²v/dxdt v + (dv/dx)² v + dv/dx dv/dt + d²v/dt²
// x⁗(t) = dv/dx x‴ + 3 d²v/dxdt x" + d³v/dx³ x'³ + 3 d³v/dx²dt x'² + 3 d³v/dxdt² x' + 3 d²v/dx² x'x" + d³v/dt³
// x⁵(t) = dv/dx x⁗ + 4 d²v/dxdt x‴ + 6 d³v/dxdt² x" + 3 d²v/dx² x"² + d⁴v/dx⁴ x'⁴ + 4 d⁴v/dx³dt x'³ + 6 d⁴v/dx²dt² x'² + 4 d⁴v/dxdt³ x' + 4 d²v/dx² x'x‴ + 6 d³v/dx³ x"x'² + 12 d³v/dx²dt x'x" + d⁴v/dt⁴

// Standard Euler method: x + v h; Error: x"(t) h²/2 + O(h³), O(h²)
// Standard midpoint method: x + v h + x"(t) h²/2 + (d²v/dx² v² + 2 d²v/dxdt v + d²v/dt²) h³/8 + O(h⁴); Error: O(h³)
// 4th order Runge-Kutta method:


// p(t+h) = p(t) + p'(t) h + p"(t) h²/2 + p"'(t) h³/6 + p""(t) h⁴/24 + O(h⁵)
// v(t+h) = p'(t) + p"(t) h + p"'(t) h²/2 + p""(t) h³/6 + O(h⁴)
// p(t) = p
// p'(t) = v
// p"(t) = a(p,v,t)
// p‴(t) = ∂a/∂p v + ∂a/∂v a + ∂a/∂t
// p⁗(t) = [ ∂²a/∂p² v v + ∂²a/∂v² a a + 2 ∂²a/∂p∂v v a ] + [ 2 ∂²a/∂p∂t v + 2 ∂²a/∂v∂t a + ∂a/∂v ∂a/∂p v + (∂a/∂v)² a ] + ∂a/∂v ∂a/∂t + ∂a/∂p a + ∂²a/∂t²

// t, h: scalar
// p, v, a, ∂a/∂t, ∂²a/∂t²: column vector
// ∂a/∂p, ∂a/∂v, ∂²a/∂p∂t, ∂²a/∂v∂t: 3x3 matrix
// ∂²a/∂p², ∂²a/∂v², ∂²a/∂p∂v: 3x3x3 cube matrix??
// Is (∂²a/∂v∂p a v) equal to (∂²a/∂p∂v v a) ?  - Yes.
// Is (∂²a/∂p∂v a v) equal to (∂²a/∂p∂v v a) ?  - Maybe not.
// hope my math isn't wrong... (I swear I don't know anything about tensor)
*/

Path RefPath;
void initReferencePath() {
	const double dt = 0.00001;
	const int dtN = 1000;
	vec3 p0 = T.P0, v0 = T.V0, p = p0, v = v0, a;
	int N = (int)(T.t1 / dt);
	for (int i = 0; i <= N; i++) {
		double t = i * dt;
		a = T.Acceleration(p, v, t);
		vec3 _p = p;
		p += v * dt + a * (.5*dt*dt);
		v += T.Acceleration(.5*(_p + p), v + a * (.5*dt), t + .5*dt) * dt;
		if (i % dtN == 0) RefPath.push_back(keyFrame{ t, p0, v0 * (dt*dtN) });
		p0 = p, v0 = v;
	}
	vec3 Min(INFINITY), Max = -Min;
	for (int i = 0, l = RefPath.size(); i < l; i++) {
		Min = pMin(Min, RefPath[i].p), Max = pMax(Max, RefPath[i].p);
	}
	Center = 0.5*(Min + Max);
	if (isnan(Center.sqr())) Center = vec3(0.0);
}

// orange; p: 1/6 h³ p³(t0); v: 1/2 h² p³(t0)
Path EulersMethodPath;
void EulersMethodInit() {
	vec3 p0, p = p0 = T.P0, v0, v = v0 = T.V0, a;
	double t_max = T.t1, dt = T.dt, t;
	for (t = 0.0; t < t_max; t += dt) {
		a = T.Acceleration(p, v, t);
#if 1
		p += v * dt;
#else
		p += v * dt + a * (.5*dt*dt);
#endif
		v += a * dt;
		EulersMethodPath.push_back(keyFrame{ t, p0, v0*dt });
		p0 = p, v0 = v;
	}
	EulersMethodPath.push_back(keyFrame{ t, p0, v0*dt });
}

// yellow; p: 1/6 h³ p³(t0); v: h³ p⁴(t0) missing some terms
Path MidpointMethodPath;
void MidpointMethodInit() {
	vec3 p0, p = p0 = T.P0, v0, v = v0 = T.V0, a, am;
	double t_max = T.t1, dt = 2.0*T.dt, t;
	for (t = 0.0; t < t_max; t += dt) {
		a = T.Acceleration(p, v, t);
#if 1
		am = T.Acceleration(p + v * (.5*dt), v + a * (.5*dt), t + .5*dt);
		p += (v + a * (.5*dt)) * dt;
		v += am * dt;
#else
		// am: a + h/2 da/dt + (h/2)² [ ∂²a/∂p² v v + ∂²a/∂v² a a + ∂²a/∂p∂v v a + ∂²a/∂v∂t a + ∂a/∂p a + ∂²a/∂t² ]
		am = T.Acceleration(p + v * (.5*dt) + a * (.25*dt*dt), v + a * (.5*dt), t + .5*dt);
		//p += v * dt + a * (0.5*dt*dt);
		p += v * dt + (a / 6. + am / 3.)*(dt*dt);
		v += am * dt;
#endif
		MidpointMethodPath.push_back(keyFrame{ t, p0, v0*dt });
		p0 = p, v0 = v;
	}
	MidpointMethodPath.push_back(keyFrame{ t, p0, v0*dt });
}

// red; standard Runge Kutta methods
// relies on continuous derivative; better when step size is small
Path RungeKuttaMethodPath;
void RungeKuttaMethodInit() {
	vec3 p0, p = p0 = T.P0, v0, v = v0 = T.V0;
	double t_max = T.t1, dt = 4.0*T.dt, t;
	for (t = 0.0; t < t_max; t += dt) {
		vec3 p1 = v * dt;
		vec3 v1 = T.Acceleration(p, v, t) * dt;
		vec3 p2 = (v + 0.5*v1) * dt;
		vec3 v2 = T.Acceleration(p + 0.5*p1, v + 0.5*v1, t + 0.5*dt) * dt;
		vec3 p3 = (v + 0.5*v2) * dt;
		vec3 v3 = T.Acceleration(p + 0.5*p2, v + 0.5*v2, t + 0.5*dt) * dt;
		vec3 p4 = (v + v3) * dt;
		vec3 v4 = T.Acceleration(p + p3, v + v3, t + dt) * dt;
		p += (p1 + 2.*p2 + 2.*p3 + p4) / 6.;
		v += (v1 + 2.*v2 + 2.*v3 + v4) / 6.;
		RungeKuttaMethodPath.push_back(keyFrame{ t, p0, v0*dt });
		p0 = p, v0 = v;
	}
	RungeKuttaMethodPath.push_back(keyFrame{ t, p0, v0*dt });
}

// yellow green; p: 1/6 h³ p³(t0); v: 5/12 h³ v³(t0) ??
// non-standard method, obtains derivative from the previous calculation, accuracy similar to Midpoint method
Path MultistepMidpointPath;
void MultistepMidpointInit() {
	vec3 p0, p = p0 = T.P0, v0, v = v0 = T.V0;
	double t_max = T.t1, dt = T.dt, t;
	vec3 a0, a = a0 = T.Acceleration(p0, v0, 0);
	for (t = 0.0; t < t_max; t += dt) {
		a = T.Acceleration(p, v, t);
		p += v * dt + a * (.5*dt*dt);
		v += (1.5*a - 0.5*a0) * dt;
		MultistepMidpointPath.push_back(keyFrame{ t, p0, v0*dt });
		p0 = p, v0 = v, a0 = a;
	}
	MultistepMidpointPath.push_back(keyFrame{ t, p0, v0*dt });
}

// sky blue; p: 1/12 h⁴ p⁴(t0); v: 5/12 h³ v³(t0) ????
// modified, works best when acceleration only depend on the position (eg. static gravity field)
Path MultistepVerletMethodPath;
void MultistepVerletMethodInit() {
	vec3 v0 = T.V0, p0 = T.P0;
	double t_max = T.t1, dt = T.dt, t;
	vec3 a = T.Acceleration(p0, v0, 0), a0 = T.Acceleration(p0 - v0 * dt + a * (.5*dt*dt), v0 - a * dt, -dt);
	vec3 v = v0 + a * dt, p = p0 + v0 * dt + a * (.5*dt*dt);
	for (t = 0.; t < t_max; t += dt) {
		a = T.Acceleration(p, v, t);
		vec3 p1 = 2.*p - p0 + a * (dt*dt);
		v += (1.5*a - 0.5*a0) * dt;
		MultistepVerletMethodPath.push_back(keyFrame{ t, p0, v0*dt });
		p0 = p, a0 = a, p = p1;
	}
	MultistepVerletMethodPath.push_back(keyFrame{ t, p0, v0*dt });
}


#include "numerical/linearsystem.h"
#include "numerical/optimization.h"

// as an experiment, may not be practical
// contains a bug as it sometimes changes direction rapidly
Path BackwardEulersMethodPath;
void BackwardEulersMethodInit() {
	vec3 p0, p = p0 = T.P0, v0, v = v0 = T.V0, a;
	double t_max = T.t1, dt = T.dt, t;
	double M[6][6]; vec3 x[2];
	for (t = 0.0; t < t_max; t += dt) {
		a = T.Acceleration(p, v, t);

		// numerical differentiation
		const double e = .0001;
		for (int i = 0; i < 36; i++) M[0][i] = 0;
		for (int i = 0; i < 3; i++) {
			vec3 d = vec3(i == 0, i == 1, i == 2) * e;
			*(vec3*)M[i] = (-.5 / e) * (T.Acceleration(p + d, v, t) - T.Acceleration(p - d, v, t));
			*((vec3*)&M[i + 3][3]) = (-.5 / e) * (T.Acceleration(p, v + d, t) - T.Acceleration(p, v - d, t));
			M[i][i] += 1. / dt, M[i + 3][i + 3] += 1. / dt;
		}
		vec3 dadt = (.5 / e) * (T.Acceleration(p, v, t + e) - T.Acceleration(p, v, t - e));

		// solve linear system
		x[0] = v + a * dt;
		x[1] = a + dadt * dt;
		solveLinear(6, &M[0][0], (double*)x);
		if (!(x[0].sqr() < 1e20 && x[1].sqr() < 1e20)) {
			x[0] = v * dt;
			x[1] = a * dt;
		}
		p += x[0];
		v += x[1];

		BackwardEulersMethodPath.push_back(keyFrame{ t, p0, v0*dt });
		p0 = p, v0 = v;
	}
	BackwardEulersMethodPath.push_back(keyFrame{ t, p0, v0*dt });
}

#pragma endregion



void render() {
	if (!_WINIMG) return;

	auto t0 = NTime::now();
	// initialize window
	for (int i = 0, l = _WIN_W * _WIN_H; i < l; i++) _WINIMG[i] = 0;
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
		drawLine_F(vec3(0, 0, -R), vec3(0, 0, R), 0x4040FF);
		drawCross3D(Center, 4, 0x00FF00);
	}


	// simulation
	if (!Alt) iTime = fmod(fsec(NTime::now() - start_time).count(), T.t1);

	// reference path
	plotPath(RefPath, iTime, 0x606080);

	// simulation paths
	bool drawCross = true;  // step highlighting
	plotPath(EulersMethodPath, iTime, 0xFF8000, drawCross);
	//plotPath(MidpointMethodPath, iTime, 0xFFFF00, drawCross);
	//plotPath(RungeKuttaMethodPath, iTime, 0xFF0000, drawCross);
	//plotPath(MultistepMidpointPath, iTime, 0xA0FF00, drawCross);
	//plotPath(MultistepVerletMethodPath, iTime, 0x0080FF, drawCross);
	plotPath(BackwardEulersMethodPath, iTime, 0x00FF00, drawCross);

	// additional rendering
	T.additional_render(iTime);

	double t = fsec(NTime::now() - t0).count();
	sprintf(text, "[%d×%d]  %.1fms (%.1ffps)\n", _WIN_W, _WIN_H, 1000.0*t, 1. / t);
	SetWindowTextA(_HWND, text);

	// why
	for (int i = 0, l = _WIN_W * _WIN_H; i < l; i++) _WINIMG[i] ^= -1;
}



#include <thread>
bool inited = false;
void Init() {
	if (inited) return; inited = true;
	initReferencePath();
	EulersMethodInit();
	MidpointMethodInit();
	RungeKuttaMethodInit();
	MultistepMidpointInit();
	MultistepVerletMethodInit();
	BackwardEulersMethodInit();
	new std::thread([](int x) {while (1) {
		SendMessage(_HWND, WM_NULL, NULL, NULL);
		Sleep(20);
	}}, 5);
}


// ============================================== User ==============================================


void keyDownShared(WPARAM _KEY) {
	if (_KEY == VK_CONTROL) Ctrl = true;
	else if (_KEY == VK_SHIFT) Shift = true;
	else if (_KEY == VK_MENU) Alt = true;
}
void keyUpShared(WPARAM _KEY) {
	if (_KEY == VK_CONTROL) Ctrl = false;
	else if (_KEY == VK_SHIFT) Shift = false;
	else if (_KEY == VK_MENU) Alt = false;
}


void WindowResize(int _oldW, int _oldH, int _W, int _H) {
	if (_W*_H == 0 || _oldW * _oldH == 0) return;  // window is minimized
	double pw = _oldW, ph = _oldH, w = _W, h = _H;
	double s = sqrt((w * h) / (pw * ph));
	Unit *= s, dist /= s;
}
void WindowClose() {
}

void MouseWheel(int _DELTA) {
	if (Ctrl) Center.z += 0.1 * _DELTA / Unit;
	else if (Shift) dist *= exp(-0.001*_DELTA);
	else {
		double s = exp(0.001*_DELTA);
		Unit *= s, dist /= s;
	}
}
void MouseDownL(int _X, int _Y) {
	clickCursor = Cursor = vec2(_X, _Y);
	mouse_down = true;
}
void MouseMove(int _X, int _Y) {
	vec2 P0 = Cursor, P = vec2(_X, _Y), D = P - P0;
	Cursor = P;

	if (mouse_down) {
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
			rz -= cos(ry)*d.x + sin(ry)*d.y, rx -= -sin(ry)*d.x + cos(ry)*d.y;
			//rz -= d.x, rx -= d.y;
		}
	}

}
void MouseUpL(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	bool moved = (int)length(clickCursor - Cursor) != 0;   // be careful: coincidence
	mouse_down = false;
}
void MouseDownR(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
}
void MouseUpR(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
#ifdef _DEBUG
	bool topmost = GetWindowLong(_HWND, GWL_EXSTYLE) & WS_EX_TOPMOST;
	SetWindowPos(_HWND, topmost ? HWND_NOTOPMOST : HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
#endif
}
void KeyDown(WPARAM _KEY) {
	keyDownShared(_KEY);
}
void KeyUp(WPARAM _KEY) {
	keyUpShared(_KEY);
}

