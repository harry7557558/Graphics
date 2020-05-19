// Test numerical ODE solving

// To-do:
// * Runge-Kutta method
// * Adaptive step size


// debug
#define _USE_CONSOLE 0

#include <cmath>
#include <stdio.h>
#include <algorithm>
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

// First Window (Main Window): UI Editor

#define WIN_NAME "UI"
#define WinW_Padding 100
#define WinH_Padding 100
#define WinW_Default 640
#define WinH_Default 400
#define WinW_Min 400
#define WinH_Min 300
#define WinW_Max 3840
#define WinH_Max 2160

void Init();  // only use this function to initialize variables (or test)
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

#pragma endregion


// Win32 Entry

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
#define _RDBK { HDC hdc = GetDC(_HWND), HImgMem = CreateCompatibleDC(hdc); HBITMAP hbmOld = (HBITMAP)SelectObject(HImgMem, _HIMG); render(); BitBlt(hdc, 0, 0, _WIN_W, _WIN_H, HImgMem, 0, 0, SRCCOPY); SelectObject(HImgMem, hbmOld), DeleteDC(HImgMem), DeleteDC(hdc); } break;
	switch (message) {
	case WM_NULL: {_RDBK}
	case WM_CREATE: { if (!_HWND) Init(); break; }
	case WM_CLOSE: { WindowClose(); DestroyWindow(hWnd); return 0; }
	case WM_DESTROY: { PostQuitMessage(0); return 0; }
	case WM_MOVE:; case WM_SIZE: {
		RECT Client; GetClientRect(hWnd, &Client); WindowResize(_WIN_W, _WIN_H, Client.right, Client.bottom); _WIN_W = Client.right, _WIN_H = Client.bottom;
		BITMAPINFO bmi; bmi.bmiHeader.biSize = sizeof(BITMAPINFO), bmi.bmiHeader.biWidth = Client.right, bmi.bmiHeader.biHeight = Client.bottom, bmi.bmiHeader.biPlanes = 1, bmi.bmiHeader.biBitCount = 32; bmi.bmiHeader.biCompression = BI_RGB, bmi.bmiHeader.biSizeImage = 0, bmi.bmiHeader.biXPelsPerMeter = bmi.bmiHeader.biYPelsPerMeter = 0, bmi.bmiHeader.biClrUsed = bmi.bmiHeader.biClrImportant = 0; bmi.bmiColors[0].rgbBlue = bmi.bmiColors[0].rgbGreen = bmi.bmiColors[0].rgbRed = bmi.bmiColors[0].rgbReserved = 0;
		if (_HIMG != NULL) DeleteObject(_HIMG); HDC hdc = GetDC(hWnd); _HIMG = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, (void**)&_WINIMG, NULL, 0); DeleteDC(hdc); _RDBK
	}
	case WM_GETMINMAXINFO: { LPMINMAXINFO lpMMI = (LPMINMAXINFO)lParam; lpMMI->ptMinTrackSize.x = WinW_Min, lpMMI->ptMinTrackSize.y = WinH_Min, lpMMI->ptMaxTrackSize.x = WinW_Max, lpMMI->ptMaxTrackSize.y = WinH_Max; break; }
	case WM_PAINT: { PAINTSTRUCT ps; HDC hdc = BeginPaint(hWnd, &ps), HMem = CreateCompatibleDC(hdc); HBITMAP hbmOld = (HBITMAP)SelectObject(HMem, _HIMG); BitBlt(hdc, 0, 0, _WIN_W, _WIN_H, HMem, 0, 0, SRCCOPY); SelectObject(HMem, hbmOld); EndPaint(hWnd, &ps); DeleteDC(HMem), DeleteDC(hdc); break; }
#define _USER_FUNC_PARAMS GET_X_LPARAM(lParam), _WIN_H - 1 - GET_Y_LPARAM(lParam)
	case WM_MOUSEMOVE: { MouseMove(_USER_FUNC_PARAMS); _RDBK }
	case WM_MOUSEWHEEL: { MouseWheel(GET_WHEEL_DELTA_WPARAM(wParam)); _RDBK }
	case WM_LBUTTONDOWN: { SetCapture(hWnd); MouseDownL(_USER_FUNC_PARAMS); _RDBK }
	case WM_LBUTTONUP: { ReleaseCapture(); MouseUpL(_USER_FUNC_PARAMS); _RDBK }
	case WM_RBUTTONDOWN: { MouseDownR(_USER_FUNC_PARAMS); _RDBK }
	case WM_RBUTTONUP: { MouseUpR(_USER_FUNC_PARAMS); _RDBK }
	case WM_SYSKEYDOWN:; case WM_KEYDOWN: { if (wParam >= 0x08) KeyDown(wParam); _RDBK }
	case WM_SYSKEYUP:; case WM_KEYUP: { if (wParam >= 0x08) KeyUp(wParam); _RDBK }
	} return DefWindowProc(hWnd, message, wParam, lParam);
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow) {
	if (_USE_CONSOLE) if (AttachConsole(ATTACH_PARENT_PROCESS) || AllocConsole()) freopen("CONIN$", "r", stdin), freopen("CONOUT$", "w", stdout), freopen("CONOUT$", "w", stderr);
	WNDCLASSEX wc; wc.cbSize = sizeof(WNDCLASSEX), wc.style = 0, wc.lpfnWndProc = WndProc, wc.cbClsExtra = wc.cbWndExtra = 0, wc.hInstance = hInstance; wc.hIcon = wc.hIconSm = 0, wc.hCursor = LoadCursor(NULL, IDC_ARROW), wc.hbrBackground = CreateSolidBrush(RGB(0, 0, 0)), wc.lpszMenuName = NULL, wc.lpszClassName = _T(WIN_NAME);
	if (!RegisterClassEx(&wc)) return -1;
	_HWND = CreateWindow(_T(WIN_NAME), _T(WIN_NAME), WS_OVERLAPPEDWINDOW, WinW_Padding, WinH_Padding, WinW_Default, WinH_Default, NULL, NULL, hInstance, NULL);
	ShowWindow(_HWND, nCmdShow); UpdateWindow(_HWND);
	MSG message; while (GetMessage(&message, 0, 0, 0)) { TranslateMessage(&message); DispatchMessage(&message); } return (int)message.wParam;
}

#pragma endregion


// ================================== Vector Classes/Functions ==================================

#pragma region Vector & Matrix

#define PI 3.1415926535897932384626
#define mix(x,y,a) ((x)*(1.0-(a))+(y)*(a))
#define clamp(x,a,b) ((x)<(a)?(a):(x)>(b)?(b):(x))
double mod(double x, double m) { return x - m * floor(x / m); }

class vec2 {
public:
	double x, y;
	explicit vec2() {}
	explicit vec2(const double &a) :x(a), y(a) {}
	explicit vec2(const double &x, const double &y) :x(x), y(y) {}
	vec2 operator - () const { return vec2(-x, -y); }
	vec2 operator + (const vec2 &v) const { return vec2(x + v.x, y + v.y); }
	vec2 operator - (const vec2 &v) const { return vec2(x - v.x, y - v.y); }
	vec2 operator * (const vec2 &v) const { return vec2(x * v.x, y * v.y); }	// not standard
	vec2 operator * (const double &a) const { return vec2(x*a, y*a); }
	double sqr() const { return x * x + y * y; } 	// not standard
	friend double length(const vec2 &v) { return sqrt(v.x*v.x + v.y*v.y); }
	friend vec2 normalize(const vec2 &v) { return v * (1. / sqrt(v.x*v.x + v.y*v.y)); }
	friend double dot(const vec2 &u, const vec2 &v) { return u.x*v.x + u.y*v.y; }
	friend double det(const vec2 &u, const vec2 &v) { return u.x*v.y - u.y*v.x; } 	// not standard
#if 1
	void operator += (const vec2 &v) { x += v.x, y += v.y; }
	void operator -= (const vec2 &v) { x -= v.x, y -= v.y; }
	void operator *= (const vec2 &v) { x *= v.x, y *= v.y; }
	friend vec2 operator * (const double &a, const vec2 &v) { return vec2(a*v.x, a*v.y); }
	void operator *= (const double &a) { x *= a, y *= a; }
	vec2 operator / (const double &a) const { return vec2(x / a, y / a); }
	void operator /= (const double &a) { x /= a, y /= a; }
#endif
	vec2 yx() const { return vec2(y, x); }
	vec2 rot() const { return vec2(-y, x); }
	vec2 rotr() const { return vec2(y, -x); }
#if 1
	// added when needed
	bool operator == (const vec2 &v) const { return x == v.x && y == v.y; }
	bool operator != (const vec2 &v) const { return x != v.x || y != v.y; }
	vec2 operator / (const vec2 &v) const { return vec2(x / v.x, y / v.y); }
	friend vec2 pMax(const vec2 &a, const vec2 &b) { return vec2(max(a.x, b.x), max(a.y, b.y)); }
	friend vec2 pMin(const vec2 &a, const vec2 &b) { return vec2(min(a.x, b.x), min(a.y, b.y)); }
	friend vec2 abs(const vec2 &a) { return vec2(abs(a.x), abs(a.y)); }
	friend vec2 floor(const vec2 &a) { return vec2(floor(a.x), floor(a.y)); }
	friend vec2 ceil(const vec2 &a) { return vec2(ceil(a.x), ceil(a.y)); }
	friend vec2 sqrt(const vec2 &a) { return vec2(sqrt(a.x), sqrt(a.y)); }
	friend vec2 sin(const vec2 &a) { return vec2(sin(a.x), sin(a.y)); }
	friend vec2 cos(const vec2 &a) { return vec2(cos(a.x), cos(a.y)); }
	friend vec2 atan(const vec2 &a) { return vec2(atan(a.x), atan(a.y)); }
#endif
};

class vec3 {
public:
	double x, y, z;
	explicit vec3() {}
	explicit vec3(const double &a) :x(a), y(a), z(a) {}
	explicit vec3(const double &x, const double &y, const double &z) :x(x), y(y), z(z) {}
	explicit vec3(const vec2 &v, const double &z) :x(v.x), y(v.y), z(z) {}
	vec3 operator - () const { return vec3(-x, -y, -z); }
	vec3 operator + (const vec3 &v) const { return vec3(x + v.x, y + v.y, z + v.z); }
	vec3 operator - (const vec3 &v) const { return vec3(x - v.x, y - v.y, z - v.z); }
	vec3 operator * (const vec3 &v) const { return vec3(x * v.x, y * v.y, z * v.z); }
	vec3 operator * (const double &k) const { return vec3(k * x, k * y, k * z); }
	double sqr() const { return x * x + y * y + z * z; } 	// non-standard
	friend double length(vec3 v) { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
	friend vec3 normalize(vec3 v) { return v * (1. / sqrt(v.x*v.x + v.y*v.y + v.z*v.z)); }
	friend double dot(vec3 u, vec3 v) { return u.x*v.x + u.y*v.y + u.z*v.z; }
	friend vec3 cross(vec3 u, vec3 v) { return vec3(u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x); }
#if 1
	void operator += (const vec3 &v) { x += v.x, y += v.y, z += v.z; }
	void operator -= (const vec3 &v) { x -= v.x, y -= v.y, z -= v.z; }
	void operator *= (const vec3 &v) { x *= v.x, y *= v.y, z *= v.z; }
	friend vec3 operator * (const double &a, const vec3 &v) { return vec3(a*v.x, a*v.y, a*v.z); }
	void operator *= (const double &a) { x *= a, y *= a, z *= a; }
	vec3 operator / (const double &a) const { return vec3(x / a, y / a, z / a); }
	void operator /= (const double &a) { x /= a, y /= a, z /= a; }
#endif
	vec2 xy() const { return vec2(x, y); }
	// vec2& xy() { return *(vec2*)this; }
	vec2 xz() const { return vec2(x, z); }
	vec2 yz() const { return vec2(y, z); }
#if 1
	bool operator == (const vec3 &v) const { return x == v.x && y == v.y && z == v.z; }
	bool operator != (const vec3 &v) const { return x != v.x || y != v.y || z != v.z; }
	vec3 operator / (const vec3 &v) const { return vec3(x / v.x, y / v.y, z / v.z); }
	friend vec3 pMax(const vec3 &a, const vec3 &b) { return vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }
	friend vec3 pMin(const vec3 &a, const vec3 &b) { return vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
	friend vec3 abs(const vec3 &a) { return vec3(abs(a.x), abs(a.y), abs(a.z)); }
	friend vec3 floor(const vec3 &a) { return vec3(floor(a.x), floor(a.y), floor(a.z)); }
	friend vec3 ceil(const vec3 &a) { return vec3(ceil(a.x), ceil(a.y), ceil(a.z)); }
	friend vec3 mod(const vec3 &a, double m) { return vec3(mod(a.x, m), mod(a.y, m), mod(a.z, m)); }
#endif
};

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


#pragma endregion Add __inline so compiler will expand them in debug mode


// ======================================== Data / Parameters ========================================

// viewport
vec3 Center(0.0, 0.0, 0.0);  // view center in world coordinate
double rz = -0.8, rx = 0.3, ry = 0.0, dist = 120.0, Unit = 100.0;  // yaw, pitch, row, camera distance, scale to screen

#pragma region General Global Variables

// window parameters
char text[64];	// window title
Affine Tr;  // matrix
vec3 CamP, ScrO, ScrA, ScrB;  // camera and screen
auto scrDir = [](vec2 pixel) { return normalize(ScrO + (pixel.x / _WIN_W)*ScrA + (pixel.y / _WIN_H)*ScrB - CamP); };

// user parameters
vec2 Cursor = vec2(0, 0), clickCursor;  // current cursor and cursor position when mouse down
bool mouse_down = false;
bool Ctrl = false, Shift = false, Alt = false;  // these variables are shared by both windows

#pragma endregion Window, Camera/Screen, Mouse/Key

#pragma region Global Variable Related Functions

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

// these functions often need to handle perspective case
void projRange_Triangle(vec3 A, vec3 B, vec3 C, vec2 &p0, vec2 &p1) {
	vec2 p = (Tr*A).xy(); p0 = p1 = p;
	p = (Tr*B).xy(); p0 = pMin(p0, p), p1 = pMax(p1, p);
	p = (Tr*C).xy(); p0 = pMin(p0, p), p1 = pMax(p1, p);
}
void projRange_Circle(vec3 P, vec3 u, vec3 v, vec2 &p0, vec2 &p1) {  // any space curve defined by C(t)=P+u*cos(t)+v*sin(t)
	vec2 Ru(dot(Tr.u, u), dot(Tr.v, u)), Rv(dot(Tr.u, v), dot(Tr.v, v)), Rp(dot(Tr.u, P), dot(Tr.v, P));
	double Pu = dot(Tr.p, u), Pv = dot(Tr.p, v), Pp = dot(Tr.p, P);
	vec2 a = Rv * (Pp + Tr.s) - Pv * (Rp + Tr.t.xy()), b = -Ru * (Pp + Tr.s) + Pu * (Rp + Tr.t.xy()), c = Pv * Ru - Rv * Pu;
	vec2 d = sqrt(a * a + b * b - c * c);
	vec2 t0 = 2.0*atan((b + d) / (a + c)), t1 = 2.0*atan((b - d) / (a + c));
	p0.x = (Tr * (P + cos(t0.x)*u + sin(t0.x)*v)).x, p1.x = (Tr * (P + cos(t1.x)*u + sin(t1.x)*v)).x; if (p0.x > p1.x) std::swap(p0.x, p1.x);
	p0.y = (Tr * (P + cos(t0.y)*u + sin(t0.y)*v)).y, p1.y = (Tr * (P + cos(t1.y)*u + sin(t1.y)*v)).y; if (p0.y > p1.y) std::swap(p0.y, p1.y);
}
void projRange_Cylinder(vec3 A, vec3 B, double r, vec2 &p0, vec2 &p1) {
	vec3 d = B - A, u = r * normalize(cross(d, vec3(1.2345, 6.5432, -1.3579))), v = r * normalize(cross(u, d));
	projRange_Circle(A, u, v, p0, p1);
	vec2 q0, q1; projRange_Circle(B, u, v, q0, q1);
	p0 = pMin(p0, q0), p1 = pMax(p1, q1);
}
void projRange_Sphere(vec3 P, double r, vec2 &p0, vec2 &p1) {
	/*if (dot(Tr.p, P) + Tr.s < r * length(Tr.p)) {
		//if (dot(Tr.p, P) + Tr.s < -r * length(Tr.p)) {
		p0 = p1 = vec2(NAN); return;
	}*/
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
void projRange_Cone(vec3 A, vec3 B, double r, vec2 &p0, vec2 &p1) {
	vec3 d = B - A, u = r * normalize(cross(d, vec3(1.2345, 6.5432, -1.3579))), v = r * normalize(cross(u, d));
	projRange_Circle(A, u, v, p0, p1);
	vec2 q = (Tr*B).xy();
	p0 = pMin(p0, q), p1 = pMax(p1, q);
}

#pragma endregion Get Ray/Screen, projection



// simulation test cases
// *: acceleration C0 continuity
// **: acceleration "breaks"
// `: solution contains chaos

// Unless otherwise stated, the magnitude of air resistance is propor to the cube of velosity instead of square

#define Projectile      0x00    // Acceleration due to gravity
#define Projectile_R    0x01    // Acceleration due to gravity + Air resistance
#define Projectile_RB   0x02    // *Drops onto an elastic surface and bounces up
#define Projectile_RW   0x03    // **Drops into some heavy liquid and bounces up
#define Pendulum        0x04    // On a one-meter-long non-deformable rod
#define Pendulum_S      0x05    // `On a one-meter-long spring with air resistance
#define NBody_1         0x06    // A "sun" in the center
#define NBody_2         0x07    // `Two immobilized "suns" with equal mass
#define NBody_m         0x08    // `One sun and one mobilized planet
#define High_Resistance 0x09    // Air resistance proper to the sixth power of velosity

#define SIMULATION NBody_1

#if SIMULATION==Projectile

const vec3 g = vec3(0, 0, -9.81);
auto Acceleration = [](vec3 p, vec3 v, double t)->vec3 {
	return g;
};
const double t_step = 0.1;
const double tMax = 2.0;
const vec3 P0 = vec3(0, 0, 1);
const vec3 V0 = vec3(1.5, 1.5, 2);

// A parabola opening down
// Euler: a little higher
// Modified Euler and Midpoint: exactly the same

#elif SIMULATION==Projectile_R

const vec3 g = vec3(0, 0, -9.81);
auto Acceleration = [](vec3 p, vec3 v, double t)->vec3 {
	double r = 0.1*dot(v, v);
	return g - r * v;
};
const double t_step = 0.06;
const double tMax = 2.0;
const vec3 P0 = vec3(0, 0, 1);
const vec3 V0 = vec3(2, 2, 3);

// Slightly deformed parabola, no significant velosity change
// Eulers: error
// Midpoint: error hard to notice

#elif SIMULATION==Projectile_RB

const vec3 g = vec3(0, 0, -9.81);
auto Acceleration = [](vec3 p, vec3 v, double t)->vec3 {
	vec3 r = -0.1*dot(v, v)*v;
	vec3 b = min(p.z, 0.)*vec3(0, 0, -50);
	return g + r + b;
};
const double t_step = 0.02;
const double tMax = 6.0;
const vec3 P0 = vec3(-2.5, 0, 1);
const vec3 V0 = vec3(3, 0, 3);

// Damping oscillation, limit below horizon
// Euler: shorter wavelength, greater magnitude
// Modified Euler: slightly gentler
// Midpoint: few error

#elif SIMULATION==Projectile_RW

const vec3 g = vec3(0, 0, -9.81);
auto Acceleration = [](vec3 p, vec3 v, double t)->vec3 {
	vec3 r = -(p.z > 0. ? 0.05 : 0.5)*dot(v, v)*v;
	vec3 b = (p.z < 0. ? 1. : 0.)*vec3(0, 0, 50);
	return g + r + b;
};
const double t_step = 0.01;
const double tMax = 6.0;
const vec3 P0 = vec3(-3, 0, 1);
const vec3 V0 = vec3(3, 0, 3);

// Damping oscillation, converges to horizon
// Eulers: shorter wavelength and greater magnitude
// Midpoint: longer wavelength and less magnitude
// Midpoint works slightly better than Eulers after reducing t_step

#elif SIMULATION==Pendulum

const double g = 9.81;
auto Acceleration = [](vec3 p, vec3 v, double t)->vec3 {
	p -= vec3(0, 0, 1);
	vec3 u = normalize(cross(p, cross(p, veck))) * (g * length(p.xy()) / length(p));
	vec3 w = -p * dot(v, v);
	return u + w;
};
const double t_step = 0.02;
const double tMax = 6.0;
const vec3 P0 = vec3(0, 1, 1);
const vec3 V0 = vec3(1, 0, 0);

// Flower-liked path; forms a semisphere with a hole at the buttom after a long time
// Euler flies away, then the modified Euler
// Midpoint doesn't flie away and doesn't follow the path exactly

#elif SIMULATION==Pendulum_S

const vec3 g = vec3(0, 0, -9.81);
auto Acceleration = [](vec3 p, vec3 v, double t)->vec3 {
	vec3 d = p - vec3(0, 0, 1);
	vec3 N = -10.0*(length(d) - 1)*normalize(d);
	vec3 r = -0.01*dot(v, v)*v;
	return g + N + r;
};
const double t_step = 0.05;
const double tMax = 6.0;
const vec3 P0 = vec3(0, 1, 0);
const vec3 V0 = vec3(3, 0, 3);

// Chaos - "tangled clew"
// Eulers' path have longer radius, modified Euler is slightly smaller
// Whe tMax is set to 6, Midpoint has few noticeble error

#elif SIMULATION==NBody_1

auto Acceleration = [](vec3 p, vec3 v, double t)->vec3 {
	double m = length(p);
	return -20.0*p / (m*m*m);
};
const double t_step = 0.05;
const double tMax = 3.0;
const vec3 P0 = vec3(2, 2, 0);
const vec3 V0 = vec3(1, -1, -0.2);

// A high-eccentricity ellipse with a rapid speed change
// All three curves fly away close to the perihelion
// The reference curve doesn't go back to its original point as it suppose to
// Midpoint curve goes back to origin when t_step is set to 0.002
// This test demonstrates the importance of adaptive step length

#elif SIMULATION==NBody_2

auto Acceleration = [](vec3 p, vec3 v, double t)->vec3 {
	vec3 q = p - vec3(0, 1, 0);
	double m = length(q);
	vec3 a = q / (m*m*m);
	q = p + vec3(0, 1, 0);
	m = length(q);
	a += q / (m*m*m);
	return -10.0*a;
};
const double t_step = 0.01;
const double tMax = 8.0;
const vec3 P0 = vec3(2, 2, 0);
const vec3 V0 = vec3(1, -0.5, -0.2);

// Eulers: circulates one sun and flies away
// Midpoint and Reference: circulates one sun, then attracted by the other sun and rapidly turns and fly away in different directions
// Midpoint has a greater velosity than other points
// Note that the points don't have enough energy to get rid of the gravity

#elif SIMULATION==NBody_m

auto Acceleration = [](vec3 p, vec3 v, double t)->vec3 {
	double m = length(p);
	vec3 F = -9.*p / (m*m*m);
	p -= vec3(cos(3.*t), sin(3.*t), 0);
	m = length(p);
	return F - p / (m*m*m);
};
const double t_step = 0.01;
const double tMax = 2.*PI;
const vec3 P0 = vec3(2, 2, 0);
const vec3 V0 = vec3(1, -0.5, -0.2);

// "Interfered" by the gravity of the planet after the perihelion and flies away
// Eulers "deviate" the "orbit" for long
// Midpoint deviates after the interference
// Seems like that point will go back after several minutes

#elif SIMULATION==High_Resistance

const vec3 g = vec3(0, 0, -9.81);
auto Acceleration = [](vec3 p, vec3 v, double t)->vec3 {
	double m = dot(v, v);
	return g - m * m*m*normalize(v);
};
const double t_step = 0.01;
const double tMax = 1.0;
const vec3 P0 = vec3(0, 0, 1);
const vec3 V0 = vec3(20, 10, 0);

// Intended to test the limit of the solver - to Neptune!
// Set t_step to 1e-7: shoots horizontally, then turns and falls vertically

#endif



// ============================================ Rendering ============================================

#pragma region Rasterization functions

auto drawLine = [](vec2 p, vec2 q, COLORREF col) {
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
auto drawCross = [&](vec2 p, double r, COLORREF Color = 0xFFFFFF) {
	drawLine(p - vec2(r, 0), p + vec2(r, 0), Color);
	drawLine(p - vec2(0, r), p + vec2(0, r), Color);
};
auto drawCircle = [&](vec2 c, double r, COLORREF Color) {
	int s = int(r / sqrt(2) + 0.5);
	int cx = (int)c.x, cy = (int)c.y;
	for (int i = 0, im = min(s, max(_WIN_W - cx, cx)) + 1; i < im; i++) {
		int u = sqrt(r*r - i * i) + 0.5;
		setColor(cx + i, cy + u, Color); setColor(cx + i, cy - u, Color); setColor(cx - i, cy + u, Color); setColor(cx - i, cy - u, Color);
		setColor(cx + u, cy + i, Color); setColor(cx + u, cy - i, Color); setColor(cx - u, cy + i, Color); setColor(cx - u, cy - i, Color);
	}
};
auto fillCircle = [&](vec2 c, double r, COLORREF Color) {
	int x0 = max(0, int(c.x - r)), x1 = min(_WIN_W - 1, int(c.x + r));
	int y0 = max(0, int(c.y - r)), y1 = min(_WIN_H - 1, int(c.y + r));
	int cx = (int)c.x, cy = (int)c.y, r2 = int(r*r);
	for (int x = x0, dx = x - cx; x <= x1; x++, dx++) {
		for (int y = y0, dy = y - cy; y <= y1; y++, dy++) {
			if (dx * dx + dy * dy < r2) Canvas(x, y) = Color;
		}
	}
};
auto drawBox = [](vec2 Min, vec2 Max, COLORREF col = 0xFF0000) {
	drawLine(vec2(Min.x, Min.y), vec2(Max.x, Min.y), col);
	drawLine(vec2(Max.x, Min.y), vec2(Max.x, Max.y), col);
	drawLine(vec2(Max.x, Max.y), vec2(Min.x, Max.y), col);
	drawLine(vec2(Min.x, Max.y), vec2(Min.x, Min.y), col);
};
auto fillBox = [](vec2 Min, vec2 Max, COLORREF col = 0xFF0000) {
	int x0 = max((int)Min.x, 0), x1 = min((int)Max.x, _WIN_W - 1);
	int y0 = max((int)Min.y, 0), y1 = min((int)Max.y, _WIN_H - 1);
	for (int x = x0; x <= x1; x++) for (int y = y0; y <= y1; y++) Canvas(x, y) = col;
};
auto drawSquare = [](vec2 C, double r, COLORREF col = 0xFFA500) {
	drawBox(C - vec2(r, r), C + vec2(r, r), col);
};
auto fillSquare = [](vec2 C, double r, COLORREF col = 0xFFA500) {
	fillBox(C - vec2(r, r), C + vec2(r, r), col);
};

auto drawLine_F = [](vec3 A, vec3 B, COLORREF col = 0xFFFFFF) {
	//if (col != 0x404040) return;
	double u = dot(Tr.p, A) + Tr.s, v = dot(Tr.p, B) + Tr.s;
	if (u > 0 && v > 0) { drawLine((Tr*A).xy(), (Tr*B).xy(), col); return; }
	if (u < 0 && v < 0) return;
	if (u < v) std::swap(A, B), std::swap(u, v);
	double t = u / (u - v) - 1e-4;
	B = A + (B - A)*t;
	drawLine((Tr*A).xy(), (Tr*B).xy(), col);
};
auto drawCross3D = [&](vec3 P, double r, COLORREF col = 0xFFFFFF, bool relative = true) {
	return;  // comment this line to make it like barbed wire
	if (relative) r *= dot(Tr.p, P) + Tr.s;
	drawLine_F(P - vec3(r, 0, 0), P + vec3(r, 0, 0), col);
	drawLine_F(P - vec3(0, r, 0), P + vec3(0, r, 0), col);
	drawLine_F(P - vec3(0, 0, r), P + vec3(0, 0, r), col);
};

#pragma endregion


#include <chrono>
typedef std::chrono::high_resolution_clock NTime;
typedef std::chrono::duration<double> fsec;

auto start_time = NTime::now();
double iTime;

COLORREF toCOLORREF(const vec3 &col) {
	COLORREF C; byte* c = (byte*)&C;
	c[2] = (byte)(255 * clamp(col.x, 0., 1.));
	c[1] = (byte)(255 * clamp(col.y, 0., 1.));
	c[0] = (byte)(255 * clamp(col.z, 0., 1.));
	return C;
}


// testcase utility
bool drawTestcase(double t) {
#if SIMULATION==NBody_2
	fillCircle((Tr*vec3(0, 1, 0)).xy(), 4, 0x00A0FF);
	fillCircle((Tr*vec3(0, -1, 0)).xy(), 4, 0x00A0FF);
#elif SIMULATION==NBody_m
	//fillCircle((Tr*vec0).xy(), 5, 0xFF4000);
	fillCircle((Tr*vec3(cos(3.*t), sin(3.*t), 0)).xy(), 4, 0x00FF00);
#else
	return false;
#endif
	return true;
}


#define plotPath(Color) \
	drawLine_F(p0, p, Color); \
	drawCross3D(p, 2, Color); \
	double u = t + dt - iTime; if (u > 0 && u < dt) { u /= dt; fillCircle((Tr*(u*p0 + (1 - u)*p)).xy(), 6, Color); }

// orange red
void EulersMethod(vec3 p0, vec3 v0, vec3(*acceleration)(vec3, vec3, double), double dt, double tMax) {  // O(h^2)
	vec3 p = p0, v = v0, a;
	for (double t = 0.0; t < tMax; t += dt) {
		a = acceleration(p, v, t);
		p += v * dt;
		v += a * dt;
		plotPath(0xFF8000);
		p0 = p, v0 = v;
	}
}

// orange, error no much smaller than simple Euler's method
void EulersMethod_Modified(vec3 p0, vec3 v0, vec3(*acceleration)(vec3, vec3, double), double dt, double tMax) {  // p: O(h^3); v: O(h^2)
	vec3 p = p0, v = v0, a;
	for (double t = 0.0; t < tMax; t += dt) {
		a = acceleration(p, v, t);
		p += v * dt + a * (.5*dt*dt);
		v += a * dt;
		plotPath(0xFFA000);
		p0 = p, v0 = v;
	}
}

// yellow
void Euler_Midpoint(vec3 p0, vec3 v0, vec3(*acceleration)(vec3, vec3, double), double dt, double tMax) {  // O(h^3)
	dt *= 2.0;  // to be fair
	vec3 p = p0, v = v0, a;
	for (double t = 0.0; t < tMax; t += dt) {
		a = acceleration(p, v, t);
		vec3 _p = p;
		p += v * dt + a * (.5*dt*dt);
		v += acceleration(.5*(_p + p), v + a * (.5*dt), t + .5*dt) * dt;
		plotPath(0xFFFF00);
		p0 = p, v0 = v;
	}
}

void render() {
	if (!_WINIMG) return;

	auto t0 = NTime::now();
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
		drawLine_F(vec3(0, 0, -R), vec3(0, 0, R), 0x4040FF);
		drawCross3D(Center, 4, 0x00FF00);
	}


	// simulation
	if (!Alt) iTime = fmod(fsec(NTime::now() - start_time).count(), tMax);

	// reference path
	const double dt = 0.0001;
	vec3 p0 = P0, v0 = V0, p = p0, v = v0, a;
	for (double t = 0.0; t < tMax; t += dt) {
		a = Acceleration(p, v, t);
		p += v * dt, v += a * dt;
		drawLine_F(p0, p, 0x606080);
		p0 = p, v0 = v;
		double u = fmod(t, t_step);
		if (u >= 0. && u <= dt) drawCross3D(p, 2, 0x606080);
		if (t > iTime && t - iTime < dt) fillCircle((Tr*p).xy(), 6, 0x606080);
	}

	// simulation paths with large step
	EulersMethod(P0, V0, Acceleration, t_step, tMax);
	EulersMethod_Modified(P0, V0, Acceleration, t_step, tMax);
	Euler_Midpoint(P0, V0, Acceleration, t_step, tMax);

	drawTestcase(iTime);

	double t = fsec(NTime::now() - t0).count();
	sprintf(text, "[%d×%d]  %.1fms (%.1ffps)\n", _WIN_W, _WIN_H, 1000.0*t, 1. / t);
	SetWindowTextA(_HWND, text);
}


// ============================================== User ==============================================


#include <thread>
bool inited = false;
void Init() {
	if (inited) return; inited = true;
	new std::thread([](int x) {while (1) {
		SendMessage(_HWND, WM_NULL, NULL, NULL);
		Sleep(20);
	}}, 5);
}


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
	if (Ctrl) {
		Center.z += 0.1 * _DELTA / Unit;
	}
	else if (Shift) {
		double s = exp(-0.001*_DELTA);
		dist *= s;
	}
	else {
		// zoom
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

#if _USE_CONSOLE
	vec3 d = scrDir(Cursor);
	Triangle* obj = 0; double t = INFINITY;
	rayIntersectBVH(BVH_R, CamP, d, vec3(1.0) / d, t, obj);
	if (obj) printf("%d\n", obj - STL);
	else printf("-1\n");
	//if (obj) *obj = Triangle{ vec3(0),vec3(0),vec3(0),vec3(0) };
#endif
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

