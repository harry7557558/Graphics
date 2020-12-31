// (simple??) Win32 3D GUI template

// Default UI:
// Perspective axis & grid with default scenes
// Drag to rotate, scroll to zoom
// Shift+Drag camera roll, Shift+Scroll camera perspective
// Adjust center of scene: Ctrl+Drag xy, Ctrl+Scroll z
// Home / Ctrl+0 to reset viewport
// Right click to topmost the window

#include <cmath>
#include <stdio.h>
#include <algorithm>
#pragma warning(disable: 4244)  // data type conversion warning

// ========================================= Win32 Standard =========================================

#pragma region Windows

#ifndef UNICODE
#define UNICODE
#endif

#include <Windows.h>
#include <windowsx.h>
#include <tchar.h>


// debug
// _USE_CONSOLE: create a console for output
// otherwise: print to Visual Studio output window
#define _USE_CONSOLE 0
wchar_t _DEBUG_OUTPUT_BUF[0x1000];
#define dbgprint(format, ...) { if (_USE_CONSOLE) {printf(format, ##__VA_ARGS__);} else {swprintf(_DEBUG_OUTPUT_BUF, 0x1000, _T(format), ##__VA_ARGS__); OutputDebugStringW(_DEBUG_OUTPUT_BUF);} }


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
#if 1
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow) {
	if (_USE_CONSOLE) if (AttachConsole(ATTACH_PARENT_PROCESS) || AllocConsole()) freopen("CONIN$", "r", stdin), freopen("CONOUT$", "w", stdout), freopen("CONOUT$", "w", stderr);
#else
int main() {
	HINSTANCE hInstance = NULL; int nCmdShow = SW_RESTORE; if (!_USE_CONSOLE) FreeConsole();
#endif
	WNDCLASSEX wc; wc.cbSize = sizeof(WNDCLASSEX), wc.style = 0, wc.lpfnWndProc = WndProc, wc.cbClsExtra = wc.cbWndExtra = 0, wc.hInstance = hInstance; wc.hIcon = wc.hIconSm = 0, wc.hCursor = LoadCursor(NULL, IDC_ARROW), wc.hbrBackground = CreateSolidBrush(RGB(0, 0, 0)), wc.lpszMenuName = NULL, wc.lpszClassName = _T(WIN_NAME); if (!RegisterClassEx(&wc)) return -1;
	_HWND = CreateWindow(_T(WIN_NAME), _T(WIN_NAME), WS_OVERLAPPEDWINDOW, WinW_Padding, WinH_Padding, WinW_Default, WinH_Default, NULL, NULL, hInstance, NULL); ShowWindow(_HWND, nCmdShow); UpdateWindow(_HWND);
	MSG message; while (GetMessage(&message, 0, 0, 0)) { TranslateMessage(&message); DispatchMessage(&message); } return (int)message.wParam;
}

#pragma endregion  // WIN32


// ================================== Vector Classes/Functions ==================================

#pragma region Vector & Matrix

#include "numerical/geometry.h"
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

// Many from https://www.iquilezles.org/www/articles/intersectors/intersectors.htm
// Direction vectors should be normalized

// Intersection functions - return the distance, NAN means no intersection
double intHorizon(double z, vec3 p, vec3 d) {
	return (z - p.z) / d.z;
}
double intSphere(vec3 O, double r, vec3 p, vec3 d) {
	p = p - O;
#if 0
	if (dot(p, d) >= 0.0) return NAN;
	vec3 k = cross(p, d); double rd2 = dot(k, k); if (rd2 > r*r) return NAN;
	return sqrt(dot(p, p) - rd2) - sqrt(r*r - rd2);
#else
	// works when p is inside the sphere (and its slightly faster)
	double b = -dot(p, d), c = dot(p, p) - r * r;  // required to be normalized
	double delta = b * b - c;
	if (delta < 0.0) return NAN;
	delta = sqrt(delta);
	c = b - delta;
	return c > 0. ? c : b + delta;  // usually we want it to be positive
#endif
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
double intCapsule(vec3 pa, vec3 pb, double r, vec3 ro, vec3 rd) {
	vec3 ba = pb - pa, oa = ro - pa;
	double baba = dot(ba, ba), bard = dot(ba, rd), baoa = dot(ba, oa), rdoa = dot(rd, oa), oaoa = dot(oa, oa);
	double a = baba - bard * bard, b = baba * rdoa - baoa * bard, c = baba * oaoa - baoa * baoa - r * r*baba;
	double h = b * b - a * c;
	if (h >= 0.0) {
		double t = (-b - sqrt(h)) / a;
		double y = baoa + t * bard;
		if (y > 0.0 && y < baba) return t;
		vec3 oc = (y <= 0.0) ? oa : ro - pb;
		b = dot(rd, oc), c = dot(oc, oc) - r * r, h = b * b - c;
		if (h > 0.0) return -b - sqrt(h);
	}
	return NAN;
}
double intCylinder(vec3 pa, vec3 pb, double ra, vec3 ro, vec3 rd) {
	vec3 ca = pb - pa, oc = ro - pa;
	double caca = dot(ca, ca), card = dot(ca, rd), caoc = dot(ca, oc);
	double a = caca - card * card, b = caca * dot(oc, rd) - caoc * card, c = caca * dot(oc, oc) - caoc * caoc - ra * ra*caca;
	double h = b * b - a * c;
	if (h < 0.0) return NAN;
	h = sqrt(h);
	double t = (-b - h) / a;
	double y = caoc + t * card;
	if (y > 0.0 && y < caca) return t;
	t = (((y < 0.0) ? 0.0 : caca) - caoc) / card;
	if (abs(b + a * t) < h) return t;
	return NAN;
}
double intBoxC(vec3 R, vec3 ro, vec3 rd) {
	vec3 m = vec3(1.0) / rd, n = m * ro;
	vec3 k = abs(m)*R;
	vec3 t1 = -n - k, t2 = -n + k;
	double tN = max(max(t1.x, t1.y), t1.z);
	double tF = min(min(t2.x, t2.y), t2.z);
	if (tN > tF || tF < 0.0) return NAN;
	return tN;
}
double intCone(vec3 pa, vec3 pb, double r, vec3 ro, vec3 rd) {
	vec3 ba = pb - pa, oa = ro - pa, ob = ro - pb;
	double m0 = dot(ba, ba), m1 = dot(oa, ba), m2 = dot(ob, ba), m3 = dot(rd, ba);
	if (m1 < 0.0 && (oa*m3 - rd * m1).sqr() < (r*r*m3*m3)) return (-m1 / m3);
	double m4 = dot(rd, oa), m5 = dot(oa, oa);
	double rr = r, hy = m0 + rr * rr;
	double k2 = m0 * m0 - m3 * m3*hy;
	double k1 = m0 * m0*m4 - m1 * m3*hy + m0 * r*(rr*m3*1.0);
	double k0 = m0 * m0*m5 - m1 * m1*hy + m0 * r*(rr*m1*2.0 - m0 * r);
	double h = k1 * k1 - k2 * k0; if (h < 0.0) return NAN;
	double t = (-k1 - sqrt(h)) / k2;
	double y = m1 + t * m3; if (y > 0.0 && y < m0) return t;
	return NAN;
}
double intCircle(vec3 n, vec3 c, double r, vec3 ro, vec3 rd) {
	vec3 q = ro - c;
	double t = -dot(n, q) / dot(rd, n);
	q = q + rd * t;
	return dot(q, q) < r*r ? t : NAN;
}


// Normal calculation
vec3 nCapsule(vec3 a, vec3 b, double r, vec3 p) {
	vec3 ba = b - a, pa = p - a;
	double h = dot(pa, ba) / dot(ba, ba);
	return (pa - clamp(h, 0., 1.) * ba) / r;
}

// Distance functions
double sdBox(vec2 p, vec2 b) {
	vec2 d = abs(p) - b;
	return length(pMax(d, vec2(0))) + min(max(d.x, d.y), 0.0);
}
double sdBox(vec3 p, vec3 b) {
	vec3 q = abs(p) - b;
	return length(pMax(q, vec0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}
double sdCapsule(vec3 p, vec3 a, vec3 b, double r) {
	vec3 pa = p - a, ba = b - a;
	double h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
	return length(pa - ba * h) - r;
}
double sdEllipsoid(vec3 p, vec3 r) {  // approximation
	double k0 = length(p / r);
	double k1 = length(p / (r*r));
	return k0 * (k0 - 1.0) / k1;
}
double sdCylinder(vec3 p, vec3 a, vec3 b, double r) {
	vec3  ba = b - a;
	vec3  pa = p - a;
	double baba = dot(ba, ba);
	double paba = dot(pa, ba);
	double x = length(pa*baba - ba * paba) - r * baba;
	double y = abs(paba - baba * 0.5) - baba * 0.5;
	double x2 = x * x;
	double y2 = y * y*baba;
	double d = (max(x, y) < 0.0) ? -min(x2, y2) : (((x > 0.0) ? x2 : 0.0) + ((y > 0.0) ? y2 : 0.0));
	return (d > 0 ? 1. : -1.)*sqrt(abs(d)) / baba;
}

// smoothed union/intersection
double smin(double d1, double d2, double k) {
	double h = 0.5 + 0.5*(d2 - d1) / k; h = clamp(h, 0.0, 1.0);
	return mix(d2, d1, h) - k * h*(1.0 - h);
}
double smax(double d1, double d2, double k) {
	double h = 0.5 - 0.5*(d2 - d1) / k; h = clamp(h, 0.0, 1.0);
	return mix(d2, d1, h) + k * h*(1.0 - h);
}

// Bounding box calculation
void rangeSphere(vec3 c, double r, vec3 &Min, vec3 &Max) {
	Min = c - vec3(r), Max = c + vec3(r);
}
void rangeCapsule(vec3 pa, vec3 pb, double r, vec3 &Min, vec3 &Max) {
	Min = pMin(pa, pb) - vec3(r), Max = pMax(pa, pb) + vec3(r);
}

// closest point to a straight line
double closestPoint(vec3 P, vec3 d, vec3 ro, vec3 rd) {  // P+t*d, return t
	vec3 n = cross(rd, cross(rd, d));
	return dot(ro - P, n) / dot(d, n);
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

// get the range of projection on screen
// some have issues when camera is before/inside the object
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
	// This code might be made much shorter
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


// ============================================ Rendering ============================================

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
void drawCircle(vec2 c, double r, COLORREF Color) {
	int s = int(r / sqrt(2) + 0.5);
	int cx = (int)c.x, cy = (int)c.y;
	for (int i = 0, im = min(s, max(_WIN_W - cx, cx)) + 1; i < im; i++) {
		int u = sqrt(r*r - i * i) + 0.5;
		setColor(cx + i, cy + u, Color); setColor(cx + i, cy - u, Color); setColor(cx - i, cy + u, Color); setColor(cx - i, cy - u, Color);
		setColor(cx + u, cy + i, Color); setColor(cx + u, cy - i, Color); setColor(cx - u, cy + i, Color); setColor(cx - u, cy - i, Color);
	}
}
void fillCircle(vec2 c, double r, COLORREF Color) {
	int x0 = max(0, int(c.x - r)), x1 = min(_WIN_W - 1, int(c.x + r));
	int y0 = max(0, int(c.y - r)), y1 = min(_WIN_H - 1, int(c.y + r));
	int cx = (int)c.x, cy = (int)c.y, r2 = int(r*r);
	for (int x = x0, dx = x - cx; x <= x1; x++, dx++) {
		for (int y = y0, dy = y - cy; y <= y1; y++, dy++) {
			if (dx * dx + dy * dy < r2) Canvas(x, y) = Color;
		}
	}
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
void drawTriangle_F(vec3 A, vec3 B, vec3 C, COLORREF col) {
	double u = dot(Tr.p, A) + Tr.s, v = dot(Tr.p, B) + Tr.s, w = dot(Tr.p, C) + Tr.s;
	if (u > 0 && v > 0 && w > 0) { drawTriangle((Tr*A).xy(), (Tr*B).xy(), (Tr*C).xy(), col); return; }
	if (u < 0 && v < 0 && w < 0) return;
	// debug
}
void drawCross3D(vec3 P, double r, COLORREF col = 0xFFFFFF) {
	r *= dot(Tr.p, P) + Tr.s;
	drawLine_F(P - vec3(r, 0, 0), P + vec3(r, 0, 0), col);
	drawLine_F(P - vec3(0, r, 0), P + vec3(0, r, 0), col);
	drawLine_F(P - vec3(0, 0, r), P + vec3(0, 0, r), col);
}


// simply fill color, no normal calculation
void drawRod_RT(vec3 A, vec3 B, double r, COLORREF col) {
	vec2 p0, p1; projRange_Cylinder(A, B, r, p0, p1);
	int x0 = max((int)p0.x, 0), x1 = min((int)p1.x, _WIN_W - 1), y0 = max((int)p0.y, 0), y1 = min((int)p1.y, _WIN_H - 1);
	double t; for (int i = x0; i <= x1; i++) for (int j = y0; j <= y1; j++)
		if ((t = intCylinder(A, B, r, CamP, scrDir(vec2(i, j)))) > 0 && t < _DEPTHBUF[i][j]) Canvas(i, j) = col, _DEPTHBUF[i][j] = t;
}
void drawSphere_RT(vec3 P, double r, COLORREF col) {
	vec2 p0, p1; projRange_Sphere(P, r, p0, p1);
	int x0 = max((int)p0.x, 0), x1 = min((int)p1.x, _WIN_W - 1), y0 = max((int)p0.y, 0), y1 = min((int)p1.y, _WIN_H - 1);
	double t; for (int i = x0; i <= x1; i++) for (int j = y0; j <= y1; j++)
		if ((t = intSphere(P, r, CamP, scrDir(vec2(i, j)))) > 0 && t < _DEPTHBUF[i][j]) Canvas(i, j) = col, _DEPTHBUF[i][j] = t;
}
void drawTriangle_RT(vec3 A, vec3 B, vec3 C, COLORREF col) {
	vec2 p0, p1; projRange_Triangle(A, B, C, p0, p1);
	int x0 = max((int)p0.x, 0), x1 = min((int)p1.x, _WIN_W - 1), y0 = max((int)p0.y, 0), y1 = min((int)p1.y, _WIN_H - 1);
	double t; for (int i = x0; i <= x1; i++) for (int j = y0; j <= y1; j++)
		if ((t = intTriangle(A, B, C, CamP, scrDir(vec2(i, j)))) > 0 && t < _DEPTHBUF[i][j]) Canvas(i, j) = col, _DEPTHBUF[i][j] = t;
	// can be accelerated by precomputing the triangle edges
}
void drawArrow_RT(vec3 A, vec3 B, double r, COLORREF col) {
	vec2 p0, p1; projRange_Cone(A, B, r, p0, p1);
	int x0 = max((int)p0.x, 0), x1 = min((int)p1.x, _WIN_W - 1), y0 = max((int)p0.y, 0), y1 = min((int)p1.y, _WIN_H - 1);
	double t; for (int i = x0; i <= x1; i++) for (int j = y0; j <= y1; j++)
		if ((t = intCone(A, B, r, CamP, scrDir(vec2(i, j)))) > 0 && t < _DEPTHBUF[i][j]) Canvas(i, j) = col, _DEPTHBUF[i][j] = t;
}
void drawCircle_RT(vec3 C, vec3 n, double r, COLORREF col) {
	vec3 u = r * normalize(cross(n, vec3(1.2345, 6.5432, -1.3579))), v = cross(u, normalize(n));
	vec2 p0, p1; projRange_Circle(C, u, v, p0, p1);
	int x0 = max((int)p0.x, 0), x1 = min((int)p1.x, _WIN_W - 1), y0 = max((int)p0.y, 0), y1 = min((int)p1.y, _WIN_H - 1);
	double t; for (int i = x0; i <= x1; i++) for (int j = y0; j <= y1; j++)
		if ((t = intCircle(n, C, r, CamP, scrDir(vec2(i, j)))) > 0 && t < _DEPTHBUF[i][j]) Canvas(i, j) = col, _DEPTHBUF[i][j] = t;
	// n is already here so recalculation in the intersection function is unnecessary
}

void drawVector_RT(vec3 P, vec3 d, double r, COLORREF col, bool relative = true) {
	double SC = relative ? dot(P + .5*d, Tr.p) + Tr.s : 1.0;
	drawRod_RT(P, P + d, r * SC, col);
	if (relative) SC = dot(P + d, Tr.p) + Tr.s;
	drawArrow_RT(P + d, P + ((3.5*r*SC + 1.)*d), 2.*r *SC, col);
}

#pragma endregion



#include <chrono>

void render() {
	// time recorder
	auto _time_0 = std::chrono::high_resolution_clock::now();

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

	// default scene
	{
		drawRod_RT(vec0, veci, 0.05, 0xff0000);
		drawRod_RT(vec0, vecj, 0.05, 0x008000);
		drawRod_RT(vec0, veck, 0.05, 0x0000ff);
		drawSphere_RT(vec3(0.5, 0, 0), 0.1, 0xffff00);
		drawTriangle_RT(vec0, veci, vecj, 0x808080);
		drawArrow_RT(vec3(0.3, 0.3, 0), vec3(.3, .3, .5), .1, 0xff00ff);
		drawCircle_RT(vec3(0.2, 0.5, 0.1), vec3(2, 1, 1), 0.1, 0xff8000);
		drawVector_RT(vec3(1, 0, .4), vec3(-1, 1, .5), 0.03, 0x800080, false);
	}

	// the actual fps is less because of displaying time
	double time_elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - _time_0).count();
	dbgprint("%lfms (%.1lf fps)\n", 1000 * time_elapsed, 1.0 / time_elapsed);
}


// ============================================== User ==============================================


void Init() {
}


// read/save files
bool saveFile(const WCHAR* filename) {
	FILE *fp = _wfopen(filename, L"ab");  // or "wb"
	return false;
}
bool readFile(const WCHAR* filename) {
	FILE *fp = _wfopen(filename, L"rb");
	return false;
}
bool saveFileUserEntry() {
	OPENFILENAME ofn = { sizeof(OPENFILENAME) };
	WCHAR filename[MAX_PATH] = L"";
	ofn.lpstrFile = filename;
	ofn.nMaxFile = MAX_PATH;
	ofn.Flags = OFN_NOCHANGEDIR | OFN_PATHMUSTEXIST;
	if (!GetSaveFileName(&ofn)) return false;
	return saveFile(filename);
}
bool readFileUserEntry() {
	OPENFILENAME ofn = { sizeof(OPENFILENAME) };
	WCHAR filename[MAX_PATH] = L"";
	ofn.lpstrFile = filename;
	ofn.nMaxFile = MAX_PATH;
	ofn.Flags = OFN_NOCHANGEDIR | OFN_PATHMUSTEXIST;
	if (!GetOpenFileName(&ofn)) return false;
	return readFile(filename);
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
		if (Unit * s > Max) s = Max / Unit; else if (Unit * s < Min) s = Min / Unit;
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

