// A binary STL 3D model viewer for Windows
// Some features are inspired by Blender 3D and ModuleWorks FreeSTLView

// Most STL viewers I found on the internet don't match my personal preference exactly
// So I developed one for myself

// Features:
// Powerful viewport navigations
// Multiple coloring and rendering modes
// Visualization of physical properties
// Edit and export
// Small and portable binary executable

// Flaws:
// Single-thread software rasterization that can be slow for models with >500k triangles
// Not support colored STL
// Code isn't very readable


// VIEWPORT / NAVIGATION:
// A viewport center highlighted in magenta is visible when dragging/holding control keys
// There is a small model of x,y,z axes (RGB) on the top-right of the window
// Click and drag to rotate the scene around viewport center (raw/patch)
// Scroll to zoom in/out
// Shift+Drag to adjust camera roll
// Shift+Scroll to adjust camera distance (perspective)
// Ctrl+Drag to move viewport center on xOy plane (may not work well when the camera is too close to the plane)
// Ctrl+Scroll to move viewport center along z-axis
// Alt+Click object to move viewport center to the clicked point on the object (recommend)
// Press Numpad decimal key to move viewport center to the center of the object (center of its bounding box)
// The function of a single press of Numpad keys is the same as that in Blender 2.80
//   - Numpad0: Move camera to negative y direction (look at positive y)
//   - Numpad3: Move camera to positive x direction (look at negative x)
//   - Numpad7: Move camera to positive z direction (look down, x-axis at right)
//   - Numpad9: Move camera to negative z direction (look up, x-axis at right)
//   - Numpad5: Dolly zoom, move camera extremely far away to simulate orthographic projection
//   - Numpad4: Rotate camera position horizontally around viewport center for 15 degrees (clockwise)
//   - Numpad6: Rotate camera position horizontally for 15 degrees (counterclockwise)
//   - Numpad8: Increase camera position vertical angle for 15 degrees
//   - Numpad2: Decrease camera position vertical angle for 15 degrees
//       (I may not use the right terminology)
// WSAD keys are supported but not recommended (may be used along with Alt+Click)
// Press Home key or Ctrl+0 to reset viewport to default
// To-do list:
//   - Rotation and zooming that changes the position of viewport center but not camera
//   - Moving camera along xOy plane
//   - Shortcuts for camera roll (Shift+Numpad4/Numpad6 in Blender)
//   - Dynamic translation and zooming based on the position of the camera and viewport center
//   - Arrow keys to go up/dow (help with WSAD)
//   - Free rotation when grid is hidden
//   - Numpad0 to move viewport center to origin
//   - A key to move viewport center to center of mass
//   - Optional: "crawling" on the surface

// VISUALIZATION OPTIONS:
// Press C to switch coloring mode (normal color vs. Phong)
//   - Normal color: the color of the triangles are based on their normals (independent to the viewport)
//   - Phong (default): the color of the triangles should look silver-white; if a triangle's normal faces backward from view, its color is dark brown
// Press Tab or Shift+Tab to switch to next/previous polygon rendering mode
//   - Default: fill the faces by color mentioned above
//   - Stroke: shaded faces with black strokes
//   - Polygon: no fill, stroke the triangle using filling color
//   - Point cloud: no fill or stroke, use filling color to plot vertices
// Press X to hide/show axis
// Press G to hide/show grid
// Press B to switch to/out dark background
// Press M to show/hide highlighting of the object's center of mass (orange)
// Press I to show/hide highlighting of the object's inertia tensor
//   - The inertia tensor (calculated at the center of mass) is visualized as three yellow principle axes with lengths equal to the principle radiuses
//   - If one or more calculations get NAN, there will be a dark green cross at the center of mass
// By default, the center of mass and inertia tensor calculator assumes the object is a solid. Press P to switch between solid mode and surface(shell) mode
// To-do list:
//   - Clamp the outline depth to eliminate black "dashes"
//   - Fix bug: top-right axis distortion under high perspective and zooming
//   - An option to always show viewport center
//   - Hide/unhide part of object
//   - Visualization of the objects' bounding box
//   - Visualization of the volume of the object
//   - Rendering with normal loaded from file
//   - Semi-transparent shading
//   - Outline rendering based on geometry
//   - Smoothed shading (interpolation)
//   - Optimization for software rasterization

// FILE AND EDITING:
// Press Ctrl+O to open Windows file explorer to browse and open a file
// Press Ctrl+S to save edited object
// Press F5 to reload object (there will not be warning about unsaved changes)
// Right-click window to set window to topmost or non-topmost
// Hold Alt key to modify object:
//   - Press Numpad . to place the object on the center of xOy plane
//   - Press Numpad 4/6/2/8 to rotate object left/right/up/down about its center of mass
//   - Press arrow keys to move object left/right/up/down
//   - Press Numpad5 to translate object so that its center of mass coincident with the origin
//   - Press plus/minus keys to scale the object about its center of mass
// To-do list:
//   - Shortcuts to rotate object to make its principle axes axis-oriented
//   - Nonlinear transforms
//   - Reflection
//   - Mouse-involved editings (eg. dragging, scrolling)
//   - Shift+F5 to reload without updating viewport
//   - Shortcut to view next/previous model in directory



#include <cmath>
#include <stdio.h>
#include <algorithm>
#pragma warning(disable: 4244)

// ========================================= Win32 Standard =========================================

#pragma region Windows

#ifndef UNICODE
#define UNICODE
#endif

#include <Windows.h>
#include <windowsx.h>
#include <tchar.h>


// debug
#define _USE_CONSOLE 0
wchar_t _DEBUG_OUTPUT_BUF[0x1000];
#define dbgprint(format, ...) { if (_USE_CONSOLE) {printf(format, ##__VA_ARGS__);} else {swprintf(_DEBUG_OUTPUT_BUF, 0x1000, _T(format), ##__VA_ARGS__); OutputDebugStringW(_DEBUG_OUTPUT_BUF);} }


#pragma region Window Macros / Forward Declarations

#define WIN_NAME "STL Viewer"
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


#pragma endregion  // WIN32

// ================================== Vector Classes/Functions ==================================

#pragma region Vector & Matrix

// https://github.com/Harry7557558/Graphics/tree/master/fitting/numerical
#include "numerical\geometry.h"  // vec2, vec3, mat3
#include "numerical\eigensystem.h"  // EigenPairs_Jacobi
typedef _geometry_triangle<vec3> triangle;

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

// intersection function
double intTriangle(vec3 v0, vec3 v1, vec3 v2, vec3 ro, vec3 rd) {
	vec3 v1v0 = v1 - v0, v2v0 = v2 - v0, rov0 = ro - v0;
	vec3 n = cross(v1v0, v2v0);
	vec3 q = cross(rov0, rd);
	double d = 1.0 / dot(rd, n);
	double u = d * dot(-q, v2v0); if (u<0. || u>1.) return NAN;
	double v = d * dot(q, v1v0); if (v<0. || (u + v)>1.) return NAN;
	return d * dot(-n, rov0);
}

#pragma endregion  // Vector & Matrix



// ======================================== Data / Parameters ========================================

#pragma region General Global Variables

// viewport
vec3 Center;  // view center in world coordinate
double rz, rx, ry, dist, Unit;  // yaw, pitch, roll, camera distance, scale to screen

// window parameters
Affine Tr;  // matrix
vec3 CamP, ScrO, ScrA, ScrB;  // camera and screen
auto scrDir = [](vec2 pixel) { return normalize(ScrO + (pixel.x / _WIN_W)*ScrA + (pixel.y / _WIN_H)*ScrB - CamP); };

// projection - matrix vs camera/screen
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
void getScreen(vec3 &P, vec3 &O, vec3 &A, vec3 &B) {  // O+uA+vB
	auto axisAngle = [](vec3 axis, double a)->Affine {
		axis = normalize(axis); double ct = cos(a), st = sin(a);
		return Affine{
			vec3(ct + axis.x*axis.x*(1 - ct), axis.x*axis.y*(1 - ct) - axis.z*st, axis.x*axis.z*(1 - ct) + axis.y*st),
			vec3(axis.y*axis.x*(1 - ct) + axis.z*st, ct + axis.y*axis.y*(1 - ct), axis.y*axis.z*(1 - ct) - axis.x*st),
			vec3(axis.z*axis.x*(1 - ct) - axis.y*st, axis.z*axis.y*(1 - ct) + axis.x*st, ct + axis.z*axis.z*(1 - ct)),
			vec3(0), vec3(0), 1.0
		};
	};
	double cx = cos(rx), sx = sin(rx), cz = cos(rz), sz = sin(rz);
	vec3 u(-sz, cz, 0), v(-cz * sx, -sz * sx, cx), w(cz * cx, sz * cx, sx);
	Affine Y = axisAngle(w, -ry); u = Y * u, v = Y * v;
	u *= 0.5*_WIN_W / Unit, v *= 0.5*_WIN_H / Unit, w *= dist;
	P = Center + w;
	O = Center - (u + v), A = u * 2.0, B = v * 2.0;
}

// user parameters
vec2 Cursor = vec2(0, 0), clickCursor;  // current cursor and cursor position when mouse down
bool mouse_down = false;
bool Ctrl = false, Shift = false, Alt = false;

// rendering parameters
bool ShadeByNormal = false;
int RenderMode = 0;
bool showAxis = true, showGrid = true;
bool showCOM = false, showInertia = false, TreatAsSolid = true;
bool DarkBackground = false;

#pragma endregion Camera/Screen, Mouse/Key


#pragma region STL file

WCHAR filename[MAX_PATH] = L"";
int N; triangle *T = 0;
vec3 BMin, BMax;  // bounding box
double V; vec3 COM;  // volume/area & center of mass
mat3 Inertia, Inertia_O, Inertia_D;  // inertia tensor calculated at center of mass, orthogonal and diagonal components

void calcBoundingBox() {
	BMin = vec3(INFINITY), BMax = -BMin;
	for (int i = 0; i < N; i++) {
		BMin = pMin(pMin(BMin, T[i].a), pMin(T[i].b, T[i].c));
		BMax = pMax(pMax(BMax, T[i].a), pMax(T[i].b, T[i].c));
	}
}
// physics, requires surface to be closed with outward normals
void calcVolumeCenterInertia() {
	// Assume the object to be uniform density
	// I might have a bug
	V = 0; COM = vec3(0.0); Inertia = mat3(0.0);
	for (int i = 0; i < N; i++) {
		vec3 a = T[i].a, b = T[i].b, c = T[i].c;
		if (TreatAsSolid) {
			double dV = det(a, b, c) / 6.;
			V += dV;
			COM += dV * (a + b + c) / 4.;
			Inertia += dV * 0.1*(mat3(dot(a, a) + dot(b, b) + dot(c, c) + dot(a, b) + dot(a, c) + dot(b, c)) -
				(tensor(a, a) + tensor(b, b) + tensor(c, c) + 0.5*(tensor(a, b) + tensor(a, c) + tensor(b, a) + tensor(b, c) + tensor(c, a) + tensor(c, b))));
		}
		else {
			double dV = 0.5*length(cross(b - a, c - a));
			V += dV;
			COM += dV * (a + b + c) / 3.;
			Inertia += dV / 6. *(mat3(dot(a, a) + dot(b, b) + dot(c, c) + dot(a, b) + dot(a, c) + dot(b, c)) -
				(tensor(a, a) + tensor(b, b) + tensor(c, c) + 0.5*(tensor(a, b) + tensor(a, c) + tensor(b, a) + tensor(b, c) + tensor(c, a) + tensor(c, b))));
		}
	}
	COM /= V;
	Inertia = Inertia - V * (mat3(dot(COM, COM)) - tensor(COM, COM));
	vec3 L;
	EigenPairs_Jacobi(3, &Inertia.v[0][0], (double*)&L, &Inertia_O.v[0][0]);
	Inertia_D = mat3(L);
}


#pragma endregion  STL and related calculations


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
void drawLine_F(vec3 A, vec3 B, COLORREF col = 0xFFFFFF) {
	double u = dot(Tr.p, A) + Tr.s, v = dot(Tr.p, B) + Tr.s;
	if (u > 0 && v > 0) { drawLine((Tr*A).xy(), (Tr*B).xy(), col); return; }
	if (u < 0 && v < 0) return;
	if (u < v) std::swap(A, B), std::swap(u, v);
	double t = u / (u - v) - 1e-6;
	B = A + (B - A)*t;
	drawLine((Tr*A).xy(), (Tr*B).xy(), col);
}
void drawCross3D(vec3 P, double r, COLORREF col = 0xFFFFFF) {
	r *= dot(Tr.p, P) + Tr.s;
	drawLine_F(P - vec3(r, 0, 0), P + vec3(r, 0, 0), col);
	drawLine_F(P - vec3(0, r, 0), P + vec3(0, r, 0), col);
	drawLine_F(P - vec3(0, 0, r), P + vec3(0, 0, r), col);
}


void drawLine_ZB(vec3 A, vec3 B, COLORREF col) {
	double u = dot(Tr.p, A) + Tr.s, v = dot(Tr.p, B) + Tr.s;
	if (u < 0 || v < 0) {
		if (u < 0 && v < 0) return;
		if (u < v) std::swap(A, B), std::swap(u, v);
		B = A + (B - A)* (u / (u - v) - 1e-6);
	}
	vec3 p = Tr * A, q = Tr * B;
	vec3 d = q - p;
	double slope = d.y / d.x, slopez;
	if (abs(slope) <= 1.0) {
		slopez = d.z / d.x;
		if (p.x > q.x) std::swap(p, q);
		int x0 = max(0, int(p.x)), x1 = min(_WIN_W - 1, int(q.x)), y;
		double yf = p.y + slope * (x0 - p.x), zf = p.z + slopez * (x0 - p.x) - 1e-4;
		for (int x = x0; x <= x1; x++) {
			y = (int)(yf + 0.5);
			if (y >= 0 && y < _WIN_H && zf < _DEPTHBUF[x][y]) Canvas(x, y) = col, _DEPTHBUF[x][y] = zf;
			yf += slope, zf += slopez;
		}
	}
	else {
		slope = d.x / d.y, slopez = d.z / d.y;
		if (p.y > q.y) std::swap(p, q);
		int y0 = max(0, int(p.y)), y1 = min(_WIN_H - 1, int(q.y)), x;
		double xf = p.x + slope * (y0 - p.y), zf = p.z + slopez * (y0 - p.y) - 1e-4;
		for (int y = y0; y <= y1; y++) {
			x = (int)(xf + 0.5);
			if (x >= 0 && x < _WIN_W && zf < _DEPTHBUF[x][y]) Canvas(x, y) = col, _DEPTHBUF[x][y] = zf;
			xf += slope, zf += slopez;
		}
	}
}
void drawTriangle_ZB(vec3 A, vec3 B, vec3 C, COLORREF fill, COLORREF stroke = -1, COLORREF point = -1) {
	A = Tr * A, B = Tr * B, C = Tr * C;
	if (isnan(A.x + B.x + C.x)) return;  // hmm...
	vec2 a = A.xy(), b = B.xy(), c = C.xy();
	vec3 ab = B - A, ac = C - A, n = cross(ab, ac);
	double k = 1.0 / det(ac.xy(), ab.xy());

	if (fill != -1) {
		int x0 = max((int)min(min(a.x, b.x), c.x), 0), x1 = min((int)max(max(a.x, b.x), c.x), _WIN_W - 1);
		int y0 = max((int)min(min(a.y, b.y), c.y), 0), y1 = min((int)max(max(a.y, b.y), c.y), _WIN_H - 1);
		for (int i = y0; i <= y1; i++) for (int j = x0; j <= x1; j++) {
			vec2 p(j, i);
			double z = k * dot(n, vec3(p) - A);
			if (z < _DEPTHBUF[j][i]) {
				vec2 ap = p - a, bp = p - b, cp = p - c;
				if (((det(ap, bp) < 0) + (det(bp, cp) < 0) + (det(cp, ap) < 0)) % 3 == 0) {
					Canvas(j, i) = fill, _DEPTHBUF[j][i] = z;
				}
			}
		}
	}

	if (stroke != -1) {
		// WHY I HAVE THE SAME CODE COPY-PASTED THREE TIMES??
		auto drawLine = [&](vec3 p, vec3 q) {
			vec3 d = q - p;
			double slope = d.y / d.x;
			if (abs(slope) <= 1.0) {
				if (p.x > q.x) std::swap(p, q);
				int x0 = max(0, int(p.x)), x1 = min(_WIN_W - 1, int(q.x)), y;
				double yf = p.y + slope * (x0 - p.x);
				for (int x = x0; x <= x1; x++, yf += slope) {
					y = (int)(yf + 0.5);
					if (y >= 0 && y < _WIN_H) {
						double z = k * dot(n, vec3(x, y, 0) - A) - 1e-6;
						if (z <= _DEPTHBUF[x][y]) Canvas(x, y) = stroke, _DEPTHBUF[x][y] = z;
					}
				}
			}
			else {
				slope = d.x / d.y;
				if (p.y > q.y) std::swap(p, q);
				int y0 = max(0, int(p.y)), y1 = min(_WIN_H - 1, int(q.y)), x;
				double xf = p.x + slope * (y0 - p.y);
				for (int y = y0; y <= y1; y++, xf += slope) {
					x = (int)(xf + 0.5);
					if (x >= 0 && x < _WIN_W) {
						double z = k * dot(n, vec3(x, y, 0) - A) - 1e-6;
						if (z <= _DEPTHBUF[x][y]) Canvas(x, y) = stroke, _DEPTHBUF[x][y] = z;
					}
				}
			}
		};
		drawLine(A, B); drawLine(A, C); drawLine(B, C);
	}

	if (point != -1) {
		auto drawDot = [&](vec3 p) {
			int x = int(p.x + .5), y = int(p.y + .5);
			if (x >= 0 && x < _WIN_W && y >= 0 && y < _WIN_H) {
				double z = k * dot(n, vec3(x, y, 0) - A);
				if (z < _DEPTHBUF[x][y]) Canvas(x, y) = point, _DEPTHBUF[x][y] = z;
			}
		};
		drawDot(A); drawDot(B); drawDot(C);
	}
}

#pragma endregion



#include <chrono>
std::chrono::steady_clock::time_point _iTimer = std::chrono::high_resolution_clock::now();

void render() {
	// initialize window
	if (DarkBackground) {
		for (int i = 0, l = _WIN_W * _WIN_H; i < l; i++) _WINIMG[i] = 0;
	}
	else for (int j = 0; j < _WIN_H; j++) {
		COLORREF c = toCOLORREF(mix(vec3(0.95), vec3(0.65, 0.65, 1.00), j / double(_WIN_H)));
		for (int i = 0; i < _WIN_W; i++) _WINIMG[j*_WIN_W + i] = c;
	}
	for (int i = 0; i < _WIN_W; i++) for (int j = 0; j < _WIN_H; j++) _DEPTHBUF[i][j] = INFINITY;
	calcMat();
	getScreen(CamP, ScrO, ScrA, ScrB);

	// axis and grid
	{
		const double R = 20.0;
		if (showAxis) {
			drawLine_ZB(vec3(0, -R, 0), vec3(0, R, 0), 0x409040);
			drawLine_ZB(vec3(-R, 0, 0), vec3(R, 0, 0), 0xC04040);
			drawLine_ZB(vec3(0, 0, -.6*R), vec3(0, 0, .6*R), 0x4040FF);
		}
		if (showGrid) {
			for (int i = -R; i <= R; i++) {
				COLORREF color = DarkBackground ? 0x404040 : 0xA0A0A0;
				drawLine_ZB(vec3(-R, i, 0), vec3(R, i, 0), color);
				drawLine_ZB(vec3(i, -R, 0), vec3(i, R, 0), color);
			}
		}
	}

	for (int i = 0; i < N; i++) {
		vec3 A = T[i].a, B = T[i].b, C = T[i].c;
		vec3 n = normalize(cross(B - A, C - A)), c;
		if (ShadeByNormal) {  // shade by normal
			c = 0.5*(n + vec3(1.0));
		}
		else {  // phong, obvious backward normal (darkred/black)
			vec3 light = normalize(CamP - Center);
			double k = clamp(dot(n, light), 0., 1.);
			vec3 d = normalize((A + B + C) / 3. - CamP);
			double s = dot(d - (2 * dot(d, n))*n, light);
			s = pow(max(s, 0), 20.0);
			c = vec3(0.1, 0.05, 0.05) + vec3(0.75, 0.75, 0.65)*k + vec3(0.15)*s;
		}
		int mode = RenderMode % 4; if (mode < 0) mode += 4;
		switch (mode) {
		case 0: drawTriangle_ZB(A, B, C, toCOLORREF(c)); break;
		case 1: drawTriangle_ZB(A, B, C, toCOLORREF(c), 0); break;
		case 2: drawTriangle_ZB(A, B, C, -1, toCOLORREF(c)); break;
		case 3: drawTriangle_ZB(A, B, C, -1, -1, toCOLORREF(c)); break;
		}
	}

	// highlight physical properties
	if (showCOM) drawCross3D(COM, 6, 0xFF8000);
	if (showInertia) {
		bool hasNAN = isnan(sumsqr(Inertia_O));
		for (int i = 0; i < 3; i++) {
			double r = sqrt(Inertia_D.v[i][i] / V);
			drawLine_F(COM, COM + Inertia_O.row(i)*r, 0xFFFF00);
			hasNAN |= isnan(r);
		}
		if (hasNAN) drawCross3D(COM, 6, 0x006400);
	}

	// highlight center
	if (Ctrl || Shift || Alt || mouse_down) drawCross3D(Center, 6, 0xFF00FF);

	// top-right coordinate
	vec2 xi = (Tr*vec3(Tr.s, 0, 0)).xy(), yi = (Tr*vec3(0, Tr.s, 0)).xy(), zi = (Tr*vec3(0, 0, Tr.s)).xy();
	vec2 ci = (Tr*vec3(0)).xy();
	double pd = 0.1 * sqrt(_WIN_W*_WIN_H);
	vec2 C(_WIN_W - pd, _WIN_H - pd); pd *= 0.8;
	drawLine(C, C + pd * (xi - ci), 0xFF0000);
	drawLine(C, C + pd * (yi - ci), 0x008000);
	drawLine(C, C + pd * (zi - ci), 0x0000FF);

	// window title
	double time_elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - _iTimer).count();
	WCHAR text[1024]; wsprintf(text, L"%s [%d %s]   %dx%d %dfps", filename, N, TreatAsSolid ? L"solid" : L"shell", _WIN_W, _WIN_H, int(1.0 / time_elapsed));
	SetWindowText(_HWND, text);
	_iTimer = std::chrono::high_resolution_clock::now();
}


// ============================================== User ==============================================

bool readFile(const WCHAR* filename) {
	FILE *fp = _wfopen(filename, L"rb"); if (!fp) return false;
	char s[80]; if (fread(s, 1, 80, fp) != 80) return false;
	if (fread(&N, sizeof(int), 1, fp) != 1) return false;
	if (T) delete T;
	try { T = new triangle[N]; }
	catch (...) { T = 0; N = 0; return false; }
	for (int i = 0; i < N; i++) {
		float f[12];
		if (fread(f, sizeof(float), 12, fp) != 12) return false;
		double d[12]; for (int i = 0; i < 12; i++) d[i] = f[i];
		T[i] = triangle{ vec3(d[3], d[4], d[5]), vec3(d[6], d[7], d[8]), vec3(d[9], d[10], d[11]) };
		char c[2]; if (fread(c, 1, 2, fp) != 2) return false;
	}
	return true;
}
bool readFileUserEntry() {
	SetWindowPos(_HWND, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
	OPENFILENAME ofn = { sizeof(OPENFILENAME) };
	ofn.lpstrFile = filename;
	ofn.nMaxFile = MAX_PATH;
	ofn.Flags = OFN_NOCHANGEDIR | OFN_PATHMUSTEXIST;
	if (!GetOpenFileName(&ofn)) return false;
	return readFile(filename);
}
bool saveFile(const WCHAR* filename) {
	FILE *fp = _wfopen(filename, L"wb"); if (!fp) return false;
	for (int i = 0; i < 80; i++) fputc(0, fp);
	fwrite(&N, 4, 1, fp);
	for (int i = 0; i < N; i++) {
		auto writevec3 = [&](vec3 p) {
			float f = float(p.x); fwrite(&f, 4, 1, fp);
			f = float(p.y); fwrite(&f, 4, 1, fp);
			f = float(p.z); fwrite(&f, 4, 1, fp);
		};
		writevec3(normalize(cross(T[i].b - T[i].a, T[i].c - T[i].a)));
		writevec3(T[i].a); writevec3(T[i].b); writevec3(T[i].c);
		fputc(0, fp); fputc(0, fp);
	}
	return fclose(fp) == 0;
}
bool saveFileUserEntry() {
	SetWindowPos(_HWND, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
	OPENFILENAME ofn = { sizeof(OPENFILENAME) };
	WCHAR filename[MAX_PATH] = L"";
	ofn.lpstrFile = filename;
	ofn.nMaxFile = MAX_PATH;
	ofn.Flags = OFN_NOCHANGEDIR | OFN_PATHMUSTEXIST;
	if (!GetSaveFileName(&ofn)) return false;
	return saveFile(filename);
}


void setDefaultView() {
	Center = vec3(0.0, 0.0, 0.0);
	calcBoundingBox();
	calcVolumeCenterInertia();
	if (isnan(0.0*dot(BMax, BMin))) BMax = vec3(1.0), BMin = vec3(-1.0);
	if (isnan(V*Center.sqr())) Center = vec3(0.0);
	if (!(V != 0.0) || isnan(V)) V = 8.0;
	Center = 0.5*(BMax + BMin);
	vec3 Max = BMax - Center, Min = BMin - Center;
	double s = max(max(max(Max.x, Max.y), Max.z), -min(min(Min.x, Min.y), Min.z));
	s = sqrt(s*cbrt(abs(V)));
	if (_WIN_W*_WIN_H != 0.0) s /= sqrt(5e-6*_WIN_W*_WIN_H);
	rz = -1.1, rx = 0.25, ry = 0.0, dist = 12.0 * s, Unit = 100.0 / s;
}
void Init() {
	//readFileUserEntry();
	wcscpy(filename, L"Press Ctrl+O to open a file.  ");
	setDefaultView();
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
	/*else if (Alt) {
		double dist0 = dist; dist *= exp(-0.001*_DELTA);
		Center -= (dist0 - dist) * normalize(Center - CamP);
	}*/
	else {
		double s = exp(0.001*_DELTA);
		double D = length(vec2(_WIN_W, _WIN_H)), Max = 1000.0*D, Min = 0.001*D;
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

	if (Alt) {
		vec3 rd = ScrO + (Cursor.x / _WIN_W)*ScrA + (Cursor.y / _WIN_H)*ScrB - CamP;
		double mt = INFINITY;
		for (int i = 0; i < N; i++) {
			double t = intTriangle(T[i].a, T[i].b, T[i].c, CamP, rd);
			if (t > 0 && t < mt) {
				if (dot(Tr.p, T[i].a) + Tr.s > 0 && dot(Tr.p, T[i].b) + Tr.s > 0 && dot(Tr.p, T[i].c) + Tr.s > 0)
					mt = t;
			}
		}
		if (0.0*mt == 0.0) {
			Center = CamP + mt * rd;
			double s = length(Center - CamP) / dist;
			dist *= s, Unit /= s;
		}
	}
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
		setDefaultView();
	}

	if (_KEY == VK_F5) {  // reload file
		if (!readFile(filename)) {
			MessageBeep(MB_ICONSTOP);
			SetWindowText(_HWND, L"Error loading file");
		}
		setDefaultView();
	}

	if (Ctrl) {
		if (_KEY == 'O') {
			if (readFileUserEntry()) {
				calcBoundingBox(); calcVolumeCenterInertia();
				setDefaultView();
			}
			else {
				MessageBeep(MB_ICONSTOP);
				SetWindowText(_HWND, L"Error loading file");
			}
		}
		if (_KEY == 'S') {
			if (!saveFileUserEntry()) {
				MessageBeep(MB_ICONSTOP);
				SetWindowText(_HWND, L"Error saving file");
			}
		}
		Ctrl = false;
	}
	else {
		if (_KEY == 'C') ShadeByNormal ^= 1;
		if (_KEY == VK_TAB) {
			if (Shift) RenderMode--;
			else RenderMode++;
		}
		if (_KEY == 'X') showAxis ^= 1;
		if (_KEY == 'G') showGrid ^= 1;
		if (_KEY == 'M') calcVolumeCenterInertia(), showCOM ^= 1;
		if (_KEY == 'I') calcVolumeCenterInertia(), showInertia ^= 1;
		if (_KEY == 'B') DarkBackground ^= 1;
		if (_KEY == 'P') TreatAsSolid ^= 1, calcVolumeCenterInertia();
		if (_KEY == VK_DECIMAL && !Alt) calcBoundingBox(), Center = 0.5*(BMax + BMin);

		// not recommend
		if (_KEY == 'W') { vec3 d = 0.08*(Center - CamP); Center += d; }
		if (_KEY == 'S') { vec3 d = 0.08*(Center - CamP); Center -= d; }
		if (_KEY == 'A') { vec3 d = 0.05*dist * normalize(cross(veck, Center - CamP)); Center += d; }
		if (_KEY == 'D') { vec3 d = 0.05*dist * normalize(cross(veck, Center - CamP)); Center -= d; }

	}

	if (Alt) {
		// Shape modification
		if (_KEY == VK_NUMPAD4 || _KEY == VK_NUMPAD6 || _KEY == VK_NUMPAD2 || _KEY == VK_NUMPAD8) {
			mat3 M;
			if (_KEY == VK_NUMPAD4) M = axis_angle(veck, -.025*PI);
			if (_KEY == VK_NUMPAD6) M = axis_angle(veck, .025*PI);
			if (_KEY == VK_NUMPAD2) M = axis_angle(cross(veck, Center - CamP), -.025*PI);
			if (_KEY == VK_NUMPAD8) M = axis_angle(cross(veck, Center - CamP), .025*PI);
			for (int i = 0; i < N; i++) {
				T[i] = triangle{ M*(T[i].a - COM) + COM, M*(T[i].b - COM) + COM, M*(T[i].c - COM) + COM };
			}
			calcBoundingBox(); calcVolumeCenterInertia();
		}
		if (_KEY == VK_NUMPAD5) {
			for (int i = 0; i < N; i++) {
				T[i].a -= COM, T[i].b -= COM, T[i].c -= COM;
			}
			calcBoundingBox(); calcVolumeCenterInertia();
		}
		if (_KEY >= 0x25 && _KEY <= 0x28) {
			vec3 d;
			if (_KEY == VK_UP) d = veck;
			if (_KEY == VK_DOWN) d = -veck;
			if (_KEY == VK_LEFT) d = normalize(cross(veck, Center - CamP));
			if (_KEY == VK_RIGHT) d = -normalize(cross(veck, Center - CamP));
			d *= 0.06*cbrt(V);
			for (int i = 0; i < N; i++) {
				T[i].a += d, T[i].b += d, T[i].c += d;
			}
			calcBoundingBox(); calcVolumeCenterInertia();
		}
		if (_KEY == VK_OEM_PLUS || _KEY == VK_ADD || _KEY == VK_OEM_MINUS || _KEY == VK_SUBTRACT) {
			double s = (_KEY == VK_OEM_PLUS || _KEY == VK_ADD) ? exp(0.05) : exp(-0.05);
			for (int i = 0; i < N; i++) {
				T[i].a = (T[i].a - COM)*s + COM, T[i].b = (T[i].b - COM)*s + COM, T[i].c = (T[i].c - COM)*s + COM;
			}
			calcBoundingBox(); calcVolumeCenterInertia();
		}
		if (_KEY == VK_DECIMAL) {
			vec3 d = vec3(0.5*(BMin + BMax).xy(), BMin.z);
			for (int i = 0; i < N; i++) T[i].a -= d, T[i].b -= d, T[i].c -= d;
			calcBoundingBox(); calcVolumeCenterInertia();
		}
	}
	else {
		// Blender-style keys
		if (_KEY == VK_NUMPAD4) rz -= PI / 12.;
		if (_KEY == VK_NUMPAD6) rz += PI / 12.;
		if (_KEY == VK_NUMPAD2) rx -= PI / 12.;
		if (_KEY == VK_NUMPAD8) rx += PI / 12.;
		if (_KEY == VK_NUMPAD1) rx = 0, rz = -.5*PI;
		if (_KEY == VK_NUMPAD3) rx = 0, rz = 0;
		if (_KEY == VK_NUMPAD7) rx = .5*PI, rz = -.5*PI;
		if (_KEY == VK_NUMPAD9) rx = -.5*PI, rz = -.5*PI;
		if (_KEY == VK_NUMPAD5) {
			if (dist*Unit > 1e6) dist = 1200 / Unit;
			else dist = 1e8 / Unit;
		}
	}
}

