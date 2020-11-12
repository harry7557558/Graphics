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
// Windows only
// Lack the option of reading numerical values
// Code isn't very readable


// VIEWPORT / NAVIGATION:
// A viewport center highlighted in magenta is visible when dragging/holding control keys
// There is a small model of x,y,z axes (RGB) on the top-right of the window
// Click and drag to rotate the scene around the viewport center (raw/patch)
// Scroll to zoom in/out
// Shift+Drag to adjust camera roll
// Shift+Scroll to adjust camera distance (perspective)
// Ctrl+Drag to move viewport center on xOy plane (may not work well when the camera is too close to the plane)
// Ctrl+Scroll to move viewport center along z-axis
// Alt+Click object to move viewport center to the clicked point on the object (recommend)
// Press Numpad decimal key to move viewport center to the center of the object (center of its bounding box)
// Press Numpad0 to move viewport center to the coordinate origin
// The function of a single press of Numpad keys is the same as that in Blender 2.80
//   - Numpad1: Move camera to negative y-direction (look at positive y)
//   - Numpad3: Move camera to positive x-direction (look at negative x)
//   - Numpad7: Move camera to positive z-direction (look down, x-axis at right)
//   - Numpad9: Move camera to negative z-direction (look up, x-axis at right)
//   - Numpad5: Switch between orthographic and perspective projections
//   - Numpad4: Rotate camera position horizontally around viewport center for 15 degrees (clockwise)
//   - Numpad6: Rotate camera position horizontally by 15 degrees (counterclockwise)
//   - Numpad8: Increase camera position vertical angle by 15 degrees
//   - Numpad2: Decrease camera position vertical angle by 15 degrees
// WSAD keys are supported but not recommended (may be used along with Alt+Click)
// Press Home key or Ctrl+0 to reset viewport to default
// To-do list:
//   - Rotation and zooming that changes the position of the viewport center but not the camera (Shift+Numpad)
//   - Moving camera along xOy plane but not changing the viewport center (Ctrl+Shift+Drag)
//   - Shortcuts for camera roll (Shift+Numpad4/Numpad6 in Blender)
//   - Dynamic translation and zooming based on the position of the camera and viewport center
//   - Arrow keys to go up/down (help with WSAD)
//   - Free rotation when grid is hidden
//   - A key to move viewport center to center of mass
//   - Optional: "crawling" on the surface

// VISUALIZATION OPTIONS:
// Press C or Shift+C to switch to next/previous coloring mode
//   - Default: silver-white Phong shading; if a triangle's normal faces backward from view, its color is dark brown
//   - Normal color: the color of the triangles are based on their normals (independent to the viewport)
//   - Heatmap: the color of the triangles only depend on the z-position of the triangle's center, useful for visualization of math functions
//   - From file: see https://en.wikipedia.org/wiki/STL_(file_format)#Color_in_binary_STL, VisCAM and SolidView version; white if invalid
//   - File+Phong: combine color loaded from file and Phong shading
// Press Tab or Shift+Tab to switch to next/previous polygon rendering mode
//   - Default: fill the faces by color mentioned above
//   - Stroke: shaded faces with black strokes
//   - Polygon: no fill, stroke the triangle using filling color
//   - Point cloud: no fill or stroke, use filling color to plot vertices
// Press (Shift+)N to switch normal calculation mode (for rendering only)
//   - [0] n=[from file] [1] n=AB×AC [2] n=BCxBA [3] n=ACxAB
// Press X to hide/show axis
// Press G to hide/show grid ([-10,10] with grid width 1)
//   - By default there are larger and smaller grid lines if the object is too large or small; Press F or Shift+G to turn on/off this feature
//   - Press H to show/hide gridlines on xOz and yOz planes
// Right-click the shape to read the coordinates of the clicked point from windows title; right click empty space or press Esc to return to normal
// Press B to switch to/out dark background
// Press M to show/hide highlighting of the object's center of mass (orange)
// Press I to show/hide highlighting of the object's inertia tensor
//   - The inertia tensor (calculated at the center of mass) is visualized as three yellow principle axes with lengths equal to the principle radiuses
//   - If one or more calculations get NAN, there will be a dark green cross at the center of mass
// By default, the center of mass and inertia tensor calculator (based on divergence theorem) assumes the object is a solid. Press P to switch between solid mode and surface(shell) mode
// To-do list:
//   - [not trivial] Fix the stroking issue
//   - Fix bug: top-right axis distortion under high perspective and zooming
//   - An option to always show viewport center
//   - Math visualization:
//     - show grid on other planes
//     - setAxisRatio
//     - A (mandatory) clipping box
//   - Pop out dialog box to show numerical informations of the object
//   - Hide/unhide part of object
//   - Visualization of the objects' bounding box/sphere
//   - Visualization of the volume of the object
//   - Semi-transparent shading
//   - Outline rendering based on geometry
//   - Smoothed shading (interpolation)
//   - Optimization for software rasterization
//   - Option to show xOy plane with rendering for shadow

// FILE AND EDITING:
// Press Ctrl+O to open Windows file explorer to browse and open a file
// Press Ctrl+S to save an edited object
// Press F5 to reload object from file, Shift+F5 or F9 to reload without resetting the current viewport
// Press F1 to set window to topmost or non-topmost
// Hold Alt key to modify object:
//   - Press Numpad . to place the object on the center of xOy plane
//   - Press Numpad 4/6/2/8/7/9 to rotate object left/right/up/down/counter/clockwise about its center
//   - Press arrow keys to move object left/right/front/back (in screen coordinate)
//   - Press Numpad5 to translate object so that its center coincident with the origin
//   - Press plus/minus keys or scroll mouse wheel to scale the object about its center
//   - Hold Shift and press plus/minus keys (or scroll mouse wheel) to scale the object along the z-axis, about the xOy plane
// To-do list:
//   - Report error if the file contains NAN/INF values
//   - Alt+Numpad0 to move the object to the first quadrant (x-y only)
//   - Shortcuts to rotate the object to make its principal axes axis-oriented
//   - Nonlinear transforms
//   - Reflection
//   - Mouse-involved editings (eg. dragging, scrolling)
//   - Shortcut to view next/previous model in directory
//   - Support for non-standard animated STL (keyframes for position/orientation)
//   - Recording GIF



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

void Init();  // called before the window is created
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

#include "numerical\geometry.h"  // vec2, vec3, mat3
#include "numerical\eigensystem.h"  // EigenPairs_Jacobi
typedef struct { vec3 n, a, b, c; } stl_triangle;
typedef int16_t stl_color;

// covert stl_color to vec3
// https://en.wikipedia.org/wiki/STL_(file_format)#Color_in_binary_STL
vec3 stlColor2vec3(stl_color c) {
	if (c >= 0) return vec3(1.);  // otherwise, max 31/32=0.97
	double r = ((uint32_t)c & (uint32_t)0b111110000000000) >> 10;
	double g = ((uint32_t)c & (uint32_t)0b000001111100000) >> 5;
	double b = ((uint32_t)c & (uint32_t)0b000000000011111);
	return vec3(r, g, b) * (1. / 32.);
}


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
bool use_orthographic = false;

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
	Affine P{ veci, vecj, veck, vec3(0), vec3(0, 0, use_orthographic ? 0. : (1.0 / dist)), 1.0 };  // perspective
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
	u *= 0.5*_WIN_W / Unit, v *= 0.5*_WIN_H / Unit, w *= use_orthographic ? 1e8 : dist;
	P = Center + w;
	O = Center - (u + v), A = u * 2.0, B = v * 2.0;
}

// user parameters
vec2 Cursor = vec2(0, 0), clickCursor;  // current cursor and cursor position when mouse down
bool mouse_down = false;
bool Ctrl = false, Shift = false, Alt = false;
vec3 readedValue = vec3(NAN);  // for reading values

// rendering parameters
int ShadeMode = 0;
int RenderMode = 0;
int NormalMode = 0;
bool showAxis = true, showGrid = true, showMajorMinorGrids = true, showVerticalGrids = false;
bool showCOM = false, showInertia = false, TreatAsSolid = true;
bool DarkBackground = false;

#pragma endregion Camera/Screen, Mouse/Key


#pragma region STL file

WCHAR filename[MAX_PATH] = L"";
int N; stl_triangle *T = 0;  // triangles
vec3 *T_Color = 0; bool hasInvalidColor = false;  // color
vec3 BMin, BMax;  // bounding box
double Volume; vec3 COM;  // volume/area & center of mass
mat3 Inertia, Inertia_O, Inertia_D;  // inertia tensor calculated at center of mass, orthogonal and diagonal components

void calcBoundingBox() {
	BMin = vec3(INFINITY), BMax = -BMin;
	for (int i = 0; i < N; i++) {
		BMin = pMin(pMin(BMin, T[i].a), pMin(T[i].b, T[i].c));
		BMax = pMax(pMax(BMax, T[i].a), pMax(T[i].b, T[i].c));
	}
}
// physics, requires the surface to be closed with outward normals
void calcVolumeCenterInertia() {
	// Assume the object to be uniform density
	// I might have a bug
	Volume = 0; COM = vec3(0.0); Inertia = mat3(0.0);
	for (int i = 0; i < N; i++) {
		vec3 a = T[i].a, b = T[i].b, c = T[i].c;
		if (TreatAsSolid) {
			double dV = det(a, b, c) / 6.;
			Volume += dV;
			COM += dV * (a + b + c) / 4.;
			Inertia += dV * 0.1*(mat3(dot(a, a) + dot(b, b) + dot(c, c) + dot(a, b) + dot(a, c) + dot(b, c)) -
				(tensor(a, a) + tensor(b, b) + tensor(c, c) + 0.5*(tensor(a, b) + tensor(a, c) + tensor(b, a) + tensor(b, c) + tensor(c, a) + tensor(c, b))));
		}
		else {
			double dV = 0.5*length(cross(b - a, c - a));
			Volume += dV;
			COM += dV * (a + b + c) / 3.;
			Inertia += dV / 6. *(mat3(dot(a, a) + dot(b, b) + dot(c, c) + dot(a, b) + dot(a, c) + dot(b, c)) -
				(tensor(a, a) + tensor(b, b) + tensor(c, c) + 0.5*(tensor(a, b) + tensor(a, c) + tensor(b, a) + tensor(b, c) + tensor(c, a) + tensor(c, b))));
		}
	}
	COM /= Volume;
	Inertia = Inertia - Volume * (mat3(dot(COM, COM)) - tensor(COM, COM));
	vec3 L;
	EigenPairs_Jacobi(3, &Inertia.v[0][0], (double*)&L, &Inertia_O.v[0][0]);
	Inertia_D = mat3(L);
}

// ray intersection with shape, use for selection
// return the distance
double rayIntersect(vec3 ro, vec3 rd) {
	double mt = INFINITY;
	for (int i = 0; i < N; i++) {
		double t = intTriangle(T[i].a, T[i].b, T[i].c, ro, rd);
		if (t > 0 && t < mt) {
			if (use_orthographic ||
				(dot(Tr.p, T[i].a) + Tr.s > 0 && dot(Tr.p, T[i].b) + Tr.s > 0 && dot(Tr.p, T[i].c) + Tr.s > 0))
				mt = t;
		}
	}
	return mt;
}


#pragma endregion  STL and related calculations


// ============================================ Rendering ============================================

#pragma region Rasterization functions

#include "ui/colors/ColorFunctions.h"

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
void drawCircle(vec2 c, double r, COLORREF Color) {
	int s = int(r / sqrt(2) + 0.5);
	int cx = (int)c.x, cy = (int)c.y;
	for (int i = 0, im = min(s, max(_WIN_W - cx, cx)) + 1; i < im; i++) {
		int u = sqrt(r*r - i * i) + 0.5;
		setColor(cx + i, cy + u, Color); setColor(cx + i, cy - u, Color); setColor(cx - i, cy + u, Color); setColor(cx - i, cy - u, Color);
		setColor(cx + u, cy + i, Color); setColor(cx + u, cy - i, Color); setColor(cx - u, cy + i, Color); setColor(cx - u, cy - i, Color);
	}
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
			if (y >= 0 && y < _WIN_H && zf < _DEPTHBUF[x][y] + 1e-6) Canvas(x, y) = col, _DEPTHBUF[x][y] = zf;
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
			if (x >= 0 && x < _WIN_W && zf < _DEPTHBUF[x][y] + 1e-6) Canvas(x, y) = col, _DEPTHBUF[x][y] = zf;
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
					Canvas(j, i) = fill;
					_DEPTHBUF[j][i] = z;
				}
			}
		}
	}

	if (stroke != -1) {
		/*double z_min = std::min({
			k * dot(n, vec3(A.xy(), 0) - A),
			k * dot(n, vec3(B.xy(), 0) - A),
			k * dot(n, vec3(C.xy(), 0) - A) });*/
			// WHY I HAVE THE SAME CODE COPY-PASTED THREE TIMES??
		auto drawLine = [&](vec3 p, vec3 q) {
			vec3 d = q - p; double d2 = d.xy().sqr();
			double slope = d.y / d.x;
			if (abs(slope) <= 1.0) {
				if (p.x > q.x) std::swap(p, q);
				int x0 = max(0, int(p.x)), x1 = min(_WIN_W - 1, int(q.x)), y;
				double yf = p.y + slope * (x0 - p.x);
				for (int x = x0; x <= x1; x++, yf += slope) {
					y = (int)(yf + 0.5);
					if (y >= 0 && y < _WIN_H) {
						if (0) {  // an unsuccessful attempt to fix the stroking bug
							vec2 p(x, y);
							vec2 ap = p - a, bp = p - b, cp = p - c;
							if (((det(ap, bp) < 0) + (det(bp, cp) < 0) + (det(cp, ap) < 0)) % 3 == 0);
							else continue;
						}
						double z = k * dot(n, vec3(x, y, 0) - A) - 1e-6;
						if (z <= _DEPTHBUF[x][y]) {
							Canvas(x, y) = stroke;
							_DEPTHBUF[x][y] = z;
						}
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
						// the stroking bug isn't trivial to fix (at least for me)
						// (x,y) isn't always inside the triangle
						if (0) {
							vec2 p(x, y);
							vec2 ap = p - a, bp = p - b, cp = p - c;
							if (((det(ap, bp) < 0) + (det(bp, cp) < 0) + (det(cp, ap) < 0)) % 3 == 0);
							else continue;
						}
#if 0
						double h = dot(vec2(x, y) - p.xy(), d.xy()) / d2;
						double eps = k * dot(n, vec3(d.xy()*dot(vec2(slope, 1), d.xy()) / d2, 0));
						if (eps < 0.) eps *= -1;
						vec2 c = p.xy() + d.xy()*h;
						double z1 = k * dot(n, vec3(c, 0) - A) - eps;
						//c = vec2(x, y);
						double z = k * dot(n, vec3(x, y, 0) - A) - 1e-6;
						//z = max(z, z_min);
						if (z1 <= _DEPTHBUF[x][y]) {
							Canvas(x, y) = stroke;
							_DEPTHBUF[x][y] = z;
						}
#else
						double z = k * dot(n, vec3(x, y, 0) - A) - 1e-6;
						if (z <= _DEPTHBUF[x][y]) {
							Canvas(x, y) = stroke;
							_DEPTHBUF[x][y] = z;
						}
#endif
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
	else {
		for (int j = 0; j < _WIN_H; j++) {
			COLORREF c = toCOLORREF(mix(vec3(0.95), vec3(0.65, 0.65, 1.00), j / double(_WIN_H)));
			for (int i = 0; i < _WIN_W; i++) _WINIMG[j*_WIN_W + i] = c;
		}
	}
	for (int i = 0; i < _WIN_W; i++) for (int j = 0; j < _WIN_H; j++) _DEPTHBUF[i][j] = INFINITY;
	calcMat();
	getScreen(CamP, ScrO, ScrA, ScrB);

	vec3 light = normalize(CamP - Center);  // pre-compute for Phong

	// shape
	for (int i = 0; i < N; i++) {
		vec3 A = T[i].a, B = T[i].b, C = T[i].c;
		vec3 n = NormalMode == 0 ? T[i].n :
			NormalMode == 1 ? ncross(B - A, C - A) :
			NormalMode == 2 ? ncross(C - B, A - B) :
			NormalMode == 3 ? ncross(C - A, B - A) : vec3(0.);
		vec3 c = T_Color[i];  // default: from file
		switch (ShadeMode) {
		case 0: {  // phong
			double k = clamp(dot(n, light), 0., 1.);
			vec3 d = normalize((A + B + C) / 3. - CamP);
			double spc = dot(d - (2 * dot(d, n))*n, light);
			spc = pow(max(spc, 0.), 20.0);
			c = vec3(0.1, 0.05, 0.05) + vec3(0.75, 0.75, 0.65)*k + vec3(0.15)*spc;
			break;
		}
		case 1: {  // shade by normal
			c = 0.5*(n + vec3(1.0));
			break;
		}
		case 2: {  // height map, for visualization
			double t = ((A.z + B.z + C.z) / 3. - BMin.z) / (BMax.z - BMin.z);
			if (!isnan(t)) c = ColorFunctions::Rainbow(t);
			else c = 0.5*(n + vec3(1.0));  // otherwise: shade by normal
			break;
		}
		case 3: {  // directly from file
			break;  // nothing needs to be done here
		}
		case 4: {  // file + Phong
			double k = min(abs(dot(n, light)), 1.);
			vec3 d = normalize((A + B + C) / 3. - CamP);
			double spc = dot(d - (2 * dot(d, n))*n, light);
			spc = pow(max(spc, 0.), 20.0);
			c *= 0.2 + 0.8*k + 0.2*spc;
			break;
		}
		}
		switch ((RenderMode % 4 + 4) % 4) {
		case 0: drawTriangle_ZB(A, B, C, toCOLORREF(c)); break;
		case 1: drawTriangle_ZB(A, B, C, toCOLORREF(c), 0); break;
		case 2: drawTriangle_ZB(A, B, C, -1, toCOLORREF(c)); break;
		case 3: drawTriangle_ZB(A, B, C, -1, -1, toCOLORREF(c)); break;
		}
	}

	// grid and axes
	{
		if (showGrid) {
			auto drawGridScale = [](double r, double d, COLORREF color) {
				drawLine_ZB(vec3(-r, d, 0), vec3(r, d, 0), color);
				drawLine_ZB(vec3(d, -r, 0), vec3(d, r, 0), color);
				if (showVerticalGrids) {
					drawLine_ZB(vec3(-r, 0, d), vec3(r, 0, d), color);
					drawLine_ZB(vec3(d, 0, -r), vec3(d, 0, r), color);
					drawLine_ZB(vec3(0, -r, d), vec3(0, r, d), color);
					drawLine_ZB(vec3(0, d, -r), vec3(0, d, r), color);
				}
			};
			if (showMajorMinorGrids) {
				double range = N > 0 ? std::max({ BMax.x, BMax.y, -BMin.x, -BMin.y }) : NAN;
				if (range > 4.) {  // large grid
					for (int i = -10; i <= 10; i++)
						drawGridScale(100, 10 * i, DarkBackground ? 0x202020 : 0xB8B8B8);
				}
				if (range < 2.) {  // small grid
					for (int i = -10; i <= 10; i++)
						drawGridScale(1, 0.1*i, DarkBackground ? 0x202020 : 0xB8B8B8);
				}
			}
			for (int i = -10; i <= 10; i++)  // standard grid
				drawGridScale(10, i, DarkBackground ? 0x404040 : 0xA0A0A0);
		}
		if (showAxis) {
			const double R = 20.0;
			drawLine_ZB(vec3(0, -R, 0), vec3(0, R, 0), 0x409040);
			drawLine_ZB(vec3(-R, 0, 0), vec3(R, 0, 0), 0xC04040);
			drawLine_ZB(vec3(0, 0, -.6*R), vec3(0, 0, .6*R), 0x4040FF);
		}
	}

	// highlight physical properties
	bool hasNAN = false;
	if (showInertia) {
		hasNAN = isnan(sumsqr(Inertia_O));
		for (int i = 0; i < 3; i++) {
			double r = sqrt(Inertia_D.v[i][i] / Volume);
			drawLine_F(COM, COM + Inertia_O.row(i)*r, 0xFFFF00);
			hasNAN |= isnan(r);
		}
	}
	if (showCOM) drawCross3D(COM, 6, 0xFF8000);
	if (hasNAN) drawCross3D(COM, 6, 0x006400);

	// highlight center
	if (Ctrl || Shift || Alt || mouse_down) drawCross3D(Center, 6, 0xFF00FF);

	// render point of the read value
	if (!isnan(readedValue.sqr())) {
		// draw a circle
		drawCircle((Tr*readedValue).xy(), 3, 0xFFFF00);
		// draw a colored cross
		double r = 6 * (dot(Tr.p, readedValue) + Tr.s);
		drawLine_F(readedValue - vec3(r, 0, 0), readedValue + vec3(r, 0, 0), 0xFF0000);
		drawLine_F(readedValue - vec3(0, r, 0), readedValue + vec3(0, r, 0), 0x008000);
		drawLine_F(readedValue - vec3(0, 0, r), readedValue + vec3(0, 0, r), 0x0000FF);
	}

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
	WCHAR text[1024];
	if ((ShadeMode == 3 || ShadeMode == 4) && hasInvalidColor) {
		swprintf(text, L"%s  - File contains invalid color information.", filename);
	}
	else if (isnan(readedValue.sqr())) {  // information
		int lastslash = 0;
		if (wcslen(filename) > 24)
			for (int i = 0; filename[i]; i++)
				if (filename[i] == '/' || filename[i] == '\\') lastslash = i + 1;
		swprintf(text, L"%s [%d %s]  normal=%s   %dx%d %dfps",
			&filename[lastslash],
			N, TreatAsSolid ? L"solid" : L"shell",
			NormalMode == 0 ? L"[file]" : NormalMode == 1 ? L"AB×AC" : NormalMode == 2 ? L"BC×BA" : NormalMode == 3 ? L"AC×AB" : L"ERROR!",
			_WIN_W, _WIN_H, int(1.0 / time_elapsed));
	}
	else {  // read value
		swprintf(text, L" (%lg, %lg, %lg)", readedValue.x, readedValue.y, readedValue.z);
	}
	SetWindowText(_HWND, text);
	_iTimer = std::chrono::high_resolution_clock::now();
}


// ============================================== User ==============================================

bool readFile(const WCHAR* filename) {
	FILE *fp = _wfopen(filename, L"rb"); if (!fp) return false;
	char s[80]; if (fread(s, 1, 80, fp) != 80) return false;
	if (fread(&N, sizeof(int), 1, fp) != 1) return false;
	try {
		if (T) delete T; if (T_Color) delete T_Color;
		T = new stl_triangle[N];
		T_Color = new vec3[N];
	}
	catch (...) { T = 0; N = 0; return false; }
	hasInvalidColor = false;
	for (int i = 0; i < N; i++) {
		float f[12];
		if (fread(f, sizeof(float), 12, fp) != 12) return false;
		double d[12]; for (int i = 0; i < 12; i++) d[i] = f[i];
		T[i] = stl_triangle{ vec3(d[0], d[1], d[2]), vec3(d[3], d[4], d[5]), vec3(d[6], d[7], d[8]), vec3(d[9], d[10], d[11]) };
		stl_color col; if (fread(&col, 2, 1, fp) != 1) return false;
		hasInvalidColor |= col >= 0;
		T_Color[i] = stlColor2vec3(col);
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
		vec3 A = T[i].a, B = T[i].b, C = T[i].c;
		writevec3(
			NormalMode == 0 ? T[i].n :
			NormalMode == 1 ? ncross(B - A, C - A) :
			NormalMode == 2 ? ncross(C - B, A - B) :
			NormalMode == 3 ? ncross(C - A, B - A) : vec3(0.)
		);
		writevec3(A); writevec3(B); writevec3(C);
		fputc(0, fp); fputc(0, fp);
	}
	return fclose(fp) == 0;
}
bool saveFileUserEntry() {
	SetWindowPos(_HWND, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);  // ??
	OPENFILENAME ofn = { sizeof(OPENFILENAME) };
	WCHAR filename_t[MAX_PATH] = L"";
	ofn.lpstrFile = filename_t;
	ofn.nMaxFile = MAX_PATH;
	ofn.Flags = OFN_NOCHANGEDIR | OFN_PATHMUSTEXIST;
	if (!GetSaveFileName(&ofn)) return false;
	if (saveFile(filename_t)) {
		wcscpy(filename, filename_t);
		return true;
	}
	else return false;
}


void setDefaultView() {
	Center = vec3(0.0, 0.0, 0.0);
	calcBoundingBox();
	calcVolumeCenterInertia();
	if (isnan(0.0*dot(BMax, BMin))) BMax = vec3(1.0), BMin = vec3(-1.0);
	if (isnan(Volume*Center.sqr())) Center = vec3(0.0);
	if (!(Volume != 0.0) || isnan(Volume)) Volume = 8.0;
	Center = 0.5*(BMax + BMin);
	vec3 Max = BMax - Center, Min = BMin - Center;
	double s = max(max(max(Max.x, Max.y), Max.z), -min(min(Min.x, Min.y), Min.z));
	s = 0.2*dot(Max - Min, vec3(1.));
	if (_WIN_W*_WIN_H != 0.0) s /= sqrt(5e-6*_WIN_W*_WIN_H);
	rz = -.25*PI, rx = 0.25, ry = 0.0, dist = 12.0 * s, Unit = 100.0 / s;
	use_orthographic = false;
	showAxis = true, showGrid = true, showMajorMinorGrids = true, showVerticalGrids = false;
	readedValue = vec3(NAN);
	// do not reset color/rendering mode
#if 0
	if (hasInvalidColor) {
		if (ShadeMode == 3 || ShadeMode == 4) ShadeMode = 0;
}
#endif
}
void Init() {
	//wcscpy(filename, L"D:\\.stl"); readFile(filename);
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
	else if (Alt) {  // modify object
		double sc = exp(-0.001*_DELTA);
		vec3 C = 0.5*(BMin + BMax);
		if (Shift) {
			for (int i = 0; i < N; i++)
				T[i].a.z *= sc, T[i].b.z *= sc, T[i].c.z *= sc;
		}
		else {
			for (int i = 0; i < N; i++)
				T[i].a = (T[i].a - C)*sc + C, T[i].b = (T[i].b - C)*sc + C, T[i].c = (T[i].c - C)*sc + C;
		}
		calcBoundingBox(); calcVolumeCenterInertia();
	}
	else if (Shift) {
		if (!use_orthographic) dist *= exp(-0.001*_DELTA);
	}
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

	// drag to rotate the scene
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
		vec3 rd = scrDir(Cursor);
		double t = rayIntersect(CamP, rd);
		if (t < 1e12) {
			Center = CamP + t * rd;
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

	// read values from the (graph?!)
	{
		vec3 rd = scrDir(Cursor);
		double t = rayIntersect(CamP, rd);
		readedValue = t < 1e12 ? CamP + rd * t : vec3(NAN);
	}

	Render_Needed = true;
}
void KeyDown(WPARAM _KEY) {
	if (_KEY == VK_CONTROL) Render_Needed = !Ctrl, Ctrl = true;
	else if (_KEY == VK_SHIFT) Render_Needed = !Shift, Shift = true;
	else if (_KEY == VK_MENU) Render_Needed = !Alt, Alt = true;
}
void KeyUp(WPARAM _KEY) {

	if (_KEY == VK_HOME || (Ctrl && (_KEY == '0' || _KEY == VK_NUMPAD0))) {
		setDefaultView();
	}
	if (_KEY == VK_ESCAPE) {
		readedValue = vec3(NAN);
	}

	if (_KEY == VK_F5 || _KEY == VK_F9) {  // reload file
		if (!readFile(filename)) {
			MessageBeep(MB_ICONSTOP);
			SetWindowText(_HWND, L"Error loading file");
		}
		if (!(_KEY == VK_F9 || Shift)) setDefaultView();
		else {
			calcBoundingBox();
			calcVolumeCenterInertia();
			readedValue = vec3(NAN);
		}
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
		if (_KEY == 'C') ShadeMode = (ShadeMode + (Shift ? 4 : 1)) % 5;
		if (_KEY == VK_TAB) RenderMode += Shift ? -1 : 1;
		if (_KEY == 'N') NormalMode = (NormalMode + (Shift ? 3 : 1)) % 4;
		if (_KEY == 'X') showAxis ^= 1;
		if (_KEY == 'G') showGrid ^= !Shift, showMajorMinorGrids ^= Shift;
		if (_KEY == 'F') showMajorMinorGrids ^= 1;
		if (_KEY == 'H') showVerticalGrids ^= 1;
		if (_KEY == 'M') calcVolumeCenterInertia(), showCOM ^= 1;
		if (_KEY == 'I') calcVolumeCenterInertia(), showInertia ^= 1;
		if (_KEY == 'B') DarkBackground ^= 1;
		if (_KEY == 'P') TreatAsSolid ^= 1, calcVolumeCenterInertia();
		if (_KEY == VK_DECIMAL && !Alt) calcBoundingBox(), Center = 0.5*(BMax + BMin);

		// WSAD, not recommended
		if (dist*Unit < 1e5) {
			vec3 scrdir = normalize(Center - CamP);
			double sc = 0.08*dot(BMax - BMin, vec3(1.));
			if (_KEY == 'W') { vec3 d = sc * scrdir; Center += d; }
			if (_KEY == 'S') { vec3 d = sc * scrdir; Center -= d; }
			if (_KEY == 'A') { vec3 d = sc * normalize(cross(veck, scrdir)); Center += d; }
			if (_KEY == 'D') { vec3 d = sc * normalize(cross(veck, scrdir)); Center -= d; }
		}

		// set window topmost
		if (_KEY == VK_F1) {
			bool topmost = GetWindowLong(_HWND, GWL_EXSTYLE) & WS_EX_TOPMOST;
			SetWindowPos(_HWND, topmost ? HWND_NOTOPMOST : HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
		}
	}

	if (Alt) {
		// Shape modification
		vec3 Center = 0.5*(BMin + BMax);
		if (_KEY == VK_NUMPAD4 || _KEY == VK_NUMPAD6 || _KEY == VK_NUMPAD2 || _KEY == VK_NUMPAD8 || _KEY == VK_NUMPAD7 || _KEY == VK_NUMPAD9) {
			mat3 M;
			if (_KEY == VK_NUMPAD4) M = axis_angle(veck, -.025*PI);
			if (_KEY == VK_NUMPAD6) M = axis_angle(veck, .025*PI);
			if (_KEY == VK_NUMPAD2) M = axis_angle(ScrA, .025*PI);
			if (_KEY == VK_NUMPAD8) M = axis_angle(ScrA, -.025*PI);
			if (_KEY == VK_NUMPAD7) M = axis_angle(Center - CamP, -.025*PI);
			if (_KEY == VK_NUMPAD9) M = axis_angle(Center - CamP, .025*PI);
			for (int i = 0; i < N; i++) {
				T[i] = stl_triangle{ M*T[i].n, M*(T[i].a - Center) + Center, M*(T[i].b - Center) + Center, M*(T[i].c - Center) + Center };
			}
			calcBoundingBox(); calcVolumeCenterInertia();
		}
		if (_KEY == VK_NUMPAD5) {
			for (int i = 0; i < N; i++) {
				T[i].a -= Center, T[i].b -= Center, T[i].c -= Center;
			}
			calcBoundingBox(); calcVolumeCenterInertia();
		}
		if (_KEY >= 0x25 && _KEY <= 0x28) {
			vec3 d;
			if (_KEY == VK_UP) d = normalize(ScrB);
			if (_KEY == VK_DOWN) d = normalize(-ScrB);
			if (_KEY == VK_LEFT) d = normalize(-ScrA);
			if (_KEY == VK_RIGHT) d = normalize(ScrA);
			d *= 0.01*dot(BMax - BMin, vec3(1.));
			for (int i = 0; i < N; i++) {
				T[i].a += d, T[i].b += d, T[i].c += d;
			}
			calcBoundingBox(); calcVolumeCenterInertia();
		}
		if (_KEY == VK_OEM_PLUS || _KEY == VK_ADD || _KEY == VK_OEM_MINUS || _KEY == VK_SUBTRACT) {
			double s = (_KEY == VK_OEM_PLUS || _KEY == VK_ADD) ? exp(0.05) : exp(-0.05);
			if (Shift) for (int i = 0; i < N; i++) {
				T[i].a.z *= s, T[i].b.z *= s, T[i].c.z *= s;
			}
			else for (int i = 0; i < N; i++) {
				T[i].a = (T[i].a - Center)*s + Center, T[i].b = (T[i].b - Center)*s + Center, T[i].c = (T[i].c - Center)*s + Center;
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
		if (_KEY == VK_NUMPAD0) Center = vec3(0.);
		if (_KEY == VK_NUMPAD4) rz -= PI / 12.;
		if (_KEY == VK_NUMPAD6) rz += PI / 12.;
		if (_KEY == VK_NUMPAD2) rx -= PI / 12.;
		if (_KEY == VK_NUMPAD8) rx += PI / 12.;
		if (_KEY == VK_NUMPAD1) rx = 0, rz = -.5*PI, ry = 0;
		if (_KEY == VK_NUMPAD3) rx = 0, rz = 0;
		if (_KEY == VK_NUMPAD7) rx = .5*PI, rz = -.5*PI, ry = 0;
		if (_KEY == VK_NUMPAD9) rx = -.5*PI, rz = -.5*PI, ry = 0;
		if (_KEY == VK_NUMPAD5) {
			use_orthographic ^= 1;
		}
	}

	if (_KEY == VK_CONTROL) Ctrl = false;
	else if (_KEY == VK_SHIFT) Shift = false;
	else if (_KEY == VK_MENU) Alt = false;
	Render_Needed = true;
}


