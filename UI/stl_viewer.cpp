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
// Single-thread software rasterization that can be slow for large models
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
//   - Default: Phong shading with light gray-based color; Color appears warmer on the normal side and cooler on the back side
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
//   - [0] from file [1] counter-clockwise, right-hand rule [2] clockwise
// Press T to enable/disable transparent rendering mode
//   - Transparent rendering mode is designed for viewing the internal structure of models
//   - Tip: Switch coloring mode (C) to improve clearness and aesthetics
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
// Press P to tell whether the object is a surface or a solid (affects the calculation of physical properties)
// To-do list:
//   - [not trivial] Fix the stroking issue
//   - An option to always show viewport center
//   - Math visualization:
//     - setAxisRatio
//     - A (mandatory) clipping box
//   - Pop out dialog box to show numerical informations of the object
//   - Hide/unhide part of object
//   - Visualization of the objects' bounding box/sphere
//   - Visualization of the volume of the object
//   - Outline rendering based on geometry
//   - Smoothed shading (interpolation)
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
	WNDCLASSEX wc; wc.cbSize = sizeof(WNDCLASSEX), wc.style = 0, wc.lpfnWndProc = WndProc, wc.cbClsExtra = wc.cbWndExtra = 0, wc.hInstance = hInstance; wc.hIcon = wc.hIconSm = 0, wc.hCursor = LoadCursor(NULL, IDC_ARROW), wc.hbrBackground = CreateSolidBrush(RGB(0, 0, 0)), wc.lpszMenuName = NULL, wc.lpszClassName = _T(WIN_NAME); if (!RegisterClassEx(&wc)) return -1;
	_HWND = CreateWindow(_T(WIN_NAME), _T(WIN_NAME), WS_OVERLAPPEDWINDOW, WinW_Padding, WinH_Padding, WinW_Default, WinH_Default, NULL, NULL, hInstance, NULL); ShowWindow(_HWND, nCmdShow); UpdateWindow(_HWND);
	MSG message; while (GetMessage(&message, 0, 0, 0)) { TranslateMessage(&message); DispatchMessage(&message); } return (int)message.wParam;
}


#pragma endregion  // WIN32

// ================================== Vector Classes/Functions ==================================

#pragma region Vector & Matrix

#include "numerical\geometry.h"  // vec2, vec3, mat3
#include "numerical\eigensystem.h"  // EigenPairs_Jacobi
typedef struct { vec3f n, a, b, c; } stl_triangle;
typedef int16_t stl_color;

// convert stl_color to vec3
// https://en.wikipedia.org/wiki/STL_(file_format)#Color_in_binary_STL
vec3f stlColor2vec3(stl_color c) {
	if (c >= 0) return vec3f(1.);  // otherwise, max 31/32=0.97
	float r = ((uint32_t)c & (uint32_t)0b111110000000000) >> 10;
	float g = ((uint32_t)c & (uint32_t)0b000001111100000) >> 5;
	float b = ((uint32_t)c & (uint32_t)0b000000000011111);
	return vec3f(r, g, b) * 0.03125f;
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
float intTriangle(vec3f v0, vec3f v1, vec3f v2, vec3f ro, vec3f rd) {
	vec3f v1v0 = v1 - v0, v2v0 = v2 - v0, rov0 = ro - v0;
	vec3f n = cross(v1v0, v2v0);
	vec3f q = cross(rov0, rd);
	float d = 1.0 / dot(rd, n);
	float u = d * dot(-q, v2v0); if (u<0. || u>1.) return NAN;
	float v = d * dot(q, v1v0); if (v<0. || (u + v)>1.) return NAN;
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
bool TransparentShading = false;
bool showAxis = true, showGrid = true, showMajorMinorGrids = true, showVerticalGrids = false;
bool showCOM = false, showInertia = false, TreatAsSolid = false;
bool DarkBackground = false;

#pragma endregion Camera/Screen, Mouse/Key


#pragma region STL file

WCHAR filename[MAX_PATH] = L"";
int N; stl_triangle *T = 0, *T_transformed = 0;  // triangles
vec3f *T_Color = 0; bool hasInvalidColor = false;  // color
vec3f BMin, BMax;  // bounding box
double Volume; vec3 COM;  // volume/area & center of mass
mat3 Inertia, Inertia_O, Inertia_D;  // inertia tensor calculated at center of mass, orthogonal and diagonal components

void calcBoundingBox() {
	BMin = vec3f(INFINITY), BMax = -BMin;
	for (int i = 0; i < N; i++) {
		if (!isnan(dot(T[i].a, vec3f(1.f)))) BMin = pMin(BMin, T[i].a), BMax = pMax(BMax, T[i].a);
		if (!isnan(dot(T[i].b, vec3f(1.f)))) BMin = pMin(BMin, T[i].b), BMax = pMax(BMax, T[i].b);
		if (!isnan(dot(T[i].c, vec3f(1.f)))) BMin = pMin(BMin, T[i].c), BMax = pMax(BMax, T[i].c);
	}
}
// physics, requires the surface to be closed with outward normals
void calcVolumeCenterInertia() {
	// Assume the object has uniform density
	Volume = 0; COM = vec3(0.0); Inertia = mat3(0.0);
	for (int i = 0; i < N; i++) {
		// I may have a bug
		vec3 a = vec3(T[i].a), b = vec3(T[i].b), c = vec3(T[i].c);
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
double rayIntersect(vec3f ro, vec3f rd) {
	float mt = INFINITY;
	for (int i = 0; i < N; i++) {
		vec3f a = vec3f(T[i].a), b = vec3f(T[i].b), c = vec3f(T[i].c);
		float t = intTriangle(a, b, c, ro, rd);
		if (t > 0 && t < mt) {
			vec3f p = vec3f(Tr.p); float s = Tr.s;
			if (use_orthographic ||
				(dot(p, a) + s > 0 && dot(p, b) + s > 0 && dot(p, c) + s > 0))
				mt = t;
		}
	}
	return (double)mt;
}


#pragma endregion  STL and related calculations


// ============================================ Rendering ============================================

#pragma region Rasterization functions

int BUF_ALLOC = 0;
float *_DEPTHBUF = 0;
vec3f *_COLORBUF = 0;
#define depthbuf(x,y) _DEPTHBUF[(y)*_WIN_W+(x)]
#define colorbuf(x,y) _COLORBUF[(y)*_WIN_W+(x)]


typedef unsigned char byte;
COLORREF toCOLORREF(vec3f c) {
	COLORREF r = 0; byte *k = (byte*)&r;
	k[0] = byte(255.99f * clamp(c.z, 0.f, 1.f));
	k[1] = byte(255.99f * clamp(c.y, 0.f, 1.f));
	k[2] = byte(255.99f * clamp(c.x, 0.f, 1.f));
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


vec3f calcShadeFromID(int i);

// accept double-precision parameters
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
			if (y >= 0 && y < _WIN_H && zf < depthbuf(x, y) + 1e-6) Canvas(x, y) = col, depthbuf(x, y) = zf;
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
			if (x >= 0 && x < _WIN_W && zf < depthbuf(x, y) + 1e-6) Canvas(x, y) = col, depthbuf(x, y) = zf;
			xf += slope, zf += slopez;
		}
	}
}

// accept single-precision parameters
void drawTriangle_ZB(vec3f A, vec3f B, vec3f C, COLORREF fill, COLORREF stroke = -1, COLORREF point = -1) {
	// must be already transformed (due to performance concerns)
	//A = Tr * A, B = Tr * B, C = Tr * C;
	//if (isnan(A.x + B.x + C.x)) return;

	vec2f a = A.xy(), b = B.xy(), c = C.xy();
	vec3f ab = B - A, bc = C - B, ca = A - C, n = cross(ab, ca);
	float k = 1.0 / det(ca.xy(), ab.xy());
	float dz = k * n.x;

	if (fill != -1) {
		int x0 = max((int)(min(min(a.x, b.x), c.x) + 1), 0), x1 = min((int)(max(max(a.x, b.x), c.x) + 1), _WIN_W);
		int y0 = max((int)(min(min(a.y, b.y), c.y) + 1), 0), y1 = min((int)(max(max(a.y, b.y), c.y) + 1), _WIN_H);
		// choose which rasterization method to use based on the size of the triangle
		// creates branches so keep using one method may be faster for well-conditioned scenes
		int size = (x1 - x0 + 1)*(y1 - y0);  // probably not the best guess
		if (size < 8) {
			// pixelwise in/out test
			for (int i = y0; i < y1; i++) {
				for (int j = x0; j < x1; j++) {
					vec2f p(j, i);
					vec2f ap = p - a, bp = p - b, cp = p - c;
					if (((det(ap, bp) < 0) + (det(bp, cp) < 0) + (det(cp, ap) < 0)) % 3 == 0) {
						float z = k * dot(n, vec3f(p.x, p.y, 0.) - A);
						if (z < depthbuf(j, i)) {
							Canvas(j, i) = fill;
							depthbuf(j, i) = z;
						}
						if (TransparentShading) colorbuf(j, i) += calcShadeFromID(fill);
					}
				}
			}
		}
		else if (size < 1024) {
			// incremental edge functions, more initialization cost but less per-pixel cost
			vec2f ab2 = ab.xy().rot(), bc2 = bc.xy().rot(), ca2 = ca.xy().rot();
			float abm = dot(a, ab2), bcm = dot(b, bc2), cam = dot(c, ca2);
			for (int i = y0; i < y1; i++) {
				vec2f p(x0, i);
				float pab = dot(p, ab2) - abm, pbc = dot(p, bc2) - bcm, pca = dot(p, ca2) - cam;
				float z = k * dot(n, vec3f(p.x, p.y, 0.) - A);
				for (int j = x0; j < x1; j++) {
					if (((pab < 0.) + (pbc < 0.) + (pca < 0.)) % 3 == 0) {
						if (z < depthbuf(j, i)) {
							Canvas(j, i) = fill;
							depthbuf(j, i) = z;
						}
						if (TransparentShading) colorbuf(j, i) += calcShadeFromID(fill);
					}
					p.x += 1.;
					pab += ab2.x, pbc += bc2.x, pca += ca2.x;
					z += dz;
				}
			}
		}
		else {
			// scan-line algorithm
			auto fillTrapezoid = [&](int y0, int y1, float m0, float b0, float m1, float b1) {
				// fill the area from x=m0*y+b0 to x=m1*y+b1
				for (int y = y0; y < y1; y++) {
					float xs0 = m0 * y + b0, xs1 = m1 * y + b1;
					if (xs0 > xs1) std::swap(xs0, xs1);
					int x0 = max((int)xs0 + 1, 0);
					int x1 = min((int)xs1 + 1, _WIN_W);
					float z = k * dot(n, vec3f(x0, y, 0.) - A);
					for (int x = x0; x < x1; x++) {
						if (z < depthbuf(x, y)) {
							Canvas(x, y) = fill;
							depthbuf(x, y) = z;
						}
						z += dz;
						if (TransparentShading) colorbuf(x, y) += calcShadeFromID(fill);
					}
				}
			};
			vec3f P[3] = { A, B, C };
			std::swap(P[P[0].y > P[1].y && P[0].y > P[2].y ? 0 : P[1].y > P[2].y ? 1 : 2], P[2]);
			float m0 = (P[0].x - P[2].x) / (P[0].y - P[2].y);
			float m1 = (P[1].x - P[2].x) / (P[1].y - P[2].y);
			fillTrapezoid(max(int(max(P[0].y, P[1].y)) + 1, 0), y1, m0, P[2].x - m0 * P[2].y, m1, P[2].x - m1 * P[2].y);
			if (P[0].y > P[1].y) std::swap(P[0], P[1]);
			m0 = (P[1].x - P[0].x) / (P[1].y - P[0].y);
			m1 = (P[2].x - P[0].x) / (P[2].y - P[0].y);
			fillTrapezoid(y0, min(int(min(P[1].y, P[2].y)) + 1, _WIN_H), m0, P[0].x - m0 * P[0].y, m1, P[0].x - m1 * P[0].y);
		}
	}

	if (stroke != -1) {
		/*double z_min = std::min({
			k * dot(n, vec3(A.xy(), 0) - A),
			k * dot(n, vec3(B.xy(), 0) - A),
			k * dot(n, vec3(C.xy(), 0) - A) });*/
			// WHY I HAVE THE SAME CODE COPY-PASTED THREE TIMES??
		auto drawLine = [&](vec3f p, vec3f q) {
			vec3f d = q - p; float d2 = d.xy().sqr();
			float slope = d.y / d.x;
			if (abs(slope) <= 1.0) {
				if (p.x > q.x) std::swap(p, q);
				int x0 = max(0, int(p.x)), x1 = min(_WIN_W - 1, int(q.x)), y;
				float yf = p.y + slope * (x0 - p.x);
				for (int x = x0; x <= x1; x++, yf += slope) {
					y = (int)(yf + 0.5f);
					if (y >= 0 && y < _WIN_H) {
						if (0) {  // an unsuccessful attempt to fix the stroking bug
							vec2f p(x, y);
							vec2f ap = p - a, bp = p - b, cp = p - c;
							if (((det(ap, bp) < 0) + (det(bp, cp) < 0) + (det(cp, ap) < 0)) % 3 == 0);
							else continue;
						}
						float z = k * dot(n, vec3f(x, y, 0) - A) - 1e-5f;
						if (z <= depthbuf(x, y)) {
							Canvas(x, y) = stroke;
							depthbuf(x, y) = z;
						}
					}
				}
			}
			else {
				slope = d.x / d.y;
				if (p.y > q.y) std::swap(p, q);
				int y0 = max(0, int(p.y)), y1 = min(_WIN_H - 1, int(q.y)), x;
				float xf = p.x + slope * (y0 - p.y);
				for (int y = y0; y <= y1; y++, xf += slope) {
					x = (int)(xf + 0.5);
					if (x >= 0 && x < _WIN_W) {
						// the stroking bug isn't trivial to fix (at least for me)
						// (x,y) isn't always inside the triangle
						if (0) {
							vec2f p(x, y);
							vec2f ap = p - a, bp = p - b, cp = p - c;
							if (((det(ap, bp) < 0) + (det(bp, cp) < 0) + (det(cp, ap) < 0)) % 3 == 0);
							else continue;
						}
						float z = k * dot(n, vec3f(x, y, 0) - A) - 1e-5f;
						if (z <= depthbuf(x, y)) {
							Canvas(x, y) = stroke;
							depthbuf(x, y) = z;
						}
					}
				}
			}
		};
		drawLine(A, B); drawLine(A, C); drawLine(B, C);
	}

	if (point != -1) {
		auto drawDot = [&](vec3f p) {
			int x = int(p.x + .5), y = int(p.y + .5);
			if (x >= 0 && x < _WIN_W && y >= 0 && y < _WIN_H) {
				float z = k * dot(n, vec3f(x, y, 0) - A);
				if (z < depthbuf(x, y)) Canvas(x, y) = point, depthbuf(x, y) = z;
			}
		};
		drawDot(A); drawDot(B); drawDot(C);
	}
}

#pragma endregion



#include <chrono>
std::chrono::steady_clock::time_point _iTimer = std::chrono::high_resolution_clock::now();


// a function that returns the color from triangle id
vec3f light;
vec3f calcShadeFromID(int i) {
	vec3f A = vec3f(T[i].a), B = vec3f(T[i].b), C = vec3f(T[i].c);
	vec3f mid = (A + B + C) * 0.33333333f;
	vec3f n = NormalMode == 0 ? vec3f(T[i].n) :
		NormalMode == 1 ? normalize(cross(B - A, C - A)) :
		NormalMode == 2 ? normalize(cross(C - A, B - A)) : vec3f(0.);
	vec3f c = T_Color[i];  // default: from file
	switch (ShadeMode) {
	case 0: {  // phong
		float k = dot(n, light);
		vec3f d = normalize(mid - vec3f(CamP));
		float spc = dot(d - (2 * dot(d, n))*n, light);
		spc = powf(max(spc, 0.f), 20.0f);
		c = vec3f(0.1f, 0.05f, 0.05f) + (k > 0.f ? vec3f(0.8f, 0.8f, 0.7f) : vec3f(0.6f, 0.7f, 0.8f))*abs(k) + vec3f(0.15f)*spc;
		break;
	}
	case 1: {  // shade by normal
		c = 0.5*(n + vec3f(1.0));
		break;
	}
	case 2: {  // height map, for visualization
		float t = ((A.z + B.z + C.z) / 3. - BMin.z) / (BMax.z - BMin.z);
		if (!isnan(t)) {
			float r = (((((33.038589*t - 100.425221)*t + 116.136811)*t - 67.842553)*t + 23.470346)*t - 4.018730)*t + 0.498427;
			float g = (((((39.767595*t - 128.776104)*t + 160.884144)*t - 98.285228)*t + 27.859936)*t - 1.455343)*t + 0.123248;
			float b = (((((31.953017*t - 102.708635)*t + 118.024799)*t - 53.054309)*t + 3.232700)*t + 2.168052)*t + 0.513964;
			c = vec3f(clamp(r, 0.f, 1.f), clamp(g, 0.f, 1.f), clamp(b, 0.f, 1.f));
		}
		else c = 0.5f*(n + vec3f(1.0f));  // otherwise: shade by normal
		break;
	}
	case 3: {  // directly from file
		break;  // nothing needs to be done here
	}
	case 4: {  // file + Phong
		float k = min(abs(dot(n, light)), 1.f);
		vec3f d = normalize(mid - vec3f(CamP));
		float spc = dot(d - (2.f * dot(d, n))*n, light);
		spc = powf(max(spc, 0.f), 20.0f);
		c *= 0.2f + 0.8f*k + 0.2f*spc;
		break;
	}
	}
	return TransparentShading ? c / abs(dot(n, normalize(mid - vec3f(CamP)))) : c;
};


void render() {
	// initialize
	const int N_PIXEL = _WIN_W * _WIN_H;
	if (N_PIXEL > BUF_ALLOC) {
		if (_DEPTHBUF) { delete _DEPTHBUF; _DEPTHBUF = 0; }
		if (_COLORBUF) { delete _COLORBUF; _COLORBUF = 0; }
		BUF_ALLOC = N_PIXEL;
		_DEPTHBUF = new float[BUF_ALLOC];
		_COLORBUF = new vec3f[BUF_ALLOC];
	}
	for (int i = 0; i < N_PIXEL; i++) _DEPTHBUF[i] = INFINITY;
	if (TransparentShading) for (int i = 0; i < N_PIXEL; i++) _COLORBUF[i] = vec3f(0.);
	calcMat();
	getScreen(CamP, ScrO, ScrA, ScrB);
	light = vec3f(normalize(CamP - Center));

	// calculate projection
	const float TM[4][4] = {
		(float)Tr.u.x, (float)Tr.u.y, (float)Tr.u.z, (float)Tr.t.x,
		(float)Tr.v.x, (float)Tr.v.y, (float)Tr.v.z, (float)Tr.t.y,
		(float)Tr.w.x, (float)Tr.w.y, (float)Tr.w.z, (float)Tr.t.z,
		(float)Tr.p.x, (float)Tr.p.y, (float)Tr.p.z, (float)Tr.s
	};
	auto TF = [&](vec3f p) {
		float x[4] = { p.x, p.y, p.z, 1. };
		float y[4] = { 0., 0., 0., 0. };
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				y[i] += TM[i][j] * x[j];
			}
		}
		return (*(vec3f*)&y[0]) * (y[3] < 0.f ? NAN : 1.f / y[3]);
	};
	for (int i = 0; i < N; i++) {
		T_transformed[i].a = TF(T[i].a);
		T_transformed[i].b = TF(T[i].b);
		T_transformed[i].c = TF(T[i].c);
	}

	// initialize canvas
	const COLORREF BACKGROUND = 0xFFFFFFFF;
	for (int i = 0, l = N_PIXEL; i < l; i++) _WINIMG[i] = BACKGROUND;

	// rasterize triangles, fill pixels using triangle ID
	const COLORREF BORDER = 0xFFFFFFFE;
	for (int i = 0; i < N; i++) {
		vec3f At = T_transformed[i].a, Bt = T_transformed[i].b, Ct = T_transformed[i].c;
		if (!isnan(At.x + Bt.x + Ct.x)) {
			switch (((RenderMode % 4 + 4) % 4) * int(!TransparentShading)) {
			case 0: drawTriangle_ZB(At, Bt, Ct, i); break;
			case 1: drawTriangle_ZB(At, Bt, Ct, i, BORDER); break;  // special ID for border color
			case 2: drawTriangle_ZB(At, Bt, Ct, -1, i); break;
			case 3: drawTriangle_ZB(At, Bt, Ct, -1, -1, i); break;
			}
		}
	}

	// calculate color
	if (TransparentShading) for (int px = 0; px < N_PIXEL; px++) {
		vec3f cc = -0.1f*_COLORBUF[px];
		_WINIMG[px] = toCOLORREF(vec3f(1.f) - vec3f(expf(cc.x), expf(cc.y), expf(cc.z)));
	}
	else for (int px = 0; px < N_PIXEL; px++) {
		int i = _WINIMG[px];
		if (i == BORDER) _WINIMG[px] = 0;
		if (i == BACKGROUND) {
			if (DarkBackground) _WINIMG[px] = 0;
			else _WINIMG[px] = toCOLORREF(mix(vec3f(0.95f), vec3f(0.65f, 0.65f, 1.00f), float(px / _WIN_W) / float(_WIN_H)));
		}
		if (i >= 0) _WINIMG[px] = toCOLORREF(calcShadeFromID(i));
		//_WINIMG[px] = toCOLORREF(vec3(0.5 + 0.5*tanh(-0.005*(_DEPTHBUF[px % _WIN_W][px / _WIN_W] - (Tr*Center).z))));
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

	// top-right coordinate axes
	{
		double trs = Tr.s + dot(Tr.p, Center);
		vec3 xi = Tr * (Center + vec3(trs, 0, 0)), yi = Tr * (Center + vec3(0, trs, 0)), zi = Tr * (Center + vec3(0, 0, trs));
		vec3 ci = Tr * Center;
		double pd = 0.1 * sqrt(_WIN_W*_WIN_H);
		vec2 C(_WIN_W - pd, _WIN_H - pd); pd *= 0.8;
		drawLine(C, C + pd * (xi - ci).xy(), 0xFF0000);
		drawLine(C, C + pd * (yi - ci).xy(), 0x008000);
		drawLine(C, C + pd * (zi - ci).xy(), 0x0000FF);
	}

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
		swprintf(text, L"%s [%d %s]  normal=%s   %dx%d %.1lffps",
			&filename[lastslash],
			N, TreatAsSolid ? L"solid" : L"surface",
			NormalMode == 0 ? L"file" : NormalMode == 1 ? L"ccw" : NormalMode == 2 ? L"cw" : L"ERROR!",
			_WIN_W, _WIN_H, 1.0 / time_elapsed);
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
		if (T) delete T; if (T_transformed) delete T_transformed;
		if (T_Color) delete T_Color;
		T = new stl_triangle[N];
		T_transformed = new stl_triangle[N];
		T_Color = new vec3f[N];
	} catch (...) { T = 0; T_transformed = 0; T_Color = 0; N = 0; return false; }
	hasInvalidColor = false;
	for (int i = 0; i < N; i++) {
		float f[12];
		if (fread(f, sizeof(float), 12, fp) != 12) return false;
		T[i] = stl_triangle{ vec3f(f[0], f[1], f[2]), vec3f(f[3], f[4], f[5]), vec3f(f[6], f[7], f[8]), vec3f(f[9], f[10], f[11]) };
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
		auto writevec3 = [&](vec3f p) {
			float f = float(p.x); fwrite(&f, 4, 1, fp);
			f = float(p.y); fwrite(&f, 4, 1, fp);
			f = float(p.z); fwrite(&f, 4, 1, fp);
		};
		vec3f A = T[i].a, B = T[i].b, C = T[i].c;
		writevec3(
			NormalMode == 0 ? T[i].n :
			NormalMode == 1 ? normalize(cross(B - A, C - A)) :
			NormalMode == 2 ? normalize(cross(C - A, B - A)) : vec3f(0.)
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
	if (isnan(0.0*dot(BMax, BMin))) BMax = vec3f(1.0f), BMin = vec3f(-1.0f);
	if (isnan(Volume*Center.sqr())) Center = vec3(0.0);
	if (!(Volume != 0.0) || isnan(Volume)) Volume = 8.0;
	Center = 0.5*(vec3(BMax) + vec3(BMin));
	vec3 Max = vec3(BMax) - Center, Min = vec3(BMin) - Center;
	double s = max(max(max(Max.x, Max.y), Max.z), -min(min(Min.x, Min.y), Min.z));
	s = 0.2*dot(Max - Min, vec3(1.));
	if (_WIN_W*_WIN_H != 0.0) s /= sqrt(5e-6*_WIN_W*_WIN_H);
	rz = -.25*PI, rx = 0.25, ry = 0.0, dist = 12.0 * s, Unit = 100.0 / s;
	use_orthographic = false;
	//showAxis = true, showGrid = true, showMajorMinorGrids = true, showVerticalGrids = false;
	//TransparentShading = false;
	readedValue = vec3(NAN);
	// do not change color/rendering mode
#if 0
	if (hasInvalidColor) {
		if (ShadeMode == 3 || ShadeMode == 4) ShadeMode = 0;
	}
#endif
	Ctrl = Shift = Alt = false;
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
		vec3f C = 0.5*(BMin + BMax);
		if (Shift) {
			for (int i = 0; i < N; i++)
				T[i].a.z *= sc, T[i].b.z *= sc, T[i].c.z *= sc;
		}
		else {
			for (int i = 0; i < N; i++)
				T[i].a = vec3f((T[i].a - C)*sc + C), T[i].b = vec3f((T[i].b - C)*sc + C), T[i].c = vec3f((T[i].c - C)*sc + C);
		}
		calcBoundingBox(); calcVolumeCenterInertia();
	}
	else if (Shift) {
		if (!use_orthographic) dist *= exp(-0.001*_DELTA);
	}
	else {
		double s = exp(0.001*_DELTA);
		Unit *= s, dist /= s;
	}
}
void MouseDownL(int _X, int _Y) {
	clickCursor = Cursor = vec2(_X, _Y);
	mouse_down = true;
	Render_Needed = true;

}
void MouseMove(int _X, int _Y) {
	//Render_Needed = true;
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
		double t = rayIntersect(vec3f(CamP), vec3f(rd));
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
		double t = rayIntersect(vec3f(CamP), vec3f(rd));
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
		Ctrl = Shift = Alt = false;
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
		Ctrl = Shift = Alt = false;
	}
	else {
		if (_KEY == 'C') ShadeMode = (ShadeMode + (Shift ? 4 : 1)) % 5;
		if (_KEY == VK_TAB && !TransparentShading) RenderMode += Shift ? -1 : 1;
		if (_KEY == 'N') NormalMode = (NormalMode + (Shift ? 2 : 1)) % 3;
		if (_KEY == 'X') showAxis ^= 1;
		if (_KEY == 'G') showGrid ^= !Shift, showMajorMinorGrids ^= Shift;
		if (_KEY == 'F') showMajorMinorGrids ^= 1;
		if (_KEY == 'H') showVerticalGrids ^= 1;
		if (_KEY == 'T') TransparentShading ^= 1;
		if (_KEY == 'M') calcVolumeCenterInertia(), showCOM ^= 1;
		if (_KEY == 'I') calcVolumeCenterInertia(), showInertia ^= 1;
		if (_KEY == 'B') DarkBackground ^= 1;
		if (_KEY == 'P') TreatAsSolid ^= 1, calcVolumeCenterInertia();
		if (_KEY == VK_DECIMAL && !Alt) calcBoundingBox(), Center = 0.5*(vec3(BMax) + vec3(BMin));

		// WSAD, not recommended
		if (dist*Unit < 1e5) {
			vec3 scrdir = normalize(Center - CamP);
			double sc = 0.08*dot(vec3(BMax) - vec3(BMin), vec3(1.));
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
		vec3 Center = 0.5*(vec3(BMin) + vec3(BMax));
		vec3f center = vec3f(Center);
		if (_KEY == VK_NUMPAD4 || _KEY == VK_NUMPAD6 || _KEY == VK_NUMPAD2 || _KEY == VK_NUMPAD8 || _KEY == VK_NUMPAD7 || _KEY == VK_NUMPAD9) {
			mat3 M;
			if (_KEY == VK_NUMPAD4) M = axis_angle(veck, -.025*PI);
			if (_KEY == VK_NUMPAD6) M = axis_angle(veck, .025*PI);
			if (_KEY == VK_NUMPAD2) M = axis_angle(ScrA, .025*PI);
			if (_KEY == VK_NUMPAD8) M = axis_angle(ScrA, -.025*PI);
			if (_KEY == VK_NUMPAD7) M = axis_angle(Center - CamP, -.025*PI);
			if (_KEY == VK_NUMPAD9) M = axis_angle(Center - CamP, .025*PI);
			for (int i = 0; i < N; i++) {
				// do calculation in double precision
				vec3 n = M * vec3(T[i].n);
				vec3 a = M * (vec3(T[i].a) - Center) + Center;
				vec3 b = M * (vec3(T[i].b) - Center) + Center;
				vec3 c = M * (vec3(T[i].c) - Center) + Center;
				T[i] = stl_triangle{ vec3f(n), vec3f(a), vec3f(b), vec3f(c) };
			}
			calcBoundingBox(); calcVolumeCenterInertia();
		}
		if (_KEY == VK_NUMPAD5) {
			for (int i = 0; i < N; i++) {
				T[i].a -= center, T[i].b -= center, T[i].c -= center;
			}
			calcBoundingBox(); calcVolumeCenterInertia();
		}
		if (_KEY >= 0x25 && _KEY <= 0x28) {
			vec3 d;
			if (_KEY == VK_UP) d = normalize(ScrB);
			if (_KEY == VK_DOWN) d = normalize(-ScrB);
			if (_KEY == VK_LEFT) d = normalize(-ScrA);
			if (_KEY == VK_RIGHT) d = normalize(ScrA);
			d *= 0.01*dot(vec3(BMax) - vec3(BMin), vec3(1.));
			vec3f df(d);
			for (int i = 0; i < N; i++) {
				T[i].a += df, T[i].b += df, T[i].c += df;
			}
			calcBoundingBox(); calcVolumeCenterInertia();
		}
		if (_KEY == VK_OEM_PLUS || _KEY == VK_ADD || _KEY == VK_OEM_MINUS || _KEY == VK_SUBTRACT) {
			float s = float((_KEY == VK_OEM_PLUS || _KEY == VK_ADD) ? exp(0.05) : exp(-0.05));
			if (Shift) for (int i = 0; i < N; i++) {
				T[i].a.z *= s, T[i].b.z *= s, T[i].c.z *= s;
			}
			else for (int i = 0; i < N; i++) {
				T[i].a = (T[i].a - center)*s + center, T[i].b = (T[i].b - center)*s + center, T[i].c = (T[i].c - center)*s + center;
			}
			calcBoundingBox(); calcVolumeCenterInertia();
		}
		if (_KEY == VK_DECIMAL) {
			vec3f d = vec3f(vec3(0.5*vec3(BMin + BMax).xy(), BMin.z));
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


