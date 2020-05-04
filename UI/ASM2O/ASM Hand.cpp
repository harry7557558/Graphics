/*

	3D GUI editor - Created for a school project

	Viewport:
		Drag to rotate viewpoint;
		Scroll to zoom; (there is a zooming limit)
		Shift + drag vertically to roll camera;
		Shift + scroll to adjust perspective (move camera without scaling the screen)
		Press . to move view center to the 3D cursor;

	Selection:
		Click to select a point;
		Shift + Click to select multiple points;
		You can also click points on time axis window to select/unselect points (no Shift key required);
		Ctrl + A to select all points;
		Ctrl + I is for inverse selection;
		Click empty space on preview window to unselect all points;

	3D Cursor:
		The position of 3D cursor is the average of all selected points;
		Drag axis to translate points (default)
		[not implemented] Press T to enter translation mode;
		[not implemented] Press R to enter rotation mode;
		[not implemented] Press S to enter scaling mode;

	Time Axis (separated window):
		Vertical lines indicate frame, the lighter one indicates integer second;
		Right click to pin/unpin window (set/unset window to topmost);
		Simple mouse scroll is vertical;
		Ctrl + Scroll to zoom the height of control points;
		Shift + Scroll to zoom the scale of time axis;
		Alt + Scroll to move view along time axis;
		Select a time and move a point to add a keyframe;
		Press Delete/Backspace to delete all selected keyframes;
		Drag selected points along time axis to switch key frame;
		Press tab to insert keyframes identical to the previous keyframe;

	Animation Preview:
		Press space key on any window to preview/pause animation.
		The animation will start from current frame, or from the first if current frame is the last frame.
		When starting preview, all points are unselected.
		[not implemented] Select start and end frame from time axis as preview range.

	Save File:
		Ctrl + S to save file, Ctrl + O to open file.
		Viewport and control point data are saved as text, start with character '#'
		When saving file, file is opened in append mode.
		Opening file will replace the current scene with new scene.
		Due to the lack of error handling, if program crashes due to file format error, restart the program.

*/

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
#define _USE_CONSOLE 0
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
//#define Canvas(x,y) _WINIMG[(y)*_WIN_W+(x)]
inline COLORREF& Canvas(int x, int y) { return _WINIMG[(y)*_WIN_W + (x)]; }
#define setColor(x,y,col) do{if((x)>=0&&(x)<_WIN_W&&(y)>=0&&(y)<_WIN_H)Canvas(x,y)=col;}while(0)


// Second Window: Time Axis

#define WIN_NAME_T "Time Axis"
#define WinTW_Default 640
#define WinTH_Default 180
#define WinTW_Min 400
#define WinTH_Min 120
#define WinTW_Max 1400
#define WinTH_Max 300

void render_t();
void WindowResizeT(int _oldW, int _oldH, int _W, int _H);
void WindowCloseT();
void MouseMoveT(int _X, int _Y);
void MouseWheelT(int _DELTA);
void MouseDownLT(int _X, int _Y);
void MouseUpLT(int _X, int _Y);
void MouseDownRT(int _X, int _Y);
void MouseUpRT(int _X, int _Y);
void KeyDownT(WPARAM _KEY);
void KeyUpT(WPARAM _KEY);

HWND _HWND_T; int _WIN_T_W, _WIN_T_H;
HBITMAP _HIMG_T; COLORREF *_WINIMG_T;
//#define CanvasT(x,y) _WINIMG_T[(y)*_WIN_T_W+(x)]
inline COLORREF& CanvasT(int x, int y) { return _WINIMG_T[(y)*_WIN_T_W + (x)]; }


#pragma endregion


// another forward declaration
void WriteImage();


// Win32 Entry

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
	bool isM = hWnd == _HWND, isT = hWnd == _HWND_T;
#define _WNDSEL(vm,vt) (isM?(vm):isT?(vt):NULL)
#define _WNDSEL_T(vn) (isM?(vn):isT?(vn##T):NULL)
	auto hImg = _WNDSEL(&_HIMG, &_HIMG_T);
	auto winImg = _WNDSEL(&_WINIMG, &_WINIMG_T);
	auto winW = _WNDSEL(&_WIN_W, &_WIN_T_W), winH = _WNDSEL(&_WIN_H, &_WIN_T_H);
#define _RD_RAW { HDC hdc = GetDC(hWnd), HImgMem = CreateCompatibleDC(hdc); HBITMAP hbmOld = (HBITMAP)SelectObject(HImgMem, *hImg); _WNDSEL(render, render_t)(); BitBlt(hdc, 0, 0, *winW, *winH, HImgMem, 0, 0, SRCCOPY); SelectObject(HImgMem, hbmOld), DeleteDC(HImgMem), DeleteDC(hdc); }
#define _RDBK { SendMessage(_WNDSEL(_HWND_T, _HWND), WM_NULL, NULL, NULL); _RD_RAW break; }
	switch (message) {
	case WM_NULL: { _RD_RAW return 0; }
	case WM_CREATE: { if (!_HWND) Init(); break; }
	case WM_CLOSE: { if (isM) { _WNDSEL_T(WindowClose)(); DestroyWindow(hWnd); } return 0; }
	case WM_DESTROY: { if (isM) { PostQuitMessage(0); } return 0; }
	case WM_MOVE:; case WM_SIZE: {
		RECT Client; GetClientRect(hWnd, &Client); _WNDSEL_T(WindowResize)(*winW, *winH, Client.right, Client.bottom); *winW = Client.right, *winH = Client.bottom;
		BITMAPINFO bmi; bmi.bmiHeader.biSize = sizeof(BITMAPINFO), bmi.bmiHeader.biWidth = Client.right, bmi.bmiHeader.biHeight = Client.bottom, bmi.bmiHeader.biPlanes = 1, bmi.bmiHeader.biBitCount = 32; bmi.bmiHeader.biCompression = BI_RGB, bmi.bmiHeader.biSizeImage = 0, bmi.bmiHeader.biXPelsPerMeter = bmi.bmiHeader.biYPelsPerMeter = 0, bmi.bmiHeader.biClrUsed = bmi.bmiHeader.biClrImportant = 0; bmi.bmiColors[0].rgbBlue = bmi.bmiColors[0].rgbGreen = bmi.bmiColors[0].rgbRed = bmi.bmiColors[0].rgbReserved = 0;
		if (*hImg != NULL) DeleteObject(*hImg); HDC hdc = GetDC(hWnd); *hImg = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, (void**)winImg, NULL, 0); DeleteDC(hdc); _RDBK }
	case WM_GETMINMAXINFO: { if (isM || isT) { LPMINMAXINFO lpMMI = (LPMINMAXINFO)lParam; lpMMI->ptMinTrackSize.x = _WNDSEL(WinW_Min, WinTW_Min), lpMMI->ptMinTrackSize.y = _WNDSEL(WinH_Min, WinTH_Min), lpMMI->ptMaxTrackSize.x = _WNDSEL(WinW_Max, WinTW_Max), lpMMI->ptMaxTrackSize.y = _WNDSEL(WinH_Max, WinTH_Max); } break; }
	case WM_PAINT: { PAINTSTRUCT ps; HDC hdc = BeginPaint(hWnd, &ps), HMem = CreateCompatibleDC(hdc); HBITMAP hbmOld = (HBITMAP)SelectObject(HMem, *hImg); BitBlt(hdc, 0, 0, *winW, *winH, HMem, 0, 0, SRCCOPY); SelectObject(HMem, hbmOld); EndPaint(hWnd, &ps); DeleteDC(HMem), DeleteDC(hdc); break; }
#define _USER_FUNC_PARAMS GET_X_LPARAM(lParam), *winH - 1 - GET_Y_LPARAM(lParam)
	case WM_MOUSEMOVE: { _WNDSEL_T(MouseMove)(_USER_FUNC_PARAMS); _RDBK }
	case WM_MOUSEWHEEL: { _WNDSEL_T(MouseWheel)(GET_WHEEL_DELTA_WPARAM(wParam)); _RDBK }
	case WM_LBUTTONDOWN: { SetCapture(hWnd); _WNDSEL_T(MouseDownL)(_USER_FUNC_PARAMS); _RDBK }
	case WM_LBUTTONUP: { ReleaseCapture(); _WNDSEL_T(MouseUpL)(_USER_FUNC_PARAMS); _RDBK }
	case WM_RBUTTONDOWN: { _WNDSEL_T(MouseDownR)(_USER_FUNC_PARAMS); _RDBK }
	case WM_RBUTTONUP: { _WNDSEL_T(MouseUpR)(_USER_FUNC_PARAMS); _RDBK }
	case WM_SYSKEYDOWN:; case WM_KEYDOWN: { if (wParam >= 0x08) _WNDSEL_T(KeyDown)(wParam); _RDBK }
	case WM_SYSKEYUP:; case WM_KEYUP: { if (wParam >= 0x08) _WNDSEL_T(KeyUp)(wParam); _RDBK }
	} return DefWindowProc(hWnd, message, wParam, lParam);
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow) {
	//WriteImage(); return 0;
	if (_USE_CONSOLE) if (AttachConsole(ATTACH_PARENT_PROCESS) || AllocConsole()) freopen("CONIN$", "r", stdin), freopen("CONOUT$", "w", stdout), freopen("CONOUT$", "w", stderr);
	WNDCLASSEX wc; wc.cbSize = sizeof(WNDCLASSEX), wc.style = 0, wc.lpfnWndProc = WndProc, wc.cbClsExtra = wc.cbWndExtra = 0, wc.hInstance = hInstance; wc.hIcon = wc.hIconSm = 0, wc.hCursor = LoadCursor(NULL, IDC_ARROW), wc.hbrBackground = CreateSolidBrush(RGB(0, 0, 0)), wc.lpszMenuName = NULL, wc.lpszClassName = _T(WIN_NAME);
	if (!RegisterClassEx(&wc)) return -1;
	_HWND_T = CreateWindowEx(0, _T(WIN_NAME), _T(WIN_NAME_T), WS_OVERLAPPEDWINDOW ^ WS_MAXIMIZEBOX, WinW_Padding, WinH_Padding + WinH_Default, WinTW_Default, WinTH_Default, NULL, NULL, hInstance, NULL);
	ShowWindow(_HWND_T, nCmdShow); UpdateWindow(_HWND_T);
	_HWND = CreateWindow(_T(WIN_NAME), _T(WIN_NAME), WS_OVERLAPPEDWINDOW, WinW_Padding, WinH_Padding, WinW_Default, WinH_Default, NULL, NULL, hInstance, NULL);
	ShowWindow(_HWND, nCmdShow); UpdateWindow(_HWND);
	MSG message; while (GetMessage(&message, 0, 0, 0)) { TranslateMessage(&message); DispatchMessage(&message); } return (int)message.wParam;
}

#pragma endregion

// COLORREF
enum WebSafeColors {
	ALICEBLUE = 0xF0F8FF, ANTIQUEWHITE = 0xFAEBD7, AQUA = 0x00FFFF, AQUAMARINE = 0x7FFFD4, AZURE = 0xF0FFFF, BEIGE = 0xF5F5DC,
	BISQUE = 0xFFE4C4, BLACK = 0x000000, BLANCHEDALMOND = 0xFFEBCD, BLUE = 0x0000FF, BLUEVIOLET = 0x8A2BE2, BROWN = 0xA52A2A,
	BURLYWOOD = 0xDEB887, CADETBLUE = 0x5F9EA0, CHARTREUSE = 0x7FFF00, CHOCOLATE = 0xD2691E, CORAL = 0xFF7F50, CORNFLOWERBLUE = 0x6495ED,
	CORNSILK = 0xFFF8DC, CRIMSON = 0xDC143C, CYAN = 0x00FFFF, DARKBLUE = 0x00008B, DARKCYAN = 0x008B8B, DARKGOLDENROD = 0xB8860B,
	DARKGRAY = 0xA9A9A9, DARKGREY = 0xA9A9A9, DARKGREEN = 0x006400, DARKKHAKI = 0xBDB76B, DARKMAGENTA = 0x8B008B, DARKOLIVEGREEN = 0x556B2F,
	DARKORANGE = 0xFF8C00, DARKORCHID = 0x9932CC, DARKRED = 0x8B0000, DARKSALMON = 0xE9967A, DARKSEAGREEN = 0x8FBC8F, DARKSLATEBLUE = 0x483D8B,
	DARKSLATEGRAY = 0x2F4F4F, DARKSLATEGREY = 0x2F4F4F, DARKTURQUOISE = 0x00CED1, DARKVIOLET = 0x9400D3, DEEPPINK = 0xFF1493, DEEPSKYBLUE = 0x00BFFF,
	DIMGRAY = 0x696969, DIMGREY = 0x696969, DODGERBLUE = 0x1E90FF, FIREBRICK = 0xB22222, FLORALWHITE = 0xFFFAF0, FORESTGREEN = 0x228B22,
	FUCHSIA = 0xFF00FF, GAINSBORO = 0xDCDCDC, GHOSTWHITE = 0xF8F8FF, GOLD = 0xFFD700, GOLDENROD = 0xDAA520, GRAY = 0x808080,
	GREY = 0x808080, GREEN = 0x008000, GREENYELLOW = 0xADFF2F, HONEYDEW = 0xF0FFF0, HOTPINK = 0xFF69B4, INDIANRED = 0xCD5C5C,
	INDIGO = 0x4B0082, IVORY = 0xFFFFF0, KHAKI = 0xF0E68C, LAVENDER = 0xE6E6FA, LAVENDERBLUSH = 0xFFF0F5, LAWNGREEN = 0x7CFC00,
	LEMONCHIFFON = 0xFFFACD, LIGHTBLUE = 0xADD8E6, LIGHTCORAL = 0xF08080, LIGHTCYAN = 0xE0FFFF, LIGHTGOLDENRODYELLOW = 0xFAFAD2, LIGHTGRAY = 0xD3D3D3,
	LIGHTGREY = 0xD3D3D3, LIGHTGREEN = 0x90EE90, LIGHTPINK = 0xFFB6C1, LIGHTSALMON = 0xFFA07A, LIGHTSEAGREEN = 0x20B2AA, LIGHTSKYBLUE = 0x87CEFA,
	LIGHTSLATEGRAY = 0x778899, LIGHTSLATEGREY = 0x778899, LIGHTSTEELBLUE = 0xB0C4DE, LIGHTYELLOW = 0xFFFFE0, LIME = 0x00FF00, LIMEGREEN = 0x32CD32,
	LINEN = 0xFAF0E6, MAGENTA = 0xFF00FF, MAROON = 0x800000, MEDIUMAQUAMARINE = 0x66CDAA, MEDIUMBLUE = 0x0000CD, MEDIUMORCHID = 0xBA55D3,
	MEDIUMPURPLE = 0x9370DB, MEDIUMSEAGREEN = 0x3CB371, MEDIUMSLATEBLUE = 0x7B68EE, MEDIUMSPRINGGREEN = 0x00FA9A, MEDIUMTURQUOISE = 0x48D1CC, MEDIUMVIOLETRED = 0xC71585,
	MIDNIGHTBLUE = 0x191970, MINTCREAM = 0xF5FFFA, MISTYROSE = 0xFFE4E1, MOCCASIN = 0xFFE4B5, NAVAJOWHITE = 0xFFDEAD, NAVY = 0x000080,
	OLDLACE = 0xFDF5E6, OLIVE = 0x808000, OLIVEDRAB = 0x6B8E23, ORANGE = 0xFFA500, ORANGERED = 0xFF4500, ORCHID = 0xDA70D6,
	PALEGOLDENROD = 0xEEE8AA, PALEGREEN = 0x98FB98, PALETURQUOISE = 0xAFEEEE, PALEVIOLETRED = 0xDB7093, PAPAYAWHIP = 0xFFEFD5, PEACHPUFF = 0xFFDAB9,
	PERU = 0xCD853F, PINK = 0xFFC0CB, PLUM = 0xDDA0DD, POWDERBLUE = 0xB0E0E6, PURPLE = 0x800080, REBECCAPURPLE = 0x663399,
	RED = 0xFF0000, ROSYBROWN = 0xBC8F8F, ROYALBLUE = 0x4169E1, SADDLEBROWN = 0x8B4513, SALMON = 0xFA8072, SANDYBROWN = 0xF4A460,
	SEAGREEN = 0x2E8B57, SEASHELL = 0xFFF5EE, SIENNA = 0xA0522D, SILVER = 0xC0C0C0, SKYBLUE = 0x87CEEB, SLATEBLUE = 0x6A5ACD,
	SLATEGRAY = 0x708090, SLATEGREY = 0x708090, SNOW = 0xFFFAFA, SPRINGGREEN = 0x00FF7F, STEELBLUE = 0x4682B4, TAN = 0xD2B48C,
	TEAL = 0x008080, THISTLE = 0xD8BFD8, TOMATO = 0xFF6347, TURQUOISE = 0x40E0D0, VIOLET = 0xEE82EE, WHEAT = 0xF5DEB3,
	WHITE = 0xFFFFFF, WHITESMOKE = 0xF5F5F5, YELLOW = 0xFFFF00, YELLOWGREEN = 0x9ACD32,
};

// ================================== Vector Classes/Functions ==================================

#pragma region Vector & Matrix

//#define Inline inline
#define Inline __inline
//#define Inline __forceinline

#define PI 3.1415926535897932384626
#define mix(x,y,a) ((x)*(1.0-(a))+(y)*(a))
#define clamp(x,a,b) ((x)<(a)?(a):(x)>(b)?(b):(x))
Inline double mod(double x, double m) { return x - m * floor(x / m); }

class vec2 {
public:
	double x, y;
	Inline explicit vec2() {}
	Inline explicit vec2(const double &a) :x(a), y(a) {}
	Inline explicit vec2(const double &x, const double &y) :x(x), y(y) {}
	Inline vec2 operator - () const { return vec2(-x, -y); }
	Inline vec2 operator + (const vec2 &v) const { return vec2(x + v.x, y + v.y); }
	Inline vec2 operator - (const vec2 &v) const { return vec2(x - v.x, y - v.y); }
	Inline vec2 operator * (const vec2 &v) const { return vec2(x * v.x, y * v.y); }	// not standard
	Inline vec2 operator * (const double &a) const { return vec2(x*a, y*a); }
	Inline double sqr() const { return x * x + y * y; } 	// not standard
	Inline friend double length(const vec2 &v) { return sqrt(v.x*v.x + v.y*v.y); }
	Inline friend vec2 normalize(const vec2 &v) { return v * (1. / sqrt(v.x*v.x + v.y*v.y)); }
	Inline friend double dot(const vec2 &u, const vec2 &v) { return u.x*v.x + u.y*v.y; }
	Inline friend double det(const vec2 &u, const vec2 &v) { return u.x*v.y - u.y*v.x; } 	// not standard
#if 1
	Inline void operator += (const vec2 &v) { x += v.x, y += v.y; }
	Inline void operator -= (const vec2 &v) { x -= v.x, y -= v.y; }
	Inline void operator *= (const vec2 &v) { x *= v.x, y *= v.y; }
	Inline friend vec2 operator * (const double &a, const vec2 &v) { return vec2(a*v.x, a*v.y); }
	Inline void operator *= (const double &a) { x *= a, y *= a; }
	Inline vec2 operator / (const double &a) const { return vec2(x / a, y / a); }
	Inline void operator /= (const double &a) { x /= a, y /= a; }
#endif
	Inline vec2 yx() const { return vec2(y, x); }
	Inline vec2 rot() const { return vec2(-y, x); }
	Inline vec2 rotr() const { return vec2(y, -x); }
#if 1
	// added when needed
	Inline bool operator == (const vec2 &v) const { return x == v.x && y == v.y; }
	Inline bool operator != (const vec2 &v) const { return x != v.x || y != v.y; }
	Inline vec2 operator / (const vec2 &v) const { return vec2(x / v.x, y / v.y); }
	Inline friend vec2 pMax(const vec2 &a, const vec2 &b) { return vec2(max(a.x, b.x), max(a.y, b.y)); }
	Inline friend vec2 pMin(const vec2 &a, const vec2 &b) { return vec2(min(a.x, b.x), min(a.y, b.y)); }
	Inline friend vec2 abs(const vec2 &a) { return vec2(abs(a.x), abs(a.y)); }
	Inline friend vec2 floor(const vec2 &a) { return vec2(floor(a.x), floor(a.y)); }
	Inline friend vec2 ceil(const vec2 &a) { return vec2(ceil(a.x), ceil(a.y)); }
	Inline friend vec2 sqrt(const vec2 &a) { return vec2(sqrt(a.x), sqrt(a.y)); }
	Inline friend vec2 sin(const vec2 &a) { return vec2(sin(a.x), sin(a.y)); }
	Inline friend vec2 cos(const vec2 &a) { return vec2(cos(a.x), cos(a.y)); }
	Inline friend vec2 atan(const vec2 &a) { return vec2(atan(a.x), atan(a.y)); }
#endif
};

class vec3 {
public:
	double x, y, z;
	Inline explicit vec3() {}
	Inline explicit vec3(const double &a) :x(a), y(a), z(a) {}
	Inline explicit vec3(const double &x, const double &y, const double &z) :x(x), y(y), z(z) {}
	Inline explicit vec3(const vec2 &v, const double &z) :x(v.x), y(v.y), z(z) {}
	Inline vec3 operator - () const { return vec3(-x, -y, -z); }
	Inline vec3 operator + (const vec3 &v) const { return vec3(x + v.x, y + v.y, z + v.z); }
	Inline vec3 operator - (const vec3 &v) const { return vec3(x - v.x, y - v.y, z - v.z); }
	Inline vec3 operator * (const vec3 &v) const { return vec3(x * v.x, y * v.y, z * v.z); }
	Inline vec3 operator * (const double &k) const { return vec3(k * x, k * y, k * z); }
	Inline double sqr() const { return x * x + y * y + z * z; } 	// non-standard
	Inline friend double length(vec3 v) { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
	Inline friend vec3 normalize(vec3 v) { return v * (1. / sqrt(v.x*v.x + v.y*v.y + v.z*v.z)); }
	Inline friend double dot(vec3 u, vec3 v) { return u.x*v.x + u.y*v.y + u.z*v.z; }
	Inline friend vec3 cross(vec3 u, vec3 v) { return vec3(u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x); }
#if 1
	Inline void operator += (const vec3 &v) { x += v.x, y += v.y, z += v.z; }
	Inline void operator -= (const vec3 &v) { x -= v.x, y -= v.y, z -= v.z; }
	Inline void operator *= (const vec3 &v) { x *= v.x, y *= v.y, z *= v.z; }
	Inline friend vec3 operator * (const double &a, const vec3 &v) { return vec3(a*v.x, a*v.y, a*v.z); }
	Inline void operator *= (const double &a) { x *= a, y *= a, z *= a; }
	Inline vec3 operator / (const double &a) const { return vec3(x / a, y / a, z / a); }
	Inline void operator /= (const double &a) { x /= a, y /= a, z /= a; }
#endif
	Inline vec2 xy() const { return vec2(x, y); }
	Inline vec2 xz() const { return vec2(x, z); }
	Inline vec2 yz() const { return vec2(y, z); }
#if 1
	Inline bool operator == (const vec3 &v) const { return x == v.x && y == v.y && z == v.z; }
	Inline bool operator != (const vec3 &v) const { return x != v.x || y != v.y || z != v.z; }
	Inline vec3 operator / (const vec3 &v) const { return vec3(x / v.x, y / v.y, z / v.z); }
	Inline friend vec3 pMax(const vec3 &a, const vec3 &b) { return vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }
	Inline friend vec3 pMin(const vec3 &a, const vec3 &b) { return vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
	Inline friend vec3 abs(const vec3 &a) { return vec3(abs(a.x), abs(a.y), abs(a.z)); }
	Inline friend vec3 floor(const vec3 &a) { return vec3(floor(a.x), floor(a.y), floor(a.z)); }
	Inline friend vec3 ceil(const vec3 &a) { return vec3(ceil(a.x), ceil(a.y), ceil(a.z)); }
	Inline friend vec3 mod(const vec3 &a, double m) { return vec3(mod(a.x, m), mod(a.y, m), mod(a.z, m)); }
#endif
};

const vec3 veci(1, 0, 0), vecj(0, 1, 0), veck(0, 0, 1);
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


#pragma region Ray Tracing Functions

// Many from https://www.iquilezles.org/www/articles/intersectors/intersectors.htm
// Marked as /**/ means I just copy-paste it without understanding how it works.
// All direction vectors should be normalized


// Intersection functions
double intXOY(vec3 p, vec3 d) {
	return -p.z / d.z;
}
double intSphere(vec3 O, double r, vec3 p, vec3 d) {
	p = p - O;
#if 0
	if (dot(p, d) >= 0.0) return NAN;
	vec3 k = cross(p, d); double rd2 = dot(k, k); if (rd2 > r*r) return NAN;
	return sqrt(dot(p, p) - rd2) - sqrt(r*r - rd2);
#else
	// works when p is inside the sphere (and its slightly faster)
	double b = -dot(p, d), c = dot(p, p) - r * r;  // require d to be normalized
	double delta = b * b - c;
	if (delta < 0.0) return NAN;
	delta = sqrt(delta);
	c = b - delta;
	return c > 0. ? c : b + delta;  // usually we want it to be positive
#endif
}
/**/double intCapsule(vec3 pa, vec3 pb, double r, vec3 ro, vec3 rd) {
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
/**/double intCylinder(vec3 pa, vec3 pb, double ra, vec3 ro, vec3 rd) {
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
/**/double intBoxC(vec3 R, vec3 ro, vec3 rd) {
	vec3 m = vec3(1.0) / rd, n = m * ro;
	vec3 k = abs(m)*R;
	vec3 t1 = -n - k, t2 = -n + k;
	double tN = max(max(t1.x, t1.y), t1.z);
	double tF = min(min(t2.x, t2.y), t2.z);
	if (tN > tF || tF < 0.0) return NAN;
	return tN;
}

// Normal calculation
vec3 nCapsule(vec3 a, vec3 b, double r, vec3 p) {
	vec3 ba = b - a, pa = p - a;
	double h = dot(pa, ba) / dot(ba, ba);
	return (pa - clamp(h, 0., 1.) * ba) / r;
}

// SDF functions
double sdBox(vec2 p, vec2 b) {
	vec2 d = abs(p) - b;
	return length(pMax(d, vec2(0))) + min(max(d.x, d.y), 0.0);
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

#pragma endregion return the distance, NAN means no intersection



// ======================================== Data / Parameters ========================================

// viewport
vec3 Center(0.0, 0.5, 0.0);  // view center in world coordinate
double rz = 0.25*PI, rx = 0.1*PI, ry = 0.0, dist = 12.0, Unit = 100.0;  // yaw, pitch, row, camera distance, scale to screen

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

// file forward declaration
bool saveFile(const WCHAR filename[]);
bool saveFileUserEntry();
bool readFile(const WCHAR filename[]);
bool readFileUserEntry();

#pragma endregion

#pragma region Global Variable Related Functions

// projection
Affine axis_angle(vec3 axis, double a) {
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
	Affine Y = axis_angle(w, -ry); u = Y * u, v = Y * v;
	u *= 0.5*_WIN_W / Unit, v *= 0.5*_WIN_H / Unit, w *= dist;
	P = Center + w;
	O = Center - (u + v), A = u * 2.0, B = v * 2.0;
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

#pragma endregion

#pragma region Timer Axis

#include <chrono>
typedef std::chrono::high_resolution_clock NTime;
typedef std::chrono::duration<double> fsec;
NTime::time_point _Global_Timer = NTime::now();
#define iTime fsec(NTime::now()-_Global_Timer).count()

// window parameters
const int FPS = 25;  // frame rate, constant
const double FrameDelay = 1.0 / FPS;

// the animation will be based on frame instead of time
double UnitT = 15.0;  // the width of one frame show on screen
double LFrame = 0;  // the frame at the left of the screen
int currentFrame = 0;  // current frame to edit
double previewFrame = 0.0; // current time to preview
double getFrame(double Cursor_x) {  // return type is float point
	double f = (Cursor_x / UnitT) + LFrame;
	return max(f, 0);
}
double FrameToCoord(double Frame) {
	return (Frame - LFrame)*UnitT;
}

// the vertical axis of time axis window
double UnitTV = 18.0;  // the height on screen of one control point
double rdrRadiusT = 4.0;  // side length of one point on screen
int BObject = 0;  // the object at the bottom of the screen
int HObject = -1;  // mouse hover object
int getHoverObject(double Cursor_y) {  // return mouse hover object
	return int(Cursor_y / UnitTV) + BObject;
}
double ObjectIDToCoord(int d) {
	return (d - BObject)*UnitTV;
}

vec2 timeAxisSquareCenter(double frame, int obj) {  // note that frame is floatpoint
	return vec2(FrameToCoord(frame) + 0.5*UnitT, ObjectIDToCoord(obj) + 0.5*UnitTV);
}

// user parameters
vec2 CursorT = vec2(0, 0), clickCursorT;
bool mouse_down_T = false;
//int pointOnMove[NCtrPs];  // points under on time change, the value is its original frame
// the above variable is moved to the next region

#pragma endregion

#pragma region Scene Variables

#include <vector>
#include <stack>

// control points
#define NCtrPs 21  // # of control points, constant
class ControlPoint {
	struct Point {
		vec3 P;  // position
		int F; // frame, integer
	};
	friend bool readFile(const WCHAR filename[]);
public:
	std::vector<Point> keyFrames;  // F sorted in increasing order
	bool selected = false;  // whether this point is being selected and ready to be edited
private:
	int _lower_bound(int t) const {
		//return std::lower_bound(keyFrames.begin(), keyFrames.end() - 1, t, [](Point A, double B) { return A.F < B; }) - keyFrames.begin();
		for (int i = 0, n = keyFrames.size(); i < n; i++) if (keyFrames[i].F > t) return max(i - 1, 0); return keyFrames.size() - 1;
	}
public:
	// just too lazy to do optimization for this part ;)
	ControlPoint() {}
	ControlPoint(vec3 P) { keyFrames.push_back(Point{ P, 0 }); }
	bool existFrame(int t) {
		return keyFrames[_lower_bound(t)].F == t;
	}
	bool isOver(int t) {
		return t > keyFrames.back().F;
	}
	void sortFrames() {
		std::sort(keyFrames.begin(), keyFrames.end(), [](Point A, Point B)->bool {return A.F < B.F; });
	}
	//void detectFrameCollision() {}

	vec3 P(double t = previewFrame) const {  // write interpolation code there
		int d = _lower_bound(t);
		//return keyFrames[d].P;
		int e = d + 1;
		if (e == keyFrames.size()) return keyFrames[d].P;
		double u = (t - keyFrames[d].F) / (keyFrames[e].F - keyFrames[d].F);
		return (1 - u)*keyFrames[d].P + u * keyFrames[e].P;
	}
	vec3& getP(int t = currentFrame, bool interp = true) {  // if the keyframe exists, return point; otherwise, create
		int d = _lower_bound(t);
		if (keyFrames[d].F == t) return keyFrames[d].P;
		keyFrames.insert(keyFrames.begin() + d + 1, Point{ interp ? P(t) : keyFrames[d].P, t });
		return keyFrames[d + 1].P;
	}

	bool deleteFrame(int t) {
		int d = _lower_bound(t);
		if (keyFrames[d].F != t) return false;
		keyFrames.erase(keyFrames.begin() + d);
		return true;
	}
	bool setFrame(int old, int _new) {
		if (!existFrame(old) || existFrame(_new)) return false;
		int d = _lower_bound(old);
		keyFrames[d].F = _new;
		sortFrames();
		return true;
	}
};
namespace Hand {
	// names for control points - this scene is a human hand (left)
	enum ControlPoints {
		A0, A1, A2, A3,
		B1, B2, B3, B4,
		C1, C2, C3, C4,
		D1, D2, D3, D4,
		E0, E1, E2, E3, E4,
	};
}
#define CP(...) ControlPoint{vec3(##__VA_ARGS__)}
ControlPoint CPs[NCtrPs] = {  // default positions of control points
#if 0
	CP(-0.20,-0.50,0.20), CP(-0.50,-0.15,0.02), CP(-0.70,0.15,0.15), CP(-1.00, 0.30,0.15),
	CP(-0.22,0.31,-0.05), CP(-0.36,0.65,-0.1), CP(-0.42,0.85,-0.15), CP(-0.48,1.09,-0.08),
	CP(0.00,0.28,-0.08), CP(0.00,0.74,-0.23), CP(0.00,1.00,-0.16), CP(0.00,1.25,-0.08),
	CP(0.17,0.20,-0.08), CP(0.30,0.65,-0.08), CP(0.34,0.84,-0.12), CP(0.41,1.09,0.00),
	CP(0.23,-0.50,0.10), CP(0.39,0.16,-0.06), CP(0.58,0.36,-0.14), CP(0.62,0.51,-0.04), CP(0.68,0.75,0.09)
#else
	CP(-0.2,-0.5,0.08), CP(-0.5,-0.18,-0.01), CP(-0.68,0.15,0.03), CP(-0.98,0.3,0.03),
	CP(-0.22,0.31,-0.05), CP(-0.36,0.65,-0.14), CP(-0.42,0.86,-0.15), CP(-0.50,1.12,-0.08),
	CP(0,0.28,-0.08), CP(0,0.74,-0.23), CP(0,1,-0.16), CP(0,1.25,-0.08),
	CP(0.17,0.2,-0.08), CP(0.3,0.65,-0.17), CP(0.34,0.84,-0.12), CP(0.41,1.09,0),
	CP(0.23,-0.5,0.1), CP(0.39,0.10,-0.06), CP(0.53,0.36,-0.09), CP(0.62,0.51,-0.04), CP(0.68,0.75,0.09),
#endif
};
#undef CP

int NSelectedPoints() {  // # of selected control points
	int N = 0;
	for (int i = 0; i < NCtrPs; i++) N += CPs[i].selected;
	return N;
}
int lastFrame() {  // return the last frame of the animation
	int endFrame = 0;
	for (int i = 0; i < NCtrPs; i++) {
		endFrame = max(endFrame, CPs[i].keyFrames.back().F);
	}
	return endFrame;
}

// 3D cursor
vec3 CPC(NAN);  // location of 3D cursor (average of all selected points)
bool selected = false;  // true if at least one point is selected
const double selRadiusP = 3.0;  // side length for selecting a control point
const double selRadius = 6.0;  // radius for selecting the cursor
const double selAxisRatio = 0.3;  // ratio of radius for selecting cursor and selecting cursor axis
const double selLength = 60.0;  // (approximate) maximum length of cursor axis in screen coordinate
enum moveDirection { none = -1, unlimited, xAxis, yAxis, zAxis, xOy, xOz, yOz };  // not all implemented
moveDirection moveAlong(none);  // which part of the cursor is being moved
int updateCursorPosition() {  // return the # of selected pointss
	double totP = 0.0; vec3 sumP(0.0);
	for (int i = 0; i < NCtrPs; i++) {
		if (CPs[i].selected) totP++, sumP += CPs[i].P();
	}
	CPC = sumP / totP;  // no point selected -> 0/0=NAN
	if (totP) selected = true;
	else selected = false;
	return totP;
}
void centerViewport() { // Move viewport center to the location of the cursor
	if (selected) Center = CPC;
	else {
		Center = vec3(0.0);
		for (int i = 0; i < NCtrPs; i++) Center += CPs[i].P();
		Center /= NCtrPs;
	}
}

// Selection
void selectAll() {
	for (int i = 0; i < NCtrPs; i++) CPs[i].selected = true;
	updateCursorPosition();
}
void inverseSelection() {
	for (int i = 0; i < NCtrPs; i++) CPs[i].selected ^= 1;
	updateCursorPosition();
}


int pointOnMove[NCtrPs];

#pragma endregion

#pragma region Preview Animation

#include <thread>

int startFrame = 0, endFrame = 0;  // preview start and end frames
auto startTime = NTime::now();  // the time when animation started
auto animateTime = []()->double { return fsec(NTime::now() - startTime).count(); };  // elapsed time after animation started
bool isAnimating = false;  // is previewing?
HANDLE animationThread;  // the thread running this animation

DWORD WINAPI animationProcessFunction(HANDLE H) {
	startTime = NTime::now();
	for (int i = startFrame; i <= endFrame; i++) {
		previewFrame = currentFrame = i;
		SendMessage(_HWND, WM_NULL, NULL, NULL);
		SendMessage(_HWND_T, WM_NULL, NULL, NULL);
		double t = (i - startFrame)*FrameDelay - animateTime();
		std::this_thread::sleep_for(std::chrono::milliseconds(int(1000 * t)));
		// handle the case when rendering too slow
		if (i >= endFrame) break;
		i = min(int(animateTime() / FrameDelay + 0.01) + startFrame, endFrame - 1);  // -1 because i++
	}
	dbgprint("Elapsed Time: %lfsecs, %lfframes\n", animateTime(), animateTime() / FrameDelay);
	isAnimating = false;
	return 0;
}
void startAnimation(int frame = currentFrame) {
	startFrame = frame;
	endFrame = lastFrame();
	// unselect elements so the animation can be run smoothly
	for (int i = 0; i < NCtrPs; i++) CPs[i].selected = false;
	selected = false;
	// start animation
	isAnimating = true;
	animationThread = CreateThread(NULL, NULL, &animationProcessFunction, NULL, NULL, NULL);
}
void endAnimation() {
	TerminateThread(animationThread, 0);
	previewFrame = currentFrame = (int)(startFrame + animateTime() / FrameDelay);
	isAnimating = false;
}

#pragma endregion


// ============================================ Rendering ============================================

#pragma region Rasterization functions

auto drawLine = [](vec2 p, vec2 q, COLORREF col, COLORREF& (canvas)(int, int) = Canvas, int MAX_W = _WIN_W, int MAX_H = _WIN_H) {
	vec2 d = q - p;
	double slope = d.y / d.x;
	if (abs(slope) <= 1.0) {
		if (p.x > q.x) std::swap(p, q);
		int x0 = max(0, int(p.x)), x1 = min(MAX_W - 1, int(q.x)), y;
		double yf = slope * x0 + (p.y - slope * p.x);
		for (int x = x0; x <= x1; x++) {
			y = (int)yf;
			if (y >= 0 && y < MAX_H) canvas(x, y) = col;
			yf += slope;
		}
	}
	else {
		slope = d.x / d.y;
		if (p.y > q.y) std::swap(p, q);
		int y0 = max(0, int(p.y)), y1 = min(MAX_H - 1, int(q.y)), x;
		double xf = slope * y0 + (p.x - slope * p.y);
		for (int y = y0; y <= y1; y++) {
			x = (int)xf;
			if (x >= 0 && x < MAX_W) canvas(x, y) = col;
			xf += slope;
		}
	}
};
auto drawCross = [&](vec2 p, double r, COLORREF Color = WHITE, COLORREF& (canvas)(int, int) = Canvas, int MAX_W = _WIN_W, int MAX_H = _WIN_H) {
	drawLine(p - vec2(r, 0), p + vec2(r, 0), Color, canvas, MAX_W, MAX_H);
	drawLine(p - vec2(0, r), p + vec2(0, r), Color, canvas, MAX_W, MAX_H);
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
auto drawTriangle = [](vec2 A, vec2 B, vec2 C, COLORREF col, bool stroke = false, COLORREF strokecol = WHITE) {
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
};
auto drawBox = [](vec2 Min, vec2 Max, COLORREF col = RED, COLORREF& (canvas)(int, int) = Canvas, int MAX_W = _WIN_W, int MAX_H = _WIN_H) {
	drawLine(vec2(Min.x, Min.y), vec2(Max.x, Min.y), col, canvas, MAX_W, MAX_H);
	drawLine(vec2(Max.x, Min.y), vec2(Max.x, Max.y), col, canvas, MAX_W, MAX_H);
	drawLine(vec2(Max.x, Max.y), vec2(Min.x, Max.y), col, canvas, MAX_W, MAX_H);
	drawLine(vec2(Min.x, Max.y), vec2(Min.x, Min.y), col, canvas, MAX_W, MAX_H);
};
auto fillBox = [](vec2 Min, vec2 Max, COLORREF col = RED, COLORREF& (canvas)(int, int) = Canvas, int MAX_W = _WIN_W, int MAX_H = _WIN_H) {
	int x0 = max((int)Min.x, 0), x1 = min((int)Max.x, MAX_W - 1);
	int y0 = max((int)Min.y, 0), y1 = min((int)Max.y, MAX_H - 1);
	for (int x = x0; x <= x1; x++) for (int y = y0; y <= y1; y++) canvas(x, y) = col;
};
auto drawSquare = [](vec2 C, double r, COLORREF col = ORANGE, COLORREF& (canvas)(int, int) = Canvas, int MAX_W = _WIN_W, int MAX_H = _WIN_H) {
	drawBox(C - vec2(r, r), C + vec2(r, r), col, canvas, MAX_W, MAX_H);
};
auto fillSquare = [](vec2 C, double r, COLORREF col = ORANGE, COLORREF& (canvas)(int, int) = Canvas, int MAX_W = _WIN_W, int MAX_H = _WIN_H) {
	fillBox(C - vec2(r, r), C + vec2(r, r), col, canvas, MAX_W, MAX_H);
};

auto drawLine_F = [](vec3 A, vec3 B, COLORREF col = WHITE) {
	double u = dot(Tr.p, A) + Tr.s, v = dot(Tr.p, B) + Tr.s;
	if (u > 0 && v > 0) { drawLine((Tr*A).xy(), (Tr*B).xy(), col); return; }
	if (u < 0 && v < 0) return;
	if (u < v) std::swap(A, B), std::swap(u, v);
	double t = u / (u - v) - 1e-6;
	B = A + (B - A)*t;
	drawLine((Tr*A).xy(), (Tr*B).xy(), col);
};
auto drawTriangle_F = [](vec3 A, vec3 B, vec3 C, COLORREF col) {
	double u = dot(Tr.p, A) + Tr.s, v = dot(Tr.p, B) + Tr.s, w = dot(Tr.p, C) + Tr.s;
	if (u > 0 && v > 0 && w > 0) { drawTriangle((Tr*A).xy(), (Tr*B).xy(), (Tr*C).xy(), col); return; }
	if (u < 0 && v < 0 && w < 0) return;
	// debug
};
auto drawCross3D = [&](vec3 P, double r, COLORREF col = WHITE) {
	r /= Unit;
	drawLine_F(P - vec3(r, 0, 0), P + vec3(r, 0, 0), col);
	drawLine_F(P - vec3(0, r, 0), P + vec3(0, r, 0), col);
	drawLine_F(P - vec3(0, 0, r), P + vec3(0, 0, r), col);
};
auto drawRod = [](vec3 A, vec3 B, double r, COLORREF col) {
	vec3 d = normalize(B - A);
	vec2 p0, p1; projRange_Cylinder(A, B, r, p0, p1);
	for (int i = max((int)p0.x, 0), im = min((int)p1.x, _WIN_W - 1); i <= im; i++)  // need to handle perspective issue
		for (int j = max((int)p0.y, 0), jm = min((int)p1.y, _WIN_H - 1); j <= jm; j++)
			if (intCylinder(A, B, r, CamP, scrDir(vec2(i, j))) > 0) Canvas(i, j) = col;
};

#pragma endregion


auto t0 = NTime::now();
const vec3 light = normalize(vec3(-0.5, 0.5, 1));


#define NFinger 16

void render_raycasting() {
	// initialization for ray intersection
	struct Capsule {
		vec3 A, B; double r;
	} Fingers[NFinger] = {
#define CS(d1,d2) Capsule{CPs[Hand::d1].P(), CPs[Hand::d2].P(), 0.1}
		CS(A0, A1), CS(A1, A2), CS(A2, A3),
		CS(B1, B2), CS(B2, B3), CS(B3, B4),
		CS(C1, C2), CS(C2, C3), CS(C3, C4),
		CS(D1, D2), CS(D2, D3), CS(D3, D4),
		CS(E0, E1), CS(E1, E2), CS(E2, E3), CS(E3, E4)
#undef CS
	};
	vec3 RMin(INFINITY), RMax(-INFINITY), rmin, rmax;
	for (int i = 0; i < NFinger; i++) {
		rangeCapsule(Fingers[i].A, Fingers[i].B, Fingers[i].r, rmin, rmax);
		RMin = pMin(RMin, rmin), RMax = pMax(RMax, rmax);
	}
	vec3 BoxC = 0.5*(RMin + RMax), BoxR = 0.5*(RMax - RMin);

	int x0 = 0, x1 = _WIN_W - 1, y0 = 0, y1 = _WIN_H - 1;
	for (int i = x0; i <= x1; i++) for (int j = y0; j <= y1; j++) {
		vec3 p = CamP, d = scrDir(vec2(i, j)), n;
		double t, mt = intBoxC(BoxR, CamP - BoxC, d);
		if (!isnan(mt)) {
			mt = NAN;
			// intersection
			for (int i = 0; i < NFinger; i++) {
				if ((t = intCapsule(Fingers[i].A, Fingers[i].B, Fingers[i].r, p, d)) > 0 && !(t > mt))
					mt = t, n = nCapsule(Fingers[i].A, Fingers[i].B, Fingers[i].r, p + t * d);
			}
			// lighting
			vec3 col(0.0);
			if ((t = mt) > 0.0) {
				p = p + t * d;
				d = d - n * (2.0*dot(d, n));
				// lighting
				vec3 bkg = vec3(0.2, 0.15, 0.1);
				vec3 dif = vec3(0.7, 0.65, 0.6) * max(dot(n, light), 0.0);
				vec3 spc = vec3(0.2) * pow(max(dot(d, light), 0.0), 5.0);
				col = bkg + dif + spc;
				// put the color on the screen
				byte* c = (byte*)&Canvas(i, j);
				c[0] = 255 * clamp(col.z, 0, 1), c[1] = 255 * clamp(col.y, 0, 1), c[2] = 255 * clamp(col.x, 0, 1);
			}
		}
	}
	//drawBox(p0, p1 + vec2(1), YELLOW);

}

void render() {
	// timer
	auto tt0 = NTime::now();

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
		//drawLine_F(vec3(0.0), vec3(0, 0, R), ROYALBLUE);
	}

	render_raycasting();

#if 1
	// scene
#define DW(p,q) drawLine_F(CPs[Hand::p].P(),CPs[Hand::q].P())
	DW(A0, A1), DW(A1, A2), DW(A2, A3);
	DW(B1, B2), DW(B2, B3), DW(B3, B4);
	DW(C1, C2), DW(C2, C3), DW(C3, C4);
	DW(D1, D2), DW(D2, D3), DW(D3, D4);
	DW(E0, E1), DW(E1, E2), DW(E2, E3), DW(E3, E4);
#undef DW

	// 3D cursor
	if (selected && !mouse_down_T && !isAnimating) {
		double Unit = dot(CPC, Tr.p) + Tr.s;
		double sR = selRadius * Unit, sL = selLength * Unit, sA = selAxisRatio * sR;
		if (mouse_down) {
			if (moveAlong == xAxis) drawLine_F(CPC - 1e4*veci, CPC + 1e4*veci, 0x80FF00);
			if (moveAlong == yAxis) drawLine_F(CPC - 1e4*vecj, CPC + 1e4*vecj, 0x80FF00);
			if (moveAlong == zAxis) drawLine_F(CPC - 1e4*veck, CPC + 1e4*veck, 0x80FF00);
		}
		drawRod(CPC, CPC + vec3(sL, 0, 0), sA, moveAlong == xAxis ? YELLOW : RED);
		drawRod(CPC, CPC + vec3(0, sL, 0), sA, moveAlong == yAxis ? YELLOW : GREEN);
		drawRod(CPC, CPC + vec3(0, 0, sL), sA, moveAlong == zAxis ? YELLOW : BLUE);
		if (moveAlong == unlimited) fillCircle((Tr*CPC).xy(), selRadius, YELLOW);
		drawCircle((Tr*CPC).xy(), selRadius, LIME);
	}

	// control points
	for (int i = 0; i < NCtrPs; i++) {
		if (CPs[i].selected) fillSquare((Tr*CPs[i].P()).xy(), selRadiusP, mouse_down && moveAlong != none ? YELLOW : ORANGE);
		drawSquare((Tr*CPs[i].P()).xy(), selRadiusP, i == HObject ? RED : ORANGE);
	}

#endif

	// timer
	auto t1 = NTime::now();
	sprintf(text, "[%d×%d]  %.1fms (%.1ffps)\n", _WIN_W, _WIN_H, 1000.0*std::chrono::duration<double>(t1 - tt0).count(), 1. / std::chrono::duration<double>(t1 - t0).count());
	SetWindowTextA(_HWND, text);
	t0 = t1;
}

void render_t() {
	for (int i = 0, l = _WIN_T_W * _WIN_T_H; i < l; i++) _WINIMG_T[i] = 0;

	// highlight mouse-hower frame
	int f = (int)getFrame(CursorT.x); double f0 = FrameToCoord(f);
	for (int x = max((int)f0, 0), x1 = min((int)(f0 + UnitT), _WIN_T_W); x < x1; x++) {
		for (int y = 0; y < _WIN_T_H; y++) CanvasT(x, y) = 0x101418;
	}

	// highlight selected points
	for (int i = 0; i < NCtrPs; i++) if (CPs[i].selected) {
		double y0 = ObjectIDToCoord(i), y1 = y0 + UnitTV;
		for (int y = max((int)y0, 0), ym = min((int)y1, _WIN_T_H); y < ym; y++) {
			for (int x = 0; x < _WIN_T_W; x++) CanvasT(x, y) = INDIGO;
		}
	}

	// draw axis and grid
	{
		double f0 = LFrame, f1 = f0 + _WIN_T_W / UnitT;
		for (int i = (int)ceil(f0); i < f1; i++) {
			int x = (int)((i - f0)*UnitT);
			for (int j = 0; j < _WIN_T_H; j++) CanvasT(x, j) = i % FPS ? 0x404040 : 0xA0A0A0;
		}
	}

	// highlight current frame
	f0 = FrameToCoord(currentFrame);
	for (int x = max((int)f0, 0), x1 = min((int)(f0 + UnitT), _WIN_T_W); x < x1; x++) {
		for (int y = 0; y < _WIN_T_H; y++) CanvasT(x, y) = NAVY;
	}
	// highlight current time
	f0 = previewFrame - LFrame, f = (int)(f0 * UnitT);
	if (f >= 0 && f < _WIN_T_W) for (int y = 0; y < _WIN_T_H; y++) CanvasT(f, y) = LIME;

	// draw control points
	{
		int last = lastFrame(), last_x = (last - LFrame)*UnitT;
		if (last_x >= 0 && last_x < _WIN_T_W) for (int j = 0; j < _WIN_T_H; j++) CanvasT(last_x, j) = MAGENTA;  // highlight the last frame
		for (int i = BObject; i < NCtrPs; i++) {
			auto *P = &CPs[i].keyFrames;
			int back = P->back().F;
			drawLine(timeAxisSquareCenter(0, i), timeAxisSquareCenter(back, i), DARKGREEN, CanvasT, _WIN_T_W, _WIN_T_H);
			if (pointOnMove[i] != -1) {
				fillSquare(timeAxisSquareCenter(pointOnMove[i], i), rdrRadiusT, MAGENTA, CanvasT, _WIN_T_W, _WIN_T_H);
			}
			for (int d = 0, n = P->size(); d < n; d++) {
				drawSquare(timeAxisSquareCenter(P->at(d).F, i), rdrRadiusT, i == HObject ? RED : d + 1 == n ? YELLOW : ORANGE, CanvasT, _WIN_T_W, _WIN_T_H);
			}
		}
	}
}


#pragma region Write File

#define GIF_FLIP_VERT
#include "libraries\gif.h"	// animated gif writer, https://github.com/charlietangora/gif-h

void WriteImage() {
	_WIN_W = 600, _WIN_H = 400;
	_WINIMG = new COLORREF[_WIN_W*_WIN_H];
	GifWriter gif;
	GifBegin(&gif, "D:\\ASM Hand.gif", _WIN_W, _WIN_H, 4);
	readFile(L"D:\\ASM Hand.txt");
	for (currentFrame = 0, endFrame = lastFrame(); currentFrame < endFrame; currentFrame++) {
		previewFrame = currentFrame;
		render();
		for (int i = 0, l = _WIN_W * _WIN_H; i < l; i++) std::swap(((byte*)&_WINIMG[i])[0], ((byte*)&_WINIMG[i])[2]);
		GifWriteFrame(&gif, (uint8_t*)_WINIMG, _WIN_W, _WIN_H, 4);
	}
	GifEnd(&gif);
}

#pragma endregion This part is to be manually called on compiler


// ============================================== User ==============================================


// This function is called once before the windows are created
// It will also be called after a file is opened
// Only use to initialize variables
void Init() {
	for (int i = 0; i < NCtrPs; i++) pointOnMove[i] = -1;
	for (int i = 0; i < NCtrPs; i++) CPs[i].sortFrames();
	previewFrame = currentFrame;
	dbgprint("Init\n");
}


// Key events shared by both windows
void keyDownShared(WPARAM _KEY) {
	if (_KEY == VK_CONTROL) Ctrl = true;
	else if (_KEY == VK_SHIFT) Shift = true;
	else if (_KEY == VK_MENU) Alt = true;
}
void keyUpShared(WPARAM _KEY) {
	if (_KEY == VK_CONTROL) Ctrl = false;
	else if (_KEY == VK_SHIFT) Shift = false;
	else if (_KEY == VK_MENU) Alt = false;

	// Space: preview animation
	if (_KEY == VK_SPACE) {
		if (isAnimating) endAnimation();
		else startAnimation(currentFrame >= lastFrame() ? 0 : currentFrame);
	}

	// Period/Decimal: move viewport center to the location of 3D cursor
	if (_KEY == VK_OEM_PERIOD || _KEY == VK_DECIMAL) {
		centerViewport();
	}

	// Tab: insert keyframes
	if (_KEY == VK_TAB) {
		for (int i = 0; i < NCtrPs; i++) if (CPs[i].selected) {
			if (!CPs[i].existFrame(currentFrame)) CPs[i].getP(currentFrame, false);
		}
		updateCursorPosition();
	}

	// Ctrl combination keys
	if (Ctrl) {
		if (_KEY == 'A') selectAll();
		else if (_KEY == 'I') inverseSelection();
		if (_KEY == 'S') {  // Ctrl+S save file
			if (!saveFileUserEntry()) MessageBeep(MB_ICONSTOP);
			Ctrl = false;
		}
		else if (_KEY == 'O') {  // Ctrl+O open file
			if (!readFileUserEntry()) MessageBeep(MB_ICONSTOP);
			Ctrl = false;
		}
	}
}


// Viewport window

void WindowResize(int _oldW, int _oldH, int _W, int _H) {
	if (_W*_H == 0 || _oldW * _oldH == 0) return;  // window is minimized
	double pw = _oldW, ph = _oldH, w = _W, h = _H;
	double s = sqrt((w * h) / (pw * ph));
	Unit *= s, dist /= s;
}
void WindowClose() {
	if (isAnimating) endAnimation();
}

void MouseWheel(int _DELTA) {
	if (Shift) {
		double s = exp(-0.001*_DELTA);
		dist *= s;
	}
	else {
		// zoom
		double s = exp(0.001*_DELTA);
		double D = length(vec2(_WIN_W, _WIN_H)), Max = D, Min = 0.015*D;
		if (Unit * s > Max) s = Max / Unit;
		else if (Unit * s < Min) s = Min / Unit;  // clamp zooming depth
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

	// drag to rotate scene
	if (mouse_down && moveAlong == none) {
		if (Shift) {
			ry += 0.002*D.y;
		}
		else {
			vec2 d = 0.01*D;
			rz -= cos(ry)*d.x + sin(ry)*d.y, rx -= -sin(ry)*d.x + cos(ry)*d.y;
			//rz -= d.x, rx -= d.y;
		}
	}

	if (selected) {  // mouse hover 3D cursor
		getScreen(CamP, ScrO, ScrA, ScrB);
		if (mouse_down) {
			// dragging 3D cursor
			vec3 vecd(0.0);  // record the displacement of the cursor
			if (moveAlong >= xAxis && moveAlong <= zAxis) {
				vecd = moveAlong == xAxis ? veci : moveAlong == yAxis ? vecj : veck;
				double t0 = closestPoint(CPC, vecd, CamP, scrDir(P0));
				double t1 = closestPoint(CPC, vecd, CamP, scrDir(P));
				CPC += (vecd = (t1 - t0)*vecd);
			}
			// update position of points
			if (vecd != vec3(0.0)) {
				for (int i = 0; i < NCtrPs; i++) {
					if (CPs[i].selected) CPs[i].getP() += vecd;
				}
			}
			updateCursorPosition();  // not necessary just for safe
		}
		else {
			// test if which part the mouse hover 3D cursor
			vec3 p, d; getRay(Cursor, p, d);
			double Unit = dot(CPC, Tr.p) + Tr.s;
			double sR = selRadius * Unit, sL = selLength * Unit, sA = selAxisRatio * sR;
			double t, mt = intSphere(CPC, sR, p, d);
			moveDirection dir = unlimited;
			if (0.0*mt != 0.0) mt = INFINITY, dir = none;
			t = intCylinder(CPC, CPC + vec3(sL, 0, 0), sA, p, d); if (t < mt) mt = t, dir = xAxis;
			t = intCylinder(CPC, CPC + vec3(0, sL, 0), sA, p, d); if (t < mt) mt = t, dir = yAxis;
			t = intCylinder(CPC, CPC + vec3(0, 0, sL), sA, p, d); if (t < mt) mt = t, dir = zAxis;
			moveAlong = dir;
		}
	}
	else moveAlong = none;

	HObject = -1;
}
void MouseUpL(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	bool moved = (int)length(clickCursor - Cursor) != 0;   // be careful: coincidence
	mouse_down = false;

	if (!moved) {  // click
		if (moveAlong == none) {
			// click to select/deselect points
			int N = NSelectedPoints(), NHover = 0;
			bool hover[NCtrPs];
			for (int i = 0; i < NCtrPs; i++) {
				vec3 P = Tr * CPs[i].P();
				NHover += hover[i] = sdBox(P.xy() - Cursor, vec2(selRadiusP)) < 0.;
			}
			for (int i = 0; i < NCtrPs; i++) {
				if (Shift) CPs[i].selected ^= hover[i];
				else CPs[i].selected = hover[i];
			}
			updateCursorPosition();
		}
	}

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


// Time axis window

void WindowResizeT(int _oldW, int _oldH, int _W, int _H) {
	if (_H > UnitTV * NCtrPs) UnitTV = _H / (double)NCtrPs;
}
void WindowCloseT() {
	if (isAnimating) endAnimation();
}

void MouseWheelT(int _DELTA) {
	if (Ctrl) {  // zoom object axis
		double s = exp(0.0005*_DELTA);
		double SCMin = max(_WIN_T_H / (double)NCtrPs, 12.0);
		if (UnitTV*s > 30.0) s = 30.0 / UnitTV;
		if (UnitTV*s < SCMin) s = SCMin / UnitTV;
		UnitTV *= s;
	}
	else if (Shift) {  // zoom time axis
		double s = exp(0.0005*_DELTA);
		if (UnitT*s > 25.0) s = 25.0 / UnitT;
		if (UnitT*s < 5.0) s = 5.0 / UnitT;
		double CF = LFrame + (CursorT.x / UnitT);
		UnitT *= s;
		LFrame = CF - CursorT.x / UnitT;
	}
	else if (Alt) {  // scroll time axis
		LFrame -= 0.25*_DELTA / UnitT;
	}
	else {
		BObject += _DELTA / abs(_DELTA);
		BObject = clamp(BObject, 0, NCtrPs - int(_WIN_T_H / UnitTV));
	}
	LFrame = max(LFrame, 0);
}
void MouseDownLT(int _X, int _Y) {
	clickCursorT = CursorT = vec2(_X, _Y);
	mouse_down_T = true;
	previewFrame = getFrame(CursorT.x), currentFrame = (int)previewFrame;

	// ready to drag key frames
	if (currentFrame != 0 && (int)getFrame(CursorT.x) == currentFrame) {
		for (int i = 0; i < NCtrPs; i++) if (CPs[i].selected) {
			if (CPs[i].existFrame(currentFrame)) pointOnMove[i] = currentFrame;
			else pointOnMove[i] = -1;
		}
	}
}
void MouseMoveT(int _X, int _Y) {
	CursorT = vec2(_X, _Y);
	HObject = clamp(getHoverObject(CursorT.y), 0, NCtrPs - 1);
	if (mouse_down_T) {
		previewFrame = getFrame(CursorT.x), currentFrame = (int)previewFrame;

		// drag keyframes
		if (currentFrame > 0) {
			for (int i = 0; i < NCtrPs; i++) if (pointOnMove[i] != -1) {
				if (CPs[i].setFrame(pointOnMove[i], currentFrame)) pointOnMove[i] = currentFrame;
			}
		}
	}
}
void MouseUpLT(int _X, int _Y) {
	CursorT = vec2(_X, _Y);
	bool moved = (int)length(clickCursorT - CursorT) != 0;   // be careful: coincidence
	mouse_down_T = false;

	if (!moved) {
		// click to select edit frame
		currentFrame = (int)getFrame(CursorT.x);

		// select point from time axis
		if (CPs[HObject].existFrame(currentFrame) && sdBox(CursorT - timeAxisSquareCenter(currentFrame, HObject), vec2(rdrRadiusT)) < 0.) {
			CPs[HObject].selected ^= 1;
		}
	}

	// stop point dragging
	for (int i = 0; i < NCtrPs; i++) pointOnMove[i] = -1;

	previewFrame = currentFrame;
	updateCursorPosition();
}
void MouseDownRT(int _X, int _Y) {}
void MouseUpRT(int _X, int _Y) {
	// right click to pin/unpin window
	bool topmost = GetWindowLong(_HWND_T, GWL_EXSTYLE) & WS_EX_TOPMOST;
	SetWindowPos(_HWND_T, topmost ? HWND_NOTOPMOST : HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
	sprintf(text, "%s - %s", WIN_NAME_T, topmost ? "unpinned" : "pinned");
	SetWindowTextA(_HWND_T, text);
}
void KeyDownT(WPARAM _KEY) {
	keyDownShared(_KEY);
}
void KeyUpT(WPARAM _KEY) {
	keyUpShared(_KEY);

	// Home/End: go to the beginning/end
	if (_KEY == VK_HOME) LFrame = 0, currentFrame = previewFrame = 0;
	if (_KEY == VK_END) currentFrame = previewFrame = lastFrame(), LFrame = max(lastFrame() - _WIN_T_W / UnitT, 0);

	// Delete: delete keyframes
	if (_KEY == VK_DELETE || _KEY == VK_BACK) {
		if (currentFrame != 0) {
			for (int i = 0; i < NCtrPs; i++) if (CPs[i].selected) {
				CPs[i].deleteFrame(currentFrame);
				CPs[i].selected = false;
			}
			updateCursorPosition();
		}
	}

}



// ============================================== File ==============================================

/*
	// Text File

	# _WIN_W _WIN_H
	Center.x Center.y Center.z rz rx ry dist Unit
	NCtrPs FPS currentFrame
	keyFrames.size() F P.x P.y P.z ......
	[new line]

	File starts with character '#'
	The program searches for that character after opening the file.

	There will be a basic error handling for file open error, but no further handling on file format error.
	Opening file will clear the current scene. If errors occur or program crashes due to file format error, restart this program.

*/

#include <iostream>
#include <fstream>

bool saveFile(const WCHAR filename[]) {
	FILE *fp = _wfopen(filename, L"ab");
	if (fp == 0) return false;

	fprintf(fp, "# %d %d\n", _WIN_W, _WIN_H);
	fprintf(fp, "%.8lg %.8lg %.8lg ", Center.x, Center.y, Center.z);
	rz = mod(rz, 2.0*PI), rx = mod(rx, 2.0*PI), ry = mod(ry, 2.0*PI);
	fprintf(fp, "%.8lg %.8lg %.8lg %.8lg %.8lg\n", rz, rx, ry, dist, Unit);
	fprintf(fp, "%d %d %d\n", NCtrPs, FPS, currentFrame);
	for (int i = 0; i < NCtrPs; i++) {
		const auto *P = &CPs[i].keyFrames;
		int n = P->size();
		fprintf(fp, "%d ", n);
		for (int i = 0; i < n; i++) {
			fprintf(fp, "%d ", P->at(i).F);
			fprintf(fp, "%.8lg %.8lg %.8lg", P->at(i).P.x, P->at(i).P.y, P->at(i).P.z);
			if (i + 1 != n) fprintf(fp, " ");
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n");
	fclose(fp);
	return true;
}

bool readFile(const WCHAR filename[]) {
	FILE *fp = _wfopen(filename, L"rb");
	if (fp == 0) return false;
	int c; while ((c = fgetc(fp)) != '#') if (c == EOF) return false;

	int WinW, WinH; fscanf(fp, "%d %d", &WinW, &WinH);
	fscanf(fp, "%lf %lf %lf ", &Center.x, &Center.y, &Center.z);
	fscanf(fp, "%lf %lf %lf %lf %lf\n", &rz, &rx, &ry, &dist, &Unit);
	WindowResize(WinW, WinH, _WIN_W, _WIN_H);
	int N, fps; fscanf(fp, "%d %d %d\n", &N, &fps, &currentFrame);
	if (N != NCtrPs) throw "OPEN FILE ERROR: NCtrPs\n";
	if (fps != FPS) throw "OPEN FILE ERROR: FPS\n";
	for (int i = 0; i < NCtrPs; i++) {
		auto *P = &CPs[i].keyFrames;
		P->clear();
		int n; fscanf(fp, "%d ", &n);
		for (int i = 0; i < n; i++) {
			int F; vec3 C;
			fscanf(fp, "%d ", &F);
			fscanf(fp, "%lf %lf %lf", &C.x, &C.y, &C.z);
			P->push_back(ControlPoint::Point{ C, F });
		}
	}
	fclose(fp);
	Init();
	return true;
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
