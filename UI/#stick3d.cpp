

// ========================================= Win32 Standard =========================================

#pragma region Windows

#include <Windows.h>
#include <windowsx.h>
#include <tchar.h>

#define WIN_NAME "Spline Tester"
#define WinW_Default 600
#define WinH_Default 400
#define MIN_WinW 400
#define MIN_WinH 300
#define MAX_WinW 3840
#define MAX_WinH 2160

// implement the following functions:
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

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
#define _RDBK { HDC hdc = GetDC(_HWND), HImgMem = CreateCompatibleDC(hdc); HBITMAP hbmOld = (HBITMAP)SelectObject(HImgMem, _HIMG); render(); BitBlt(hdc, 0, 0, _WIN_W, _WIN_H, HImgMem, 0, 0, SRCCOPY); SelectObject(HImgMem, hbmOld), DeleteDC(HImgMem), DeleteDC(hdc); } break;
	switch (message) {
	case WM_CREATE: { RECT Client; GetClientRect(hWnd, &Client); _WIN_W = Client.right, _WIN_H = Client.bottom; WindowCreate(_WIN_W, _WIN_H); break; }
	case WM_CLOSE: { DestroyWindow(hWnd); WindowClose(); return 0; } case WM_DESTROY: { PostQuitMessage(0); return 0; }
	case WM_MOVE:; case WM_SIZE: {
		RECT Client; GetClientRect(hWnd, &Client); WindowResize(_WIN_W, _WIN_H, Client.right, Client.bottom); _WIN_W = Client.right, _WIN_H = Client.bottom;
		BITMAPINFO bmi; bmi.bmiHeader.biSize = sizeof(BITMAPINFO), bmi.bmiHeader.biWidth = Client.right, bmi.bmiHeader.biHeight = Client.bottom, bmi.bmiHeader.biPlanes = 1, bmi.bmiHeader.biBitCount = 32; bmi.bmiHeader.biCompression = BI_RGB, bmi.bmiHeader.biSizeImage = 0, bmi.bmiHeader.biXPelsPerMeter = bmi.bmiHeader.biYPelsPerMeter = 0, bmi.bmiHeader.biClrUsed = bmi.bmiHeader.biClrImportant = 0; bmi.bmiColors[0].rgbBlue = bmi.bmiColors[0].rgbGreen = bmi.bmiColors[0].rgbRed = bmi.bmiColors[0].rgbReserved = 0;
		if (_HIMG != NULL) DeleteObject(_HIMG); HDC hdc = GetDC(hWnd); _HIMG = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, (void**)&_WINIMG, NULL, 0); DeleteDC(hdc); _RDBK }
	case WM_GETMINMAXINFO: { LPMINMAXINFO lpMMI = (LPMINMAXINFO)lParam; lpMMI->ptMinTrackSize.x = MIN_WinW, lpMMI->ptMinTrackSize.y = MIN_WinH, lpMMI->ptMaxTrackSize.x = MAX_WinW, lpMMI->ptMaxTrackSize.y = MAX_WinH; break; }
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
	WNDCLASSEX wc; wc.cbSize = sizeof(WNDCLASSEX), wc.style = 0, wc.lpfnWndProc = WndProc, wc.cbClsExtra = wc.cbWndExtra = 0, wc.hInstance = hInstance; wc.hIcon = wc.hIconSm = 0, wc.hCursor = LoadCursor(NULL, IDC_ARROW), wc.hbrBackground = CreateSolidBrush(RGB(0, 0, 0)), wc.lpszMenuName = NULL, wc.lpszClassName = _T(WIN_NAME); if (!RegisterClassEx(&wc)) return -1;
	//_HWND = CreateWindowEx(WS_EX_TOPMOST, _T(WIN_NAME), _T(WIN_NAME), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, WinW_Default, WinH_Default, NULL, NULL, hInstance, NULL); ShowWindow(_HWND, nCmdShow); UpdateWindow(_HWND);
	_HWND = CreateWindow(_T(WIN_NAME), _T(WIN_NAME), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, WinW_Default, WinH_Default, NULL, NULL, hInstance, NULL); ShowWindow(_HWND, nCmdShow); UpdateWindow(_HWND);
	MSG message; while (GetMessage(&message, 0, 0, 0)) { TranslateMessage(&message); DispatchMessage(&message); } return (int)message.wParam;
}

// debug
#define dbgprint(format, ...) { wchar_t buf[0x4FFF]; swprintf(buf, 0x4FFF, _T(format), ##__VA_ARGS__); OutputDebugStringW(buf); }

#pragma endregion


// ================================== Vector Classes/Functions ==================================

#pragma region Vector & Matrix Classes/Functions

#include <cmath>
#pragma warning(disable: 4244 4305)

#define PI 3.1415926535897932384626
#define mix(x,y,a) ((x)*(1.0-(a))+(y)*(a))
#define clamp(x,a,b) ((x)<(a)?(a):(x)>(b)?(b):(x))

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
	//vec2 operator == (const vec2 &v) const { return x == v.x && y == v.y; }
	//vec2 operator != (const vec2 &v) const { return x != v.x || y != v.y; }
	void operator += (const vec2 &v) { x += v.x, y += v.y; }
	void operator -= (const vec2 &v) { x -= v.x, y -= v.y; }
	void operator *= (const vec2 &v) { x *= v.x, y *= v.y; }
	vec2 operator / (const vec2 &v) const { return vec2(x / v.x, y / v.y); }
	void operator /= (const vec2 &v) { x /= v.x, y /= v.y; }
	friend vec2 operator * (const double &a, const vec2 &v) { return vec2(a*v.x, a*v.y); }
	void operator *= (const double &a) { x *= a, y *= a; }
	vec2 operator / (const double &a) const { return vec2(x / a, y / a); }
	void operator /= (const double &a) { x /= a, y /= a; }
	friend vec2 abs(const vec2 &a) { return vec2(abs(a.x), abs(a.y)); }
#endif
	vec2 yx() const { return vec2(y, x); }
	vec2 rot() const { return vec2(-y, x); }
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
	vec3 operator * (const double &k) const { return vec3(k * x, k * y, k * z); }
	double sqr() { return x * x + y * y + z * z; } 	// non-standard
	friend double length(vec3 v) { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
	friend vec3 normalize(vec3 v) { return v * (1. / sqrt(v.x*v.x + v.y*v.y + v.z*v.z)); }
	friend double dot(vec3 u, vec3 v) { return u.x*v.x + u.y*v.y + u.z*v.z; }
	friend vec3 cross(vec3 u, vec3 v) { return vec3(u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x); }
#if 1
	void operator += (const vec3 &v) { x += v.x, y += v.y, z += v.z; }
	void operator -= (const vec3 &v) { x -= v.x, y -= v.y, z -= v.z; }
	void operator *= (const vec3 &v) { x *= v.x, y *= v.y, z *= v.z; }
	vec3 operator / (const vec3 &v) const { return vec3(x / v.x, y / v.y, z / v.z); }
	void operator /= (const vec3 &v) { x /= v.x, y /= v.y, z /= v.z; }
	friend vec3 operator * (const double &a, const vec3 &v) { return vec3(a*v.x, a*v.y, a*v.z); }
	void operator *= (const double &a) { x *= a, y *= a, z *= a; }
	vec3 operator / (const double &a) const { return vec3(x / a, y / a, z / a); }
	void operator /= (const double &a) { x /= a, y /= a, z /= a; }
	friend vec3 abs(const vec3 &a) { return vec3(abs(a.x), abs(a.y), abs(a.z)); }
#endif
	vec2 xy() const { return vec2(x, y); }
	vec2 xz() const { return vec2(x, z); }
	vec2 yz() const { return vec2(y, z); }
};

const vec3 veci(1, 0, 0), vecj(0, 1, 0), veck(0, 0, 1);

#if 0
class mat3 {
public:
	vec3 _0, _1, _2;  // columns
	mat3() :_0(1, 0, 0), _1(0, 1, 0), _2(0, 0, 1) {}
	mat3(const double &s) :_0(s, 0, 0), _1(0, s, 0), _2(0, 0, s) {}
	mat3(const vec3 &d) :_0(d.x, 0, 0), _1(0, d.y, 0), _2(0, 0, d.z) {}
	mat3(const vec3 &u, const vec3 &v, const vec3 &w) :_0(u), _1(v), _2(w) {}

};
#endif

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


#pragma endregion


// ======================================== Data / Parameters ========================================

#include <stdio.h>
#pragma warning(disable: 4996)

#pragma region Global Variables

// window parameters
char text[64];	// window title
vec3 Center(0.0);
double rz = 0.25*PI, rx = 0.2*PI, inv_d = 0.08, Unit = 50.0;  // spherical, inverse of camera distance, scale to screen
struct Affine { vec3 u, v, w, t, p; double s; };  // 4x4 matrix: first row, second row, third row, translation, perspective, scale
vec3 operator * (Affine T, vec3 p) {
	vec3 q = vec3(dot(T.u, p), dot(T.v, p), dot(T.w, p)) + T.t;
	double d = dot(T.p, p) + T.s;
	//return d < 0.0 ? vec3(NAN) : q * (1. / d);
	return q * (1. / d);
}
Affine operator * (const Affine &A, const Affine &B) {
	Affine R;
	R.u = A.u.x*B.u + A.u.y*B.v + A.u.z*B.w + A.t.x*B.p;
	R.v = A.v.x*B.u + A.v.y*B.v + A.v.z*B.w + A.t.y*B.p;
	R.w = A.w.x*B.u + A.w.y*B.v + A.w.z*B.w + A.t.z*B.p;
	R.t = vec3(dot(A.u, B.t), dot(A.v, B.t), dot(A.w, B.t)) + A.t*B.s;
	R.p = vec3(A.p.x*B.u.x + A.p.y*B.v.x + A.p.z*B.w.x, A.p.x*B.u.y + A.p.y*B.v.y + A.p.z*B.w.y, A.p.x*B.u.z + A.p.y*B.v.z + A.p.z*B.w.z) + B.p*A.s;
	R.s = A.s*B.s;
	return R;
}
Affine calcMat(double rx, double rz) {
	double cx = cos(rx), sx = sin(rx), cz = cos(rz), sz = sin(rz);
	Affine D{ veci, vecj, veck, -Center, vec3(0), 1.0 };  // global translation
	Affine R{ vec3(-sz, cz, 0), vec3(-cz * sx, -sz * sx, cx), vec3(-cz * cx, -sz * cx, -sx), vec3(0), vec3(0), 1.0 };  // rotation
	Affine T{ veci, vecj, veck, vec3(0.5*vec2(_WIN_W, _WIN_H), 0.0), vec3(0), 1.0 };  // translation
	Affine P{ veci, vecj, veck, vec3(0), vec3(0, 0, inv_d), 1.0 };  // perspective
	Affine S{ veci, vecj, veck, vec3(0), vec3(0), 1.0 / Unit };  // scale
	return T * S * P * R * D;
}


// user parameters
vec2 Cursor = vec2(0, 0), clickCursor;
bool mouse_down = false;
bool Ctrl = false, Shift = false, Alt = false;

#include <chrono>
typedef std::chrono::high_resolution_clock NTime;
NTime::time_point _Global_Timer = NTime::now();
#define iTime std::chrono::duration<double>(NTime::now()-_Global_Timer).count()

#pragma endregion


#include <vector>



// ============================================ Rendering ============================================

auto t0 = NTime::now();


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
auto drawCross = [&](vec2 p, double r, COLORREF Color) {
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


void render() {
	// debug
	auto t1 = NTime::now();
	double dt = std::chrono::duration<double>(t1 - t0).count();
	dbgprint("[%d×%d] time elapsed: %.1fms (%.1ffps)\n", _WIN_W, _WIN_H, 1000.0*dt, 1. / dt);
	t0 = t1;

	// initialize window
	for (int i = 0, l = _WIN_W * _WIN_H; i < l; i++) _WINIMG[i] = 0;
	Affine Tr = calcMat(rx, rz);

	auto drawLine_F = [&](vec3 A, vec3 B, COLORREF col) {
		double u = dot(Tr.p, A) + Tr.s, v = dot(Tr.p, B) + Tr.s;
		if (u > 0 && v > 0) { drawLine((Tr*A).xy(), (Tr*B).xy(), col); return; }
		if (u < 0 && v < 0) return;
		if (u < v) std::swap(A, B), std::swap(u, v);
		double t = u / (u - v) - 0.001;
		B = A + (B - A)*t;
		drawLine((Tr*A).xy(), (Tr*B).xy(), col);
	};

	// axis and grid
	{
		const double R = 20.0;
		for (int i = -R; i <= R; i++) {
			drawLine_F(vec3(-R, i, 0), vec3(R, i, 0), 0x404040);
			drawLine_F(vec3(i, -R, 0), vec3(i, R, 0), 0x404040);
		}
		drawLine_F(vec3(0.0), vec3(R, 0, 0), RED);
		drawLine_F(vec3(0.0), vec3(0, R, 0), GREEN);
		drawLine_F(vec3(0.0), vec3(0, 0, R), BLUE);
	}


	sprintf(text, "");
	SetWindowTextA(_HWND, text);
}


// ============================================== User ==============================================


void WindowCreate(int _W, int _H) {

}
void WindowResize(int _oldW, int _oldH, int _W, int _H) {
	if (_W*_H == 0 || _oldW * _oldH == 0) return;
	double pw = _oldW, ph = _oldH, w = _W, h = _H;
	double s = sqrt((w * h) / (pw * ph));
	Unit *= s;
}
void WindowClose() {

}

void MouseMove(int _X, int _Y) {
	vec2 P0 = Cursor, P = vec2(_X, _Y), D = P - P0;
	Cursor = P;
	if (mouse_down) {
		rz -= 0.01*D.x;
		rx -= 0.01*D.y;
	}
}

void MouseWheel(int _DELTA) {
	double s = exp(0.001*_DELTA);
	double D = length(vec2(_WIN_W, _WIN_H)), Max = D, Min = 0.02*D;
	if (Unit * s > Max) s = Max / Unit;
	else if (Unit * s < Min) s = Min / Unit;
	Unit *= s;
}

void MouseDownL(int _X, int _Y) {
	clickCursor = Cursor = vec2(_X, _Y);
	mouse_down = true;
}

void MouseUpL(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	bool moved = (int)length(clickCursor - Cursor) != 0;
	mouse_down = false;

}

void MouseDownR(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
}

void MouseUpR(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
}

void KeyDown(WPARAM _KEY) {
	if (_KEY == VK_CONTROL) Ctrl = true;
	else if (_KEY == VK_SHIFT) Shift = true;
	else if (_KEY == VK_MENU) Alt = true;
}

void KeyUp(WPARAM _KEY) {

}

