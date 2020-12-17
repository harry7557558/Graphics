// Graphing Implicit Curve


/* ==================== User Instructions ====================

 *  Move View:                  drag background
 *  Zoom:                       mouse scroll, hold shift to lock center
 */



 // ========================================= Win32 GUI =========================================

#pragma region Windows

#include <Windows.h>
#include <windowsx.h>
#include <tchar.h>

#define WIN_NAME "Implicit curve visualization"
#define WinW_Default 600
#define WinH_Default 400
#define MIN_WinW 400
#define MIN_WinH 300

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

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
#define _RDBK { HDC hdc = GetDC(_HWND), HImgMem = CreateCompatibleDC(hdc); HBITMAP hbmOld = (HBITMAP)SelectObject(HImgMem, _HIMG); render(); BitBlt(hdc, 0, 0, _WIN_W, _WIN_H, HImgMem, 0, 0, SRCCOPY); SelectObject(HImgMem, hbmOld), DeleteDC(HImgMem), DeleteDC(hdc); } break;
	switch (message) {
	case WM_CREATE: { RECT Client; GetClientRect(hWnd, &Client); _WIN_W = Client.right, _WIN_H = Client.bottom; WindowCreate(_WIN_W, _WIN_H); break; }
	case WM_CLOSE: { DestroyWindow(hWnd); WindowClose(); return 0; } case WM_DESTROY: { PostQuitMessage(0); return 0; }
	case WM_MOVE:; case WM_SIZE: {
		RECT Client; GetClientRect(hWnd, &Client); WindowResize(_WIN_W, _WIN_H, Client.right, Client.bottom); _WIN_W = Client.right, _WIN_H = Client.bottom;
		BITMAPINFO bmi; bmi.bmiHeader.biSize = sizeof(BITMAPINFO), bmi.bmiHeader.biWidth = Client.right, bmi.bmiHeader.biHeight = Client.bottom, bmi.bmiHeader.biPlanes = 1, bmi.bmiHeader.biBitCount = 32; bmi.bmiHeader.biCompression = BI_RGB, bmi.bmiHeader.biSizeImage = 0, bmi.bmiHeader.biXPelsPerMeter = bmi.bmiHeader.biYPelsPerMeter = 0, bmi.bmiHeader.biClrUsed = bmi.bmiHeader.biClrImportant = 0; bmi.bmiColors[0].rgbBlue = bmi.bmiColors[0].rgbGreen = bmi.bmiColors[0].rgbRed = bmi.bmiColors[0].rgbReserved = 0;
		if (_HIMG != NULL) DeleteObject(_HIMG); HDC hdc = GetDC(hWnd); _HIMG = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, (void**)&_WINIMG, NULL, 0); DeleteDC(hdc); _RDBK }
	case WM_GETMINMAXINFO: { LPMINMAXINFO lpMMI = (LPMINMAXINFO)lParam; lpMMI->ptMinTrackSize.x = MIN_WinW, lpMMI->ptMinTrackSize.y = MIN_WinH; break; }
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

// debug
#define dbgprint(format, ...) { wchar_t buf[0x4FFF]; swprintf(buf, 0x4FFF, _T(format), ##__VA_ARGS__); OutputDebugStringW(buf); }

#pragma endregion




// ======================================== Data / Parameters ========================================

#include <stdio.h>
#include "numerical/geometry.h"

#pragma region Global Variables

// window parameters
char text[64];	// window title
vec2 Center = vec2(0, 0);	// origin in screen coordinate
double Unit = 100.0;		// screen unit to object unit
#define fromInt(p) (((p) - Center) * (1.0 / Unit))
#define fromFloat(p) ((p) * Unit + Center)

// user parameters
vec2 Cursor = vec2(0, 0);
bool mouse_down = false;
bool Ctrl = false, Shift = false, Alt = false;

#include <chrono>
typedef std::chrono::high_resolution_clock NTime;
NTime::time_point _Global_Timer = NTime::now();
#define iTime std::chrono::duration<double>(NTime::now()-_Global_Timer).count()

#pragma endregion



namespace combined_function {
	double sdLine(vec2 p, vec2 a, vec2 b) { vec2 pa = p - a, ba = b - a; return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0., 1.)); }
	double u(vec2 p, double t) { return min(min(p.x - 0.16, sdLine(p, vec2(0.), vec2(1.22, 0.))), min(sdLine(vec2(p.x, abs(p.y)), vec2(0.45, 0.), vec2(0.45 + 0.42*cos(t), 0.42*sin(t))), sdLine(vec2(p.x, abs(p.y)), vec2(0.8, 0.), vec2(0.8 + 0.35*cos(t), 0.35*sin(t))))); }
	double fun(vec2 p, double t) { double a = abs(fmod(atan2(p.y, p.x) + .5*t, PI / 3.)) - PI / 6.; return u(vec2(cos(a), sin(a))*length(p)*(1.0 + 0.3*cos(t)), 0.9 - 0.3*cos(t)) - 0.05; }
}

int evals;
double fun(double x, double y) {
	evals++;
	//return x * x + y * y - 1;  // 3.141593
	//return hypot(x, y) - 1;  // 3.141593
	//return x * x*x*(x - 2) + y * y*y*(y - 2) + x;  // 5.215079
	//return x * x*x*x + y * y*y*y + x * y;  // 0.785398
	//return abs(abs(max(abs(x) - 0.9876, abs(y + 0.1*sin(3.*x)) - 0.4321)) - 0.3456) - 0.1234;
	//return max(2.*y - sin(10. * x), 4.*x*x + 4.*y * y - 9.);  // 14.13[6-7]
	//return max(abs(x) - abs(y) - 1, x*x + y * y - 2);  // 5.96[7-8]
	//return max(.5 - abs(x*y), x*x + 2 * y*y - 3);  // 1.81[0-3]
	//return max(sin(10*x) + cos(10*y) + 1., x*x + y * y - 1.);  // 0.57[3-4]
	return combined_function::fun(vec2(x, y), 1.7);  // 1.7[6-8]
}

// ============================================ Marching ============================================


// forward declarations of rendering functions
void drawLine(vec2 p, vec2 q, COLORREF col);
void drawBox(double x0, double x1, double y0, double y1, COLORREF col);
#define drawLineF(p,q,col) drawLine(fromFloat(p),fromFloat(q),col)


// global array to store marched segments
#include <vector>
struct segment {
	vec2 a, b;
	segment(vec2 a, vec2 b, bool swap = false) {  // swap: make sure the negative region is on the left
		if (swap) this->a = b, this->b = a;
		else this->a = a, this->b = b;
	}
	void swap() { std::swap(a, b); }
};
std::vector<segment> Segments;


namespace MARCH_LOOKUP_TABLES {

	// list of vertices on a unit square
	const static ivec2 VERTICE_LIST[4] = {
		ivec2(0,0), ivec2(1,0), ivec2(1,1), ivec2(0,1)
	};
	// list of edges connecting two vertices on a unit square
	const static ivec2 EDGE_LIST[4] = {
		ivec2(0,1), ivec2(1,2), ivec2(2,3), ivec2(3,0)
	};

	const static int SEGMENT_TABLE[16][4] = {
		{ -1, }, // 0000
		{ 0, 3, -1 }, // 1000
		{ 1, 0, -1 }, // 0100
		{ 1, 3, -1 }, // 1100
		{ 2, 1, -1 }, // 0010
		{ 0, 3, 2, 1 }, // 1010
		{ 2, 0, -1 }, // 0110
		{ 2, 3, -1 }, // 1110
		{ 3, 2, -1 }, // 0001
		{ 0, 2, -1 }, // 1001
		{ 1, 0, 3, 2 }, // 0101
		{ 1, 2, -1 }, // 1101
		{ 3, 1, -1 }, // 0011
		{ 0, 1, -1 }, // 1011
		{ 3, 0, -1 }, // 0111
		{ -1 }, // 1111
	};

}

void marchSquare(vec2 p0, vec2 p1, int SEARCH_DPS, int PLOT_DPS) {
	int N = 1 << SEARCH_DPS, M = N + 1;
	vec2 dp = (p1 - p0) / N;
	double *val = new double[M*M];
	for (int i = 0; i <= N; i++) for (int j = 0; j <= N; j++) {
		vec2 p = p0 + vec2(i, j)*dp;
		val[i*M + j] = fun(p.x, p.y);
	}
	auto getVal = [&](ivec2 ij) { return val[ij.x*M + ij.y]; };
	for (int xi = 0; xi < N; xi++) {
		for (int yi = 0; yi < N; yi++) {
			using namespace MARCH_LOOKUP_TABLES;
			// get signs and calculate index
			vec2 pos[4]; double val[4];
			for (int u = 0; u < 4; u++) {
				ivec2 ips = ivec2(xi, yi) + VERTICE_LIST[u];
				pos[u] = p0 + vec2(ips)*dp;
				val[u] = getVal(ips);
			}
			int index = int(val[0] < 0) | (int(val[1] < 0) << 1) | (int(val[2] < 0) << 2) | (int(val[3] < 0) << 3);
			// calculate linear interpolation
			auto getInterpolation = [&](int u) {
				double v0 = val[EDGE_LIST[u].x];
				double v1 = val[EDGE_LIST[u].y];
				vec2 p0 = pos[EDGE_LIST[u].x];
				vec2 p1 = pos[EDGE_LIST[u].y];
				return p0 + (v0 / (v0 - v1))*(p1 - p0);
			};
			// draw segments
			for (int u = 0; u < 4; u += 2) {
				int d0 = SEGMENT_TABLE[index][u];
				int d1 = SEGMENT_TABLE[index][u + 1];
				if (d0 == -1) break;
				Segments.push_back(segment(getInterpolation(d0), getInterpolation(d1)));
			}
			// debug
			if (SEGMENT_TABLE[index][0] != -1) drawBox(pos[0].x, pos[2].x, pos[0].y, pos[2].y, 0xFF0000);
		}
	}
}

// ============================================ Rendering ============================================

auto t0 = NTime::now();

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
void drawBox(double x0, double x1, double y0, double y1, COLORREF col) {
	drawLineF(vec2(x0, y0), vec2(x1, y0), col);
	drawLineF(vec2(x0, y1), vec2(x1, y1), col);
	drawLineF(vec2(x0, y0), vec2(x0, y1), col);
	drawLineF(vec2(x1, y0), vec2(x1, y1), col);
};


void remarch() {
	const bool CONSTANT = 1;
	int PC = (int)(0.5*log2(_WIN_W*_WIN_H) + 0.5);  // depth
	//SEARCH_DPS = max(PC - 3, 6), PLOT_DPS = max(PC - 1, 12);
	vec2 p0 = fromInt(vec2(0, 0)), p1 = fromInt(vec2(_WIN_W, _WIN_H));  // starting position and increasement
	if (CONSTANT) p0 = vec2(-2.5), p1 = vec2(2.5);
	evals = 0;
	Segments.clear();
	marchSquare(p0, p1, 4, 8);
}

void render() {
	// debug
	auto t1 = NTime::now();
	double dt = std::chrono::duration<double>(t1 - t0).count();
	dbgprint("[%d×%d] time elapsed: %.1fms (%.1ffps)\n", _WIN_W, _WIN_H, 1000.0*dt, 1. / dt);
	t0 = t1;

	// initialize window
	for (int i = 0, l = _WIN_W * _WIN_H; i < l; i++) _WINIMG[i] = 0;

	// "pixel shader" graphing implicit curve
#if 0
	for (int i = 0; i < _WIN_W; i++) for (int j = 0; j < _WIN_H; j++) {
		vec2 p = fromInt(vec2(i, j)); V[i][j] = fun(p.x, p.y);
		double dx = i ? V[i][j] - V[i - 1][j] : NAN;
		double dy = j ? V[i][j] - V[i][j - 1] : NAN;
		double v = V[i][j] / length(vec2(dx, dy));
		COLORREF col = (byte)(50.*clamp(5.0 - abs(v), 0, 1));
		_WINIMG[j*_WIN_W + i] = col | (col << 8) | (col) << 16;
	}
#endif

	// axis and grid
	{
		vec2 LB = fromInt(vec2(0, 0)), RT = fromInt(vec2(_WIN_W, _WIN_H));
		COLORREF GridCol = clamp((byte)0x10, (byte)sqrt(10.0*Unit), (byte)0x20); GridCol = GridCol | (GridCol << 8) | (GridCol) << 16;  // adaptive grid color
		for (int y = (int)round(LB.y), y1 = (int)round(RT.y); y <= y1; y++) drawLine(fromFloat(vec2(LB.x, y)), fromFloat(vec2(RT.x, y)), GridCol);  // horizontal gridlines
		for (int x = (int)round(LB.x), x1 = (int)round(RT.x); x <= x1; x++) drawLine(fromFloat(vec2(x, LB.y)), fromFloat(vec2(x, RT.y)), GridCol);  // vertical gridlines
		drawLine(vec2(0, Center.y), vec2(_WIN_W, Center.y), 0x202080);  // x-axis
		drawLine(vec2(Center.x, 0), vec2(Center.x, _WIN_H), 0x202080);  // y-axis
	}

	//if (Segments.empty())
	remarch();
	int SN = Segments.size();

	// rendering
	for (int i = 0; i < SN; i++) {
		vec2 a = Segments[i].a, b = Segments[i].b, c = (a + b)*0.5;
		drawLineF(a, b, 0xFFFFFF);
		drawLineF(c, c + (b - a).rot(), 0xFFFF00);
	}

	// calculate area
	double Area = 0;
	for (int i = 0; i < SN; i++) {
		Area += det(Segments[i].a, Segments[i].b);
	}
	Area *= 0.5;

	vec2 cursor = fromInt(Cursor);
	sprintf(text, "(%.2f,%.2f)  %d evals, %d segments, Area=%lf", cursor.x, cursor.y, evals, SN, Area);
	SetWindowTextA(_HWND, text);
}


// ============================================== User ==============================================


void WindowCreate(int _W, int _H) {
	Center = vec2(_W, _H) * 0.5;
}
void WindowResize(int _oldW, int _oldH, int _W, int _H) {
	if (_W*_H == 0 || _oldW * _oldH == 0) return;
	double pw = _oldW, ph = _oldH, w = _W, h = _H;
	double s = sqrt((w * h) / (pw * ph));
	Unit *= s;
	Center.x *= w / pw, Center.y *= h / ph;
}
void WindowClose() {

}

void MouseMove(int _X, int _Y) {
	vec2 P0 = Cursor, P = vec2(_X, _Y);
	Cursor = P;
	vec2 p0 = fromInt(P0), p = fromInt(P), d = p - p0;

	// click and drag
	if (mouse_down) {
		Center = Center + d * Unit;
	}

}

void MouseWheel(int _DELTA) {
	double s = exp((Alt ? 0.0001 : 0.001)*_DELTA);
	double D = length(vec2(_WIN_W, _WIN_H)), Max = 10.0*D, Min = 0.01*D;
	if (Unit * s > Max) s = Max / Unit;
	else if (Unit * s < Min) s = Min / Unit;
	Center = mix(Cursor, Center, s);
	Unit *= s;
}

void MouseDownL(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	mouse_down = true;
	vec2 p = fromInt(Cursor);
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

