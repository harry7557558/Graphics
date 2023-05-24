// (simple??) Win32 2D GUI template


// ========================================= Win32 GUI =========================================

#pragma region Windows

#include <Windows.h>
#include <windowsx.h>
#include <tchar.h>

#define WIN_NAME "2D GUI Template"
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
#if 1
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow) {
#else
int main() {
	HINSTANCE hInstance = NULL; int nCmdShow = SW_RESTORE;
#endif
	WNDCLASSEX wc; wc.cbSize = sizeof(WNDCLASSEX), wc.style = 0, wc.lpfnWndProc = WndProc, wc.cbClsExtra = wc.cbWndExtra = 0, wc.hInstance = hInstance; wc.hIcon = wc.hIconSm = 0, wc.hCursor = LoadCursor(NULL, IDC_ARROW), wc.hbrBackground = CreateSolidBrush(RGB(0, 0, 0)), wc.lpszMenuName = NULL, wc.lpszClassName = _T(WIN_NAME); if (!RegisterClassEx(&wc)) return -1;
	_HWND = CreateWindow(_T(WIN_NAME), _T(WIN_NAME), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, WinW_Default, WinH_Default, NULL, NULL, hInstance, NULL); ShowWindow(_HWND, nCmdShow); UpdateWindow(_HWND);
	MSG message; while (GetMessage(&message, 0, 0, 0)) { TranslateMessage(&message); DispatchMessage(&message); } return (int)message.wParam;
}

#pragma endregion  // Windows





// ======================================== Data / Parameters ========================================

#include "numerical/geometry.h"
#include <stdio.h>


#pragma region Window Variables

// window parameters
char text[64];	// window title
vec2 Origin = vec2(0, 0);	// origin in screen coordinate
double Unit = 50.0;		// screen unit to object unit
#define fromInt(p) (((p) - Origin) * (1.0 / Unit))
#define fromFloat(p) ((p) * Unit + Origin)

// user parameters
vec2 Cursor = vec2(0, 0);
bool mouse_down = false;
bool Ctrl = false, Shift = false, Alt = false;

// forward declarations of rendering functions
void drawLine(vec2 p, vec2 q, COLORREF col);
void drawBox(vec2 p0, vec2 p1, COLORREF col);
void drawCircle(vec2 c, double r, COLORREF col);
void fillBox(vec2 p0, vec2 p1, COLORREF col);
void fillCircle(vec2 c, double r, COLORREF col);
void drawTriangle(vec2 A, vec2 B, vec2 C, COLORREF col);
void fillTriangle(vec2 A, vec2 B, vec2 C, COLORREF col);
#define drawLineF(p,q,col) drawLine(fromFloat(p),fromFloat(q),col)
#define drawBoxF(p0,p1,col) drawBox(fromFloat(p0),fromFloat(p1),col)
#define drawCircleF(c,r,col) drawCircle(fromFloat(c),(r)*Unit,col)
#define fillBoxF(p0,p1,col) fillBox(fromFloat(p0),fromFloat(p1),col)
#define fillCircleF(c,r,col) fillCircle(fromFloat(c),(r)*Unit,col)
#define drawDotF(c,r,col) fillCircle(fromFloat(c),r,col)  // r: in screen coordinate
#define drawDotSquareF(c,r,col) fillBox(fromFloat(c)-vec2(r),fromFloat(c)+vec2(r),col)  // r: in screen coordinate
#define drawDotHollowF(c,r,col) drawCircle(fromFloat(c),r,col)  // r: in screen coordinate
#define drawSquareHollowF(c,r,col) drawBox(fromFloat(c)-vec2(r),fromFloat(c)+vec2(r),col)  // r: in screen coordinate
#define drawTriangleF(A,B,C,col) drawTriangle(fromFloat(A),fromFloat(B),fromFloat(C),col)
#define fillTriangleF(A,B,C,col) fillTriangle(fromFloat(A),fromFloat(B),fromFloat(C),col)


#pragma endregion  // Window variables




// ============================================ Rendering ============================================

#include <chrono>
auto t0 = std::chrono::high_resolution_clock::now();


// rendering functions
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
void drawBox(vec2 p0, vec2 p1, COLORREF col) {
	drawLine(vec2(p0.x, p0.y), vec2(p1.x, p0.y), col);
	drawLine(vec2(p0.x, p1.y), vec2(p1.x, p1.y), col);
	drawLine(vec2(p0.x, p0.y), vec2(p0.x, p1.y), col);
	drawLine(vec2(p1.x, p0.y), vec2(p1.x, p1.y), col);
};
void drawCircle(vec2 c, double r, COLORREF col) {
	int s = int(r / sqrt(2) + 0.5);
	int cx = (int)c.x, cy = (int)c.y;
	for (int i = 0, im = min(s, max(_WIN_W - cx, cx)) + 1; i < im; i++) {
		int u = (int)sqrt(r*r - i * i);
		setColor(cx + i, cy + u, col); setColor(cx + i, cy - u, col); setColor(cx - i, cy + u, col); setColor(cx - i, cy - u, col);
		setColor(cx + u, cy + i, col); setColor(cx + u, cy - i, col); setColor(cx - u, cy + i, col); setColor(cx - u, cy - i, col);
	}
}
void fillBox(vec2 p0, vec2 p1, COLORREF col) {
	int i0 = max(0, (int)p0.x), i1 = min(_WIN_W - 1, (int)p1.x);
	int j0 = max(0, (int)p0.y), j1 = min(_WIN_H - 1, (int)p1.y);
	for (int j = j0; j <= j1; j++) for (int i = i0; i <= i1; i++) {
		Canvas(i, j) = col;
	}
}
void fillCircle(vec2 c, double r, COLORREF col) {
	c -= vec2(0.5);
	int i0 = max(0, (int)floor(c.x - r - 1)), i1 = min(_WIN_W - 1, (int)ceil(c.x + r + 1));
	int j0 = max(0, (int)floor(c.y - r - 1)), j1 = min(_WIN_H - 1, (int)ceil(c.y + r + 1));
	for (int j = j0; j <= j1; j++) for (int i = i0; i <= i1; i++) {
		double d = (vec2(i, j) - c).sqr() - r * r;
		if (d < 0.) Canvas(i, j) = col;
	}
}
void drawTriangle(vec2 A, vec2 B, vec2 C, COLORREF col) {
	drawLine(A, B, col); drawLine(B, C, col); drawLine(C, A, col);
}
void fillTriangle(vec2 A, vec2 B, vec2 C, COLORREF col) {
	int x0 = max((int)min(min(A.x, B.x), C.x), 0), x1 = min((int)max(max(A.x, B.x), C.x), _WIN_W - 1);
	int y0 = max((int)min(min(A.y, B.y), C.y), 0), y1 = min((int)max(max(A.y, B.y), C.y), _WIN_H - 1);
	for (int i = y0; i <= y1; i++) for (int j = x0; j <= x1; j++) {
		vec2 P(j, i);
		if (((det(P - A, P - B) < 0) + (det(P - B, P - C) < 0) + (det(P - C, P - A) < 0)) % 3 == 0)
			Canvas(j, i) = col;
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

	// default scene
	{
		drawBoxF(vec2(-2.5), vec2(2.5), 0xFFFF00);
		drawCircleF(vec2(0, 0), 1.0, 0xFFFFFF);
		drawDotSquareF(vec2(1, 0), 5., 0xFF00FF);
		drawDotF(vec2(0, 1), 6., 0x8000FF);
		drawLineF(vec2(-2), vec2(2), 0xFF0000);
		drawTriangleF(vec2(-1, 1), vec2(0.2), vec2(1, -1), 0xFF8000);
		fillTriangleF(vec2(1.7, 1.9), vec2(2), vec2(1.9, 1.7), 0xFF0000);
		drawDotHollowF(vec2(1, 1), 6., 0x8000FF);
		drawSquareHollowF(vec2(-2), 6., 0x8000FF);
	}


	vec2 cursor = fromInt(Cursor);
	sprintf(text, "(%.2f,%.2f)", cursor.x, cursor.y);
	SetWindowTextA(_HWND, text);
}




// ============================================== User ==============================================


void WindowCreate(int _W, int _H) {
	// by default Origin=vec2(0.)
	Origin = (fromInt(0.5*vec2(_WIN_W, _WIN_H)) - vec2(0, 0)) * Unit;
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

