// Win32 2D GUI template


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

#pragma endregion  // Window variables




// ============================================ Rendering ============================================

#include <chrono>
typedef std::chrono::high_resolution_clock NTime;
typedef std::chrono::duration<double> fsec;
auto t0 = NTime::now();


#include <vector>
#include "raytracing/brdf.h"
#include "numerical/random.h"
#include "variance.h"


COLORREF toCOLORREF(const vec3f &col) {
	COLORREF C; byte* c = (byte*)&C;
	c[2] = (byte)(255 * clamp(col.x, 0.f, 1.f));
	c[1] = (byte)(255 * clamp(col.y, 0.f, 1.f));
	c[0] = (byte)(255 * clamp(col.z, 0.f, 1.f));
	return C;
}


#define MultiThread 1
#include <thread>

#if MultiThread
void Render_Exec(void(*task)(int, int, int, bool*), int Max) {
	const int MAX_THREADS = std::thread::hardware_concurrency();
	bool* fn = new bool[MAX_THREADS];
	std::thread** T = new std::thread*[MAX_THREADS];
	for (int i = 0; i < MAX_THREADS; i++) {
		fn[i] = false;
		T[i] = new std::thread(task, i, Max, MAX_THREADS, &fn[i]);
	}
	int count; do {
		count = 0;
		for (int i = 0; i < MAX_THREADS; i++) count += fn[i];
	} while (count < MAX_THREADS);
	//for (int i = 0; i < MAX_THREADS; i++) delete T[i];
	delete fn; delete T;
}
#else
void Render_Exec(void(*task)(int, int, int, bool*), int Max) {
	task(0, Max, 1, NULL);
}
#endif


bool intersectCircle(vec2 O, double r, vec2 ro, vec2 rd, double &t, vec2 &n) {
	ro -= O;
	double b = -dot(ro, rd), c = dot(ro, ro) - r * r;
	double delta = b * b - c;
	if (delta < 0.0) return false;
	delta = sqrt(delta);
	double t1 = b - delta, t2 = b + delta;
	if (t1 > t2) std::swap(t1, t2);
	if (t1 > t || t2 < 0.) return false;
	t = t1 > 0. ? t1 : t2;
	n = normalize(ro + rd * t);
	return true;
}


VarianceObject<vec3f> colorBuffer[WinW_Max][WinH_Max];
int colorBuffer_N = 0;

// dome light
vec3f calcCol(vec2 ro, vec2 rd, uint32_t &seed) {

	vec3f m_col = vec3f(1.0f), col;


	struct Bulb {
		vec2 pos;
		double radius;
		vec3f intensity;
	};
	std::vector<Bulb> bulbs({
		Bulb{vec2(3, 2), 0.5, vec3f(2.0,2.0,5.0)},
		Bulb{vec2(-4, 0), 0.5, vec3f(2.0,5.0,2.0)},
		Bulb{vec2(3, -2), 0.5, vec3f(5.0,2.0,2.0)},
		});

	struct Ball {
		vec2 pos;
		double radius;
		double refl_index;
	};
	std::vector<Ball> balls({
		Ball{vec2(0, 0), 1.0, 1.35},
		Ball{vec2(-2, 2), 0.7, 1.5},
		Ball{vec2(-2, -2), 0.7, 2.5},
		});

	bool is_inside = false;
	for (Ball bl : balls) {
		if (length(ro - bl.pos) < bl.radius)
			is_inside = true;
	}

	// "recursive" ray-tracing
	for (int iter = 0; iter < 64; iter++) {
		vec2 n, min_n;
		double t, min_t = INFINITY;
		ro += 1e-6*rd;  // alternate of t>1e-6

		int intersect_id = -1;  // which object the ray hits

		// light, only emission
		for (Bulb bb : bulbs) {
			t = min_t;
			if (intersectCircle(bb.pos, bb.radius, ro, rd, t, n)) {
				min_t = t, min_n = n;
				intersect_id = 0;
				//col = bb.intensity * (dot(n, rd) > 0.0 ? 1.0 : -dot(n, rd));
				col = bb.intensity;
			}
		}

		// glass circles
		double refl_index = 1.0;
		for (Ball bl : balls) {
			t = min_t;
			if (intersectCircle(bl.pos, bl.radius, ro, rd, t, n)) {
				min_t = t, min_n = n;
				intersect_id = 1;
				col = vec3f(1.0f);
				refl_index = bl.refl_index;
			}
		}

		// update ray
		if (intersect_id == -1) {  // nothing
			return vec3f(0);
		}
		m_col *= col;
		min_n = dot(rd, min_n) < 0. ? min_n : -min_n;  // ray hits into the surface
		ro = ro + rd * min_t;
		if (intersect_id == 0) {  // light
			return m_col;
		}
		else if (intersect_id == 1) {  // transparent shape
			vec2 eta = is_inside ? vec2(refl_index, 1.0) : vec2(1.0, refl_index);
			rd = randdir_Fresnel(rd, n, eta.x, eta.y, seed);
		}
		if (dot(rd, min_n) < 0.0) {
			is_inside ^= 1;
		}
	}
	return vec3f(0.0);
}


void render_RT() {
	static uint32_t call_time = 0;
	call_time = lcg_next(call_time);

	colorBuffer_N++;

	Render_Exec([](int beg, int end, int step, bool* sig) {
		const int WIN_SIZE = _WIN_W * _WIN_H;
		for (int k = beg; k < end; k += step) {
			int i = k % _WIN_W, j = k / _WIN_W;
			if (1) {
				uint32_t seed = hashu(WIN_SIZE + k + call_time);
				vec2 p = fromInt(vec2(i, j) + vec2(rand01(seed), rand01(seed)));
				vec2 d;
				if (1) d = cossin(2.0*PI*VanDerCorput_2<uint32_t, double>(uint32_t(hashu(k) + call_time)));
				else d = cossin(2.0*PI*rand01(seed));
				vec3f col = calcCol(p, d, seed);
				if (colorBuffer_N == 1) colorBuffer[i][j] = VarianceObject<vec3f>();
				colorBuffer[i][j].addElement(col);
			}
			vec3f col = colorBuffer[i][j].getMean();
			Canvas(i, j) = toCOLORREF(col);
		}
		if (sig) *sig = true;
	}, _WIN_W*_WIN_H);
}




void render() {
	// initialize window
	for (int i = 0, l = _WIN_W * _WIN_H; i < l; i++) _WINIMG[i] = 0;

	render_RT();

	auto t1 = std::chrono::high_resolution_clock::now();
	double dt = std::chrono::duration<double>(t1 - t0).count();
	t0 = t1;
	vec2 cursor = fromInt(Cursor);
	sprintf(text, "[%d×%d, %d]  %.1fms (%.1ffps)  (%.2f,%.2f)\n",
		_WIN_W, _WIN_H, colorBuffer_N, 1000.0*dt, 1. / dt, cursor.x, cursor.y);
	SetWindowTextA(_HWND, text);
}




// ============================================== User ==============================================

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include ".libraries/stb_image_write.h"

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
	colorBuffer_N = 0;
}
void WindowClose() {
	if (0) {
		vec3f *pixels = new vec3f[_WIN_W*_WIN_H];
		for (int i = 0; i < _WIN_W; i++) for (int j = 0; j < _WIN_H; j++) {
			pixels[j*_WIN_W + i] = pow(colorBuffer[i][_WIN_H - 1 - j].getMean(), 2.2f);
		}
		stbi_write_hdr("D:\\.hdr", _WIN_W, _WIN_H, 3, (float*)pixels);
		delete pixels;
	}
}

void MouseMove(int _X, int _Y) {
	vec2 P0 = Cursor, P = vec2(_X, _Y);
	Cursor = P;
	vec2 p0 = fromInt(P0), p = fromInt(P), d = p - p0;

	// click and drag
	if (mouse_down) {
		colorBuffer_N = 0;
		Origin = Origin + d * Unit;
	}

	Render_Needed = true;
}

void MouseWheel(int _DELTA) {
	Render_Needed = true;
	if (_DELTA) colorBuffer_N = 0;
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

