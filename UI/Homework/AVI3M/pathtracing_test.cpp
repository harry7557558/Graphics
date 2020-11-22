

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


#define WIN_NAME "UI"
#define WinW_Padding 100
#define WinH_Padding 100
#define WinW_Default 640
#define WinH_Default 400
#define WinW_Min 640
#define WinH_Min 400
#define WinW_Max 640
#define WinH_Max 400

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



// Win32 Entry

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
#define _RDBK { HDC hdc = GetDC(_HWND), HImgMem = CreateCompatibleDC(hdc); HBITMAP hbmOld = (HBITMAP)SelectObject(HImgMem, _HIMG); render(); BitBlt(hdc, 0, 0, _WIN_W, _WIN_H, HImgMem, 0, 0, SRCCOPY); SelectObject(HImgMem, hbmOld), DeleteDC(HImgMem), DeleteDC(hdc); } break;
	switch (message) {
	case WM_CREATE: { if (!_HWND) Init(); break; }
	case WM_CLOSE: { WindowClose(); DestroyWindow(hWnd); return 0; }
	case WM_DESTROY: { PostQuitMessage(0); return 0; }
	case WM_MOVE:; case WM_SIZE: {
		RECT Client; GetClientRect(hWnd, &Client); WindowResize(_WIN_W, _WIN_H, Client.right, Client.bottom); _WIN_W = Client.right, _WIN_H = Client.bottom;
		BITMAPINFO bmi; bmi.bmiHeader.biSize = sizeof(BITMAPINFO), bmi.bmiHeader.biWidth = Client.right, bmi.bmiHeader.biHeight = Client.bottom, bmi.bmiHeader.biPlanes = 1, bmi.bmiHeader.biBitCount = 32; bmi.bmiHeader.biCompression = BI_RGB, bmi.bmiHeader.biSizeImage = 0, bmi.bmiHeader.biXPelsPerMeter = bmi.bmiHeader.biYPelsPerMeter = 0, bmi.bmiHeader.biClrUsed = bmi.bmiHeader.biClrImportant = 0; bmi.bmiColors[0].rgbBlue = bmi.bmiColors[0].rgbGreen = bmi.bmiColors[0].rgbRed = bmi.bmiColors[0].rgbReserved = 0;
		if (_HIMG != NULL) break; HDC hdc = GetDC(hWnd); _HIMG = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, (void**)&_WINIMG, NULL, 0); DeleteDC(hdc); _RDBK
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


#include "numerical/geometry.h"

const vec3 vec0(0, 0, 0), veci(1, 0, 0), vecj(0, 1, 0), veck(0, 0, 1);
#define SCRCTR vec2(0.5*_WIN_W,0.5*_WIN_H)




// ray intersection

// consider intersected if the distance is greater than this number
constexpr double epsilon = 1e-6;

#define invec3 const vec3&
inline double intTriangle_r(invec3 P, invec3 a, invec3 b, invec3 n, invec3 ro, invec3 rd) {  // relative with precomputer normal cross(a,b)
	vec3 rp = ro - P;
	vec3 q = cross(rp, rd);
	double d = 1.0 / dot(rd, n);
	double u = -d * dot(q, b); if (u<0. || u>1.) return NAN;
	double v = d * dot(q, a); if (v<0. || (u + v)>1.) return NAN;
	return -d * dot(n, rp);
}
inline double intBoxC(invec3 R, invec3 ro, invec3 inv_rd) {  // inv_rd = vec3(1.0)/rd
	vec3 p = -inv_rd * ro;
	vec3 k = abs(inv_rd)*R;
	vec3 t1 = p - k, t2 = p + k;
	double tN = max(max(t1.x, t1.y), t1.z);
	double tF = min(min(t2.x, t2.y), t2.z);
	if (tN > tF || tF < 0.0) return NAN;
	return tN;
	//return tN > 0. ? tN : tF;
}

// the "ordinary" version for reference
double intTriangle(vec3 p0, vec3 e1, vec3 e2, vec3 ro, vec3 rd) {
	ro = ro - p0;
	vec3 n = cross(e1, e2);
	vec3 q = cross(ro, rd);
	double d = 1.0 / dot(rd, n);
	double u = -d * dot(q, e2); if (u<0. || u>1.) return NAN;
	double v = d * dot(q, e1); if (v<0. || (u + v)>1.) return NAN;
	double t = -d * dot(n, ro);
	return t > epsilon ? t : NAN;
}




// ======================================== Data / Parameters ========================================


#pragma region Global Variables and Functions

// viewport
vec3 Center(0.0, 0.0, 0.0);  // view center in world coordinate
double rz = -0.8, rx = 0.3, ry = 0.0, dist = 12.0, Unit = 100.0;  // yaw, pitch, row, camera distance, scale to screen

// window parameters
char text[64];	// window title
vec3 CamP, ScrO, ScrA, ScrB;  // camera and screen
auto scrPos = [](vec2 pixel) { return ScrO + (pixel.x / _WIN_W)*ScrA + (pixel.y / _WIN_H)*ScrB; };
auto scrDir = [](vec2 pixel) { return normalize(scrPos(pixel) - CamP); };

// user parameters
vec2 Cursor = vec2(0, 0), clickCursor;  // current cursor and cursor position when mouse down
bool mouse_down = false;
bool Ctrl = false, Shift = false, Alt = false;  // these variables are shared by both windows


// projection
void getRay(vec2 Cursor, vec3 &p, vec3 &d) {
	p = CamP;
	d = normalize(ScrO + (Cursor.x / _WIN_W)*ScrA + (Cursor.y / _WIN_H)*ScrB - CamP);
}
void getScreen(vec3 &P, vec3 &O, vec3 &A, vec3 &B) {  // O+uA+vB
	double cx = cos(rx), sx = sin(rx), cz = cos(rz), sz = sin(rz);
	vec3 u(-sz, cz, 0), v(-cz * sx, -sz * sx, cx), w(cz * cx, sz * cx, sx);
	mat3 Y = axis_angle(w, -ry); u = Y * u, v = Y * v;
	u *= 0.5*_WIN_W / Unit, v *= 0.5*_WIN_H / Unit, w *= dist;
	P = Center + w;
	O = Center - (u + v), A = u * 2.0, B = v * 2.0;
}


#pragma endregion



struct Triangle {
	vec3 n;  // cross(A,B)
	vec3 P, A, B;  // P+uA+vB
} *STL;
int STL_N;



// ============================================ Rendering ============================================



#include <chrono>
typedef std::chrono::high_resolution_clock NTime;
typedef std::chrono::duration<double> fsec;

COLORREF toCOLORREF(const vec3 &col) {
	COLORREF C; byte* c = (byte*)&C;
	c[2] = (byte)(255 * clamp(col.x, 0., 1.));
	c[1] = (byte)(255 * clamp(col.y, 0., 1.));
	c[0] = (byte)(255 * clamp(col.z, 0., 1.));
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





// BVH ray tracing
#include <vector>
#define MAX_TRIG 16
struct BVH {
	Triangle *Obj[MAX_TRIG];
	vec3 C, R;  // bounding box
	BVH *b1 = 0, *b2 = 0;  // children
} *BVH_R = 0;
void constructBVH(BVH* &R, std::vector<Triangle*> &T, vec3 &Min, vec3 &Max) {  // R should not be null and T should not be empty, calculates box range
	int N = (int)T.size();
	Min = vec3(INFINITY), Max = vec3(-INFINITY);

	if (N <= MAX_TRIG) {
		for (int i = 0; i < N; i++) R->Obj[i] = T[i];
		if (N < MAX_TRIG) R->Obj[N] = 0;
		for (int i = 0; i < N; i++) {
			vec3 A = T[i]->P, B = T[i]->P + T[i]->A, C = T[i]->P + T[i]->B;
			Min = pMin(pMin(Min, A), pMin(B, C));
			Max = pMax(pMax(Max, A), pMax(B, C));
		}
		R->C = 0.5*(Max + Min), R->R = 0.5*(Max - Min);
		return;
	}
	else R->Obj[0] = NULL;

	// Analysis shows this is the most time-consuming part in this function
	const double _3 = 1. / 3;
	for (int i = 0; i < N; i++) {
		vec3 C = T[i]->P + _3 * (T[i]->A + T[i]->B);
		Min = pMin(Min, C), Max = pMax(Max, C);
	}
	vec3 dP = Max - Min;

	std::vector<Triangle*> c1, c2;
	if (dP.x >= dP.y && dP.x >= dP.z) {
		double x = 0.5*(Min.x + Max.x); for (int i = 0; i < N; i++) {
			if (T[i]->P.x + _3 * (T[i]->A.x + T[i]->B.x) < x) c1.push_back(T[i]);
			else c2.push_back(T[i]);
		}
	}
	else if (dP.y >= dP.x && dP.y >= dP.z) {
		double y = 0.5*(Min.y + Max.y); for (int i = 0; i < N; i++) {
			if (T[i]->P.y + _3 * (T[i]->A.y + T[i]->B.y) < y) c1.push_back(T[i]);
			else c2.push_back(T[i]);
		}
	}
	else {
		double z = 0.5*(Min.z + Max.z); for (int i = 0; i < N; i++) {
			if (T[i]->P.z + _3 * (T[i]->A.z + T[i]->B.z) < z) c1.push_back(T[i]);
			else c2.push_back(T[i]);
		}
	}

	if (c1.empty() || c2.empty()) {
		// faster in neither construction nor intersection
		// I keep it because...
		if (dP.x >= dP.y && dP.x >= dP.z) std::sort(T.begin(), T.end(), [](Triangle *a, Triangle *b) { return 3.*a->P.x + a->A.x + a->B.x < 3.*b->P.x + b->A.x + b->B.x; });
		else if (dP.y >= dP.x && dP.y >= dP.z) std::sort(T.begin(), T.end(), [](Triangle *a, Triangle *b) { return 3.*a->P.y + a->A.y + a->B.y < 3.*b->P.y + b->A.y + b->B.y; });
		else std::sort(T.begin(), T.end(), [](Triangle *a, Triangle *b) { return 3.*a->P.z + a->A.z + a->B.z < 3.*b->P.z + b->A.z + b->B.z; });
		int d = N / 2;
		c1 = std::vector<Triangle*>(T.begin(), T.begin() + d);
		c2 = std::vector<Triangle*>(T.begin() + d, T.end());
	}
	// A paper I haven't read yet: https://graphicsinterface.org/wp-content/uploads/gi1989-22.pdf

	vec3 b0, b1;
	R->b1 = new BVH; constructBVH(R->b1, c1, Min, Max);
	R->b2 = new BVH; constructBVH(R->b2, c2, b0, b1);
	Min = pMin(Min, b0); Max = pMax(Max, b1);
	R->C = 0.5*(Max + Min), R->R = 0.5*(Max - Min);
}
void rayIntersectBVH(const BVH* R, invec3 ro, invec3 rd, invec3 inv_rd, double &mt, Triangle* &obj) {  // assume ray already intersects current BVH
	if (R->Obj[0]) {
		for (int i = 0; i < MAX_TRIG; i++) {
			Triangle *T = R->Obj[i];
			if (!T) break;
			double t = intTriangle_r(T->P, T->A, T->B, T->n, ro, rd);
			if (t > epsilon && t < mt) mt = t, obj = T;
		}
	}
	else {
		double t1 = intBoxC(R->b1->R, ro - R->b1->C, inv_rd);
		double t2 = intBoxC(R->b2->R, ro - R->b2->C, inv_rd);
#if 0
		if (t1 < mt) rayIntersectBVH(R->b1, ro, rd, inv_rd, mt, obj);
		if (t2 < mt) rayIntersectBVH(R->b2, ro, rd, inv_rd, mt, obj);
#else
		// test intersection for the closer box first
		// there is a significant performance increase
		if (t1 < mt && t2 < mt) {
			if (t1 < t2) {
				rayIntersectBVH(R->b1, ro, rd, inv_rd, mt, obj);
				if (t2 < mt) rayIntersectBVH(R->b2, ro, rd, inv_rd, mt, obj);
			}
			else {
				rayIntersectBVH(R->b2, ro, rd, inv_rd, mt, obj);
				if (t1 < mt) rayIntersectBVH(R->b1, ro, rd, inv_rd, mt, obj);
			}
		}
		else {
			if (t1 < mt) rayIntersectBVH(R->b1, ro, rd, inv_rd, mt, obj);
			if (t2 < mt) rayIntersectBVH(R->b2, ro, rd, inv_rd, mt, obj);
		}
#endif
	}
}

bool intersectScene(vec3 ro, vec3 rd, double &t, vec3 &n) {
	t = intBoxC(BVH_R->R, ro - BVH_R->C, vec3(1.0) / rd);
	if (!isnan(t)) {
		Triangle* obj = 0;
		t = INFINITY;
		rayIntersectBVH(BVH_R, ro, rd, vec3(1.0) / rd, t, obj);
		if (obj) {
			n = normalize(obj->n);
			return true;
		}
	}
	return false;
}
bool intersectScene_bruteforce(vec3 ro, vec3 rd, double &mt, vec3 &n) {  // reference
	bool res = false;
	for (int i = 0; i < STL_N; i++) {
		double t = intTriangle(STL[i].P, STL[i].A, STL[i].B, ro, rd);
		if (t > epsilon && t < mt) {
			mt = t;
			n = ncross(STL[i].A, STL[i].B);
			res = true;
		}
	}
	return res;
}

bool intersectScene_test(vec3 ro, vec3 rd, double &t, vec3 &n) {  // test scene, a perfect sphere
	const vec3 O = vec3(0, 0, 1.1);
	const double r = 1.0;
	ro -= O;
	double b = -dot(ro, rd), c = dot(ro, ro) - r * r;
	double delta = b * b - c;
	if (delta < 0.0) return false;
	delta = sqrt(delta);
	c = b - delta;
	t = c > 0. ? c : b + delta;
	if (t < epsilon) return false;
	n = normalize(ro + rd * t);
	return true;
}



// path tracing
#include "numerical/random.h"

// importance sampling
vec3 randdir_cosWeighted(vec3 n, uint32_t &seed) {
	vec3 u = ncross(n, vec3(1.2345, 2.3456, -3.4561));
	vec3 v = cross(u, n);
	double rn = rand01(seed);
	vec2 rh = sqrt(rn) * cossin(2.*PI*rand01(seed));
	double rz = sqrt(1. - rn);
	return rh.x * u + rh.y * v + rz * n;
}
// the ray comes from a medium with reflective index n1 to a medium with reflective index n2
vec3 randdir_Fresnel(vec3 rd, vec3 n, double n1, double n2, uint32_t &seed) {
	double eta = n1 / n2;
	double ci = -dot(n, rd);
	if (ci < 0.) ci = -ci, n = -n;
	double ct = 1.0 - eta * eta * (1.0 - ci * ci);
	if (ct < 0.) return rd + 2.*ci*n;
	ct = sqrt(ct);
	double Rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct);
	double Rp = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci);
	double R = 0.5 * (Rs * Rs + Rp * Rp);
	return rand01(seed) > R ?
		rd * eta + n * (eta * ci - ct)  // refraction
		: rd + 2.*ci*n;  // reflection
}

/* Working on: Debugging this function */
vec3 calcCol(vec3 ro, vec3 rd, uint32_t &seed) {

	const vec3 light = normalize(vec3(0, 0, 1));
	vec3 m_col = vec3(2.0), col;

	bool isInside = false;  // the ray is inside a "glass" or not

	// "recursive" ray-tracing
	for (int iter = 0; iter < 20; iter++) {
		vec3 n, min_n;

		// intersect plane
		double min_t = -ro.z / rd.z;
		if (min_t > epsilon) {
			vec2 p = ro.xy() + min_t * rd.xy();
			col = max(abs(p.x) - 0.618, abs(p.y) - 1.) < 0. ? vec3(0.8) : vec3(0.6);
			min_n = vec3(0, 0, 1);
		}
		else {
			min_t = INFINITY;
			col = vec3(max(dot(rd, light), 0.));
		}

		// intersect scene
		double t = min_t;
		if (intersectScene_test(ro, rd, t, n) && t < min_t) {
			min_t = t, min_n = n;
			//col = vec3(0.5) + 0.5*n;
			//col = vec3(0.8) + 0.2*n;
			col = vec3(0.8, 1.0, 0.8);
			if (isInside) col = exp(-(vec3(1.) - col)*min_t);
			else col = vec3(1.);
		}

		// update ray
		m_col *= col;
		if (min_t == INFINITY) {
			return m_col;
		}
		min_n = dot(rd, min_n) < 0. ? min_n : -min_n;  // ray hits into the surface
		ro = ro + rd * min_t;
		if (abs(min_n.z) == 1) {
			if (isInside) printf("bug\n");  // :(
			rd = rd - min_n * (2.*dot(rd, min_n));
			//rd = randdir_cosWeighted(min_n, seed);
		}
		else {
			//rd = rd - min_n * (2.*dot(rd, min_n));
			//rd = randdir_cosWeighted(min_n, seed);
			rd = isInside ? randdir_Fresnel(rd, min_n, 1.5, 1.0, seed) : randdir_Fresnel(rd, min_n, 1.0, 1.5, seed);  // very likely that I have a bug
		}
		if (dot(rd, min_n) < 0.) {
			isInside ^= 1;  // reflected ray hits into the surface
			if (!isInside) return vec3(0, 0, 1);
		}
	}
	return m_col;
}


void render_RT_BVH() {
	if (!BVH_R) {
		auto t0 = NTime::now();
		std::vector<Triangle*> T;
		for (int i = 0; i < STL_N; i++) T.push_back(&STL[i]);
		BVH_R = new BVH;
		vec3 Min(INFINITY), Max(-INFINITY);
		constructBVH(BVH_R, T, Min, Max);
		dbgprint("BVH constructed in %lfs\n", fsec(NTime::now() - t0).count());
		//exit(0);
	}

	static uint32_t call_time = 0;
	call_time = lcg_next(call_time);

	Render_Exec([](int beg, int end, int step, bool* sig) {
		const int WIN_SIZE = _WIN_W * _WIN_H;
		for (int k = beg; k < end; k += step) {
			int i = k % _WIN_W, j = k / _WIN_W;
			// bruteforce Monte-Carlo sampling
			const int N = 1;
			vec3 col(0.);
			for (int u = 0; u < N; u++) {
				uint32_t seed = hashu(u*WIN_SIZE + k + call_time);
				vec3 d = scrDir(vec2(i + rand01(seed), j + rand01(seed)));
				col += calcCol(CamP, d, seed);
			}
			Canvas(i, j) = toCOLORREF(col / N);
		}
		if (sig) *sig = true;
	}, _WIN_W*_WIN_H);
}



void render() {
	auto t0 = NTime::now();
	// initialize window
	for (int i = 0, l = _WIN_W * _WIN_H; i < l; i++) _WINIMG[i] = 0;
	for (int i = 0; i < _WIN_W; i++) for (int j = 0; j < _WIN_H; j++) _DEPTHBUF[i][j] = INFINITY;
	getScreen(CamP, ScrO, ScrA, ScrB);


	render_RT_BVH();


	double t = fsec(NTime::now() - t0).count();
	sprintf(text, "[%d×%d]  %.1fms (%.1ffps)\n", _WIN_W, _WIN_H, 1000.0*t, 1. / t);
	SetWindowTextA(_HWND, text);
}


// ============================================== User ==============================================


void readBinarySTL(const char* filename) {
	FILE* fp = fopen(filename, "rb");
	fseek(fp, 80, SEEK_SET);
	fread(&STL_N, sizeof(int), 1, fp);
	STL = new Triangle[STL_N];
	dbgprint("%d\n", STL_N);
	auto readf = [&](double &x) {
		float t; fread(&t, sizeof(float), 1, fp);
		x = (double)t;
	};
	auto readTrig = [&](Triangle &T) {
		readf(T.n.x); readf(T.n.y); readf(T.n.z);
		readf(T.P.x); readf(T.P.y); readf(T.P.z);
		readf(T.A.x); readf(T.A.y); readf(T.A.z); T.A -= T.P;
		readf(T.B.x); readf(T.B.y); readf(T.B.z); T.B -= T.P;
		short c; fread(&c, 2, 1, fp);
		T.n = cross(T.A, T.B);
	};
	for (int i = 0; i < STL_N; i++) {
		readTrig(STL[i]);
	}
	fclose(fp);
}
void BoundingBox(Triangle *P, int N, vec3 &p0, vec3 &p1) {
	p0 = vec3(INFINITY), p1 = vec3(-INFINITY);
	for (int i = 0; i < N; i++) {
		p0 = pMin(pMin(p0, P[i].P), P[i].P + pMin(P[i].A, P[i].B));
		p1 = pMax(pMax(p1, P[i].P), P[i].P + pMax(P[i].A, P[i].B));
	}
}


bool inited = false;
void Init() {
	if (inited) return; inited = true;
	//readBinarySTL("D:\\eyes.stl");
	readBinarySTL("D:\\Coding\\Github\\Graphics\\ui\\3D Models\\Blender_Isosphere.stl");
	vec3 p0, p1; BoundingBox(STL, STL_N, p0, p1);
	vec3 c = 0.5*(p1 + p0), b = 0.5*(p1 - p0);
	double ir = 1.0 / cbrt(b.x*b.y*b.z);
	Center = vec3(vec2(0.0), b.z * ir);
	vec3 t = vec3(-c.x, -c.y, -p0.z);
	for (int i = 0; i < STL_N; i++) {
		STL[i].P = (STL[i].P + t) * ir;
		STL[i].A *= ir, STL[i].B *= ir;
		STL[i].n = cross(STL[i].A, STL[i].B);
	}
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
	if (Shift) {
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

	// drag to rotate scene
	if (mouse_down) {
		if (Shift) {
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

