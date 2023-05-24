// Graphing Implicit Curve in 2d


/* ====================== Instructions ======================

 *  Move View:                  drag background
 *  Zoom:                       mouse scroll
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



double sdLine(vec2 p, vec2 a, vec2 b) { vec2 pa = p - a, ba = b - a; return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0., 1.)); }
double snowflake(vec2 p, double t) {
	double a = abs(fmod(atan2(p.y, p.x) + .5*t, PI / 3.)) - PI / 6.;
	p = cossin(a)*length(p)*(1.0 + 0.3*cos(t)), t = 0.9 - 0.3*cos(t);
	return min(min(p.x - 0.16, sdLine(p, vec2(0.), vec2(1.22, 0.))),
		min(sdLine(vec2(p.x, abs(p.y)), vec2(0.45, 0.), vec2(0.45 + 0.42*cos(t), 0.42*sin(t))),
			sdLine(vec2(p.x, abs(p.y)), vec2(0.8, 0.), vec2(0.8 + 0.35*cos(t), 0.35*sin(t)))
		));
}
double snowflake_b(vec2 p, double t) {
	return abs(snowflake(p, t) - 0.04) - 0.01;
}
double snowflake_o(vec2 p, double t) {
	return sin(20.*PI*(snowflake(p, t) - 0.05)) + 0.5*p.sqr();
}

#define RECORD_SAMPLES 0
#include <vector>
#include <algorithm>
std::vector<vec2> recorded_samples;

int evals;
double fun(vec2 p) {
	double x = p.x, y = p.y;
	evals++;
	if (RECORD_SAMPLES) recorded_samples.push_back(p);
	//return x * x + y * y - 1;  // 3.141593
	//return hypot(x, y) - 1;  // 3.141593
	//return x * x*x*(x - 2) + y * y*y*(y - 2) + x;  // 5.215079
	//return x * x*x*x + y * y*y*y + x * y;  // 0.785398
	//return abs(abs(max(abs(x) - 0.9876, abs(y + 0.1*sin(3.*x)) - 0.4321)) - 0.3456) - 0.1234;  // 2.726510
	//return max(2.*y - sin(10. * x), 4.*x*x + 4.*y * y - 9.);  // 3.534292
	//return max(abs(x) - abs(y) - 1, x*x + y * y - 2);  // 5.968039
	//return max(.5 - abs(x*y), x*x + 2 * y*y - 3);  // 1.813026
	//return max(sin(10*x) + cos(10*y) + 1., x*x + y * y - 1.);  // 0.57[3-4]
	return snowflake(vec2(x, y), 1.7) - 0.05;  // 1.787337
	//return snowflake_b(vec2(x, y), 1.7);
	//return snowflake_o(vec2(x, y), 1.7);
}


// forward declarations of rendering functions
void drawLine(vec2 p, vec2 q, COLORREF col);
void drawBox(vec2 p0, vec2 p1, COLORREF col);
void drawDot(vec2 p, double r, COLORREF col);
#define drawLineF(p,q,col) drawLine(fromFloat(p),fromFloat(q),col)
#define drawBoxF(p0,p1,col) drawBox(fromFloat(p0),fromFloat(p1),col)
#define drawDotF(p,r,col) drawDot(fromFloat(p),r,col)

// ============================================ Marching ============================================


// global array to store marched segments
struct segment {
	vec2 p[2];
	vec2& operator[](int d) { return p[d]; }
	segment() {}
	segment(vec2 a, vec2 b) { p[0] = a, p[1] = b; }
};
std::vector<segment> Segments;




// list of vertice on a unit square
const static ivec2 VERTICE_LIST[4] = {
	ivec2(0,0), ivec2(1,0), ivec2(1,1), ivec2(0,1)  // bottom-left, bottom-right, top-right, top-left
};
const static int VERTICE_LIST_INV[2][2] = {  // reverse search of VERTICE_LIST
	{0, 3}, {1, 2},
};
// list of edges connecting two vertices on a unit square
const static ivec2 EDGE_LIST[4] = {
	ivec2(0,1), ivec2(1,2), ivec2(2,3), ivec2(3,0)  // bottom, right, top, left; opposite: (+2)%4
};
const static ivec2 EDGE_DIR[4] = {
	ivec2(0,-1), ivec2(1,0), ivec2(0,1), ivec2(-1,0)
};

// for reconstruction
// indicate segments connecting edges from bitwise compressed vertice signs
// max 2 segments, distinct vertices
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

// convert sample values to index for table lookup
uint16_t calcIndex(const double val[4]) {
	return uint16_t(val[0] < 0) | (uint16_t(val[1] < 0) << 1) | (uint16_t(val[2] < 0) << 2) | (uint16_t(val[3] < 0) << 3);
}

// linear interpolation on an edge
vec2 getInterpolation(vec2 pos[4], double val[4], int i) {
	double v0 = val[EDGE_LIST[i].x];
	double v1 = val[EDGE_LIST[i].y];
	vec2 p0 = pos[EDGE_LIST[i].x];
	vec2 p1 = pos[EDGE_LIST[i].y];
	return p0 + (v0 / (v0 - v1))*(p1 - p0);
};





vec2 p0, p1;  // search bounds
#if 1
const ivec2 SEARCH_DIF = ivec2(16, 10);
const int PLOT_DPS = 4;
#else
const ivec2 SEARCH_DIF = 6 * ivec2(16, 10);
const int PLOT_DPS = 4;
#endif
const int PLOT_SIZE = 1 << PLOT_DPS;
const ivec2 GRID_SIZE = SEARCH_DIF * PLOT_SIZE;  // define grid id
vec2 i2f(ivec2 p) {  // position ID to position
	vec2 d = vec2(p) / vec2(GRID_SIZE);
	return p0 * (vec2(1.) - d) + p1 * d;
}


// quadtree
// all integer coordinates are absolute
double getSample_global(ivec2 p);
class quadtree_node {
public:
	ivec2 p[4]; // point IDs
	double v[4]; // only the first one really matter
	int size;  // top right: p+ivec2(size)
	int index;  // calculated according to signs of v for table lookup
	quadtree_node *c[4];  // child nodes
	bool hasSignChange[4];  // indicate whether there is a sign change at each edge
	bool edge_checked[4];  // used in looking for missed samples, indicate whether the edge is already checked
	quadtree_node(int size = 0, ivec2 p = ivec2(-1)) {
		this->p[0] = this->p[1] = this->p[2] = this->p[3] = p;
		if (p != ivec2(-1)) {
			for (int i = 1; i < 4; i++)
				this->p[i] = p + VERTICE_LIST[i] * size;
		}
		v[0] = v[1] = v[2] = v[3] = NAN;
		this->size = size;
		this->index = -1;
		c[0] = c[1] = c[2] = c[3] = nullptr;
		hasSignChange[0] = hasSignChange[1] = hasSignChange[2] = hasSignChange[3] = false;
		edge_checked[0] = edge_checked[1] = edge_checked[2] = edge_checked[3] = false;
	}
	~quadtree_node() {
		for (int i = 0; i < 4; i++) if (c[i]) {
			delete c[i]; c[i] = 0;
		}
	}
	double getSample(ivec2 q) {
		if (q == p[0]) {
			if (isnan(v[0])) {
				if (c[0]) {
					v[0] = c[0]->getSample(q);
				}
				else v[0] = fun(i2f(p[0]));
			}
			return v[0];
		}
		ivec2 d = (q - p[0]) / (size >> 1);
		int i = VERTICE_LIST_INV[d.x][d.y];
		if (!c[i]) {
			c[i] = new quadtree_node(size / 2, p[0] + d * (size / 2));
		}
		return c[i]->getSample(q);
	}
	quadtree_node* getGrid(ivec2 q, int sz) {
		if (q == p[0] && sz == size) {
			if (isnan(v[0])) {
				v[0] = getSample_global(p[0]);
			}
			for (int i = 1; i < 4; i++) if (isnan(v[i])) {
				v[i] = getSample_global(p[i]);
			}
			return this;
		}
#ifdef _DEBUG
		if (q % sz != ivec2(0)) throw(__LINE__);
		if (sz > size) throw(__LINE__);
#endif
		ivec2 d = (q - p[0]) / (size >> 1);
		int i = VERTICE_LIST_INV[d.x][d.y];
		if (!c[i]) {
			c[i] = new quadtree_node(size / 2, p[0] + d * (size / 2));
		}
		return c[i]->getGrid(q, sz);
	}
	int calcIndex() {
		if (isnan(v[0] + v[1] + v[2] + v[3])) return (index = 0);
		return (index = int(v[0] < 0) | (int(v[1] < 0) << 1) | (int(v[2] < 0) << 2) | (int(v[3] < 0) << 3));
	}
	void subdivide();
};
quadtree_node** quadtree = 0;  // a grid [x][y]
void create_quadtree() {  // sample tree initialization
	quadtree = new quadtree_node*[SEARCH_DIF.x + 1];
	for (int x = 0; x <= SEARCH_DIF.x; x++) {
		quadtree[x] = new quadtree_node[SEARCH_DIF.y + 1];
		for (int y = 0; y <= SEARCH_DIF.y; y++) {
			quadtree[x][y] = quadtree_node(PLOT_SIZE, ivec2(x, y)*PLOT_SIZE);
		}
	}
}
void destroy_quadtree() {  // sample tree destruction
	for (int x = 0; x <= SEARCH_DIF.x; x++)
		delete[] quadtree[x];
	delete quadtree;
	quadtree = 0;
}
double getSample_global(ivec2 p) {  // access a sample on the sample tree
	ivec2 pi = p / PLOT_SIZE;
	return quadtree[pi.x][pi.y].getSample(p);
}
quadtree_node* getGrid_global(ivec2 p, int size) {
	ivec2 pi = p / PLOT_SIZE;
	return quadtree[pi.x][pi.y].getGrid(p, size);
}


// marching cells
std::vector<quadtree_node*> cells;


// grid subdivision
void quadtree_node::subdivide() {
	// assume v[0]-v[4] are already initialized
	for (int u = 0; u < 4; u++) if (!c[u])
		c[u] = new quadtree_node(size / 2, p[0] + VERTICE_LIST[u] * (size / 2));
	// LB, B, RB, R, TR, T, TL, L, C
	double samples[9] = { NAN,NAN,NAN,NAN,NAN,NAN,NAN,NAN,NAN };
	const static bool SEGMENT_TABLE[16][9] = {
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1 }, // 0000, force subdivide
		{ 1, 1, 1, 0, 1, 0, 1, 1, 1 }, // 1000
		{ 1, 1, 1, 1, 1, 0, 1, 0, 1 }, // 0100
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1 }, // 1100
		{ 1, 0, 1, 1, 1, 1, 1, 0, 1 }, // 0010
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1 }, // 1010
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1 }, // 0110
		{ 1, 0, 1, 0, 1, 1, 1, 1, 1 }, // 1110
		{ 1, 0, 1, 0, 1, 1, 1, 1, 1 }, // 0001
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1 }, // 1001
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1 }, // 0101
		{ 1, 0, 1, 1, 1, 1, 1, 0, 1 }, // 1101
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1 }, // 0011
		{ 1, 1, 1, 1, 1, 0, 1, 0, 1 }, // 1011
		{ 1, 1, 1, 0, 1, 0, 1, 1, 1 }, // 0111
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1 }, // 1111, force subdivide
	};
	auto s = SEGMENT_TABLE[calcIndex()];
	// SKIP: has potential to save 20% samples
	// most cells missed here should not matter since there is a check later
	// in rare case, this causes problems
	// check the sign of the sample in the center may solve this problem
	const int DISABLE_SKIP = true;
	if (DISABLE_SKIP || s[0]) samples[0] = v[0];
	if (DISABLE_SKIP || s[2]) samples[2] = v[1];
	if (DISABLE_SKIP || s[4]) samples[4] = v[2];
	if (DISABLE_SKIP || s[6]) samples[6] = v[3];
	if (DISABLE_SKIP || s[1]) samples[1] = isnan(c[1]->v[0]) ? fun(i2f(c[1]->p[0])) : c[1]->v[0];
	if (DISABLE_SKIP || s[3]) samples[3] = getSample_global(c[1]->p[2]);
	if (DISABLE_SKIP || s[5]) samples[5] = getSample_global(c[3]->p[2]);
	if (DISABLE_SKIP || s[7]) samples[7] = isnan(c[3]->v[0]) ? fun(i2f(c[3]->p[0])) : c[3]->v[0];
	if (DISABLE_SKIP || s[8]) samples[8] = fun(i2f(c[2]->p[0]));  // must be used
	const static int SUBDIV_LOOKUP[4][4] = {
		{0,1,8,7}, {1,2,3,8}, {8,3,4,5}, {7,8,5,6}
	};
	for (int u = 0; u < 4; u++) for (int v = 0; v < 4; v++)
		c[u]->v[v] = samples[SUBDIV_LOOKUP[u][v]];
}


void marchSquare(vec2 _p0, vec2 _p1) {
	p0 = _p0, p1 = _p1;

	// initialize quadtree root
	create_quadtree();
	for (int x = 0; x <= SEARCH_DIF.x; x++) {
		for (int y = 0; y <= SEARCH_DIF.y; y++) {
			quadtree[x][y].v[0] = fun(i2f(quadtree[x][y].p[0] = ivec2(x, y)*PLOT_SIZE));
		}
	}
	for (int x = 0; x < SEARCH_DIF.x; x++) {
		for (int y = 0; y < SEARCH_DIF.y; y++) {
			for (int u = 1; u < 4; u++) {
				ivec2 p = ivec2(x, y) + VERTICE_LIST[u];
				quadtree[x][y].p[u] = quadtree[p.x][p.y].p[0];
				quadtree[x][y].v[u] = quadtree[p.x][p.y].v[0];
			}
		}
	}

	// initial sample cells
	cells.clear();
	for (int x = 0; x < SEARCH_DIF.x; x++) {
		for (int y = 0; y < SEARCH_DIF.y; y++) {
			quadtree_node *n = &quadtree[x][y];
			if (SEGMENT_TABLE[n->calcIndex()][0] != -1) {
				cells.push_back(n);
			}
		}
	}
	// debug visualization
	for (int i = 0, cn = cells.size(); i < cn; i++)
		drawBoxF(i2f(cells[i]->p[0]), i2f(cells[i]->p[2]), 0x800000);

	// subdivide grid cells
	for (int size = PLOT_SIZE; size > 1; size >>= 1) {
		std::vector<quadtree_node*> new_cells;
		int s2 = size / 2;
		for (int i = 0, cn = cells.size(); i < cn; i++) {
			quadtree_node* ci = cells[i];
			ci->subdivide();
			for (int u = 0; u < 4; u++) {
				if (SEGMENT_TABLE[ci->c[u]->calcIndex()][0] != -1) {
					new_cells.push_back(ci->c[u]);
				}
			}
		}
		cells = new_cells;
		// debug visualization
		for (int i = 0, cn = cells.size(); i < cn; i++)
			drawBoxF(i2f(cells[i]->p[0]), i2f(cells[i]->p[2]), 0x600060);

		// try to add missed samples
		for (int i = 0; i < (int)cells.size(); i++) {
			quadtree_node* ci = cells[i];
			for (int u = 0; u < 4; u++) {
				ci->hasSignChange[u] = signbit(ci->v[EDGE_LIST[u].x]) ^ signbit(ci->v[EDGE_LIST[u].y]);
			}
		}
		for (int i = 0; i < (int)cells.size(); i++) {
			quadtree_node* ci = cells[i];
			for (int u = 0; u < 4; u++) if (ci->hasSignChange[u] && !ci->edge_checked[u]) {
				ivec2 nb_p = ci->p[0] + EDGE_DIR[u] * ci->size;
				if (nb_p.x >= 0 && nb_p.y >= 0 && nb_p.x < GRID_SIZE.x && nb_p.y < GRID_SIZE.y) {
					quadtree_node* nb = getGrid_global(nb_p, ci->size);
					if (!nb->hasSignChange[(u + 2) % 4]) {
						for (int u = 0; u < 4; u++)
							nb->hasSignChange[u] = signbit(nb->v[EDGE_LIST[u].x]) ^ signbit(nb->v[EDGE_LIST[u].y]);
						drawBoxF(i2f(nb->p[0]), i2f(nb->p[2]), 0x606000);  // debug visualization
						cells.push_back(nb);
					}
					nb->edge_checked[(u + 2) % 4] = true;
				}
				ci->edge_checked[u] = true;
			}
		}
	}

	// reconstruct segments
	for (int i = 0, cn = cells.size(); i < cn; i++) {
		vec2 p[4];
		for (int j = 0; j < 4; j++) p[j] = i2f(cells[i]->p[j]);
		const auto Si = SEGMENT_TABLE[cells[i]->calcIndex()];
		for (int u = 0; u < 4 && Si[u] != -1; u += 2) {
			vec2 a = getInterpolation(p, cells[i]->v, Si[u]);
			vec2 b = getInterpolation(p, cells[i]->v, Si[u + 1]);
			Segments.push_back(segment(a, b));
		}
		// debug visualization
		drawBoxF(i2f(cells[i]->p[0]), i2f(cells[i]->p[2]), 0x0000FF);
	}

	// clean up
	cells.clear();
	destroy_quadtree();
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
void drawBox(vec2 p0, vec2 p1, COLORREF col) {
	//return;
	drawLine(vec2(p0.x, p0.y), vec2(p1.x, p0.y), col);
	drawLine(vec2(p0.x, p1.y), vec2(p1.x, p1.y), col);
	drawLine(vec2(p0.x, p0.y), vec2(p0.x, p1.y), col);
	drawLine(vec2(p1.x, p0.y), vec2(p1.x, p1.y), col);
};
void drawDot(vec2 p, double r, COLORREF col) {
	int x0 = max((int)(p.x - r), 0), x1 = min((int)(p.x + r + 1), _WIN_W - 1);
	int y0 = max((int)(p.y - r), 0), y1 = min((int)(p.y + r + 1), _WIN_H - 1);
	for (int x = x0; x <= x1; x++) for (int y = y0; y <= y1; y++) {
		if ((vec2(x, y) - p).sqr() < r*r) Canvas(x, y) = col;
	}
}


void remarch() {
	int PC = (int)(0.5*log2(_WIN_W*_WIN_H) + 0.5);  // depth
	//SEARCH_DPS = max(PC - 3, 6), PLOT_DPS = max(PC - 1, 12);
	vec2 p0 = fromInt(vec2(0, 0)), p1 = fromInt(vec2(_WIN_W, _WIN_H));  // starting position and increasement
	if (0) p0 = vec2(-2.5), p1 = vec2(2.5);
	evals = 0;
	Segments.clear();
	recorded_samples.clear();
	marchSquare(p0, p1);
}

void render() {
	// debug
	auto t1 = NTime::now();
	double dt = std::chrono::duration<double>(t1 - t0).count();
	printf("[%d×%d] time elapsed: %.1fms (%.1ffps)\n", _WIN_W, _WIN_H, 1000.0*dt, 1. / dt);
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

	// debug duplicate samples
	if (RECORD_SAMPLES) {
		std::sort(recorded_samples.begin(), recorded_samples.end(), [](vec2 a, vec2 b) { return a.x == b.x ? a.y < b.y : a.x < b.x; });
		for (int i = 0; i < (int)recorded_samples.size(); i++) {
			drawDotF(recorded_samples[i], 1, 0xFFFFFF);
		}
		int duplicate_count = 0;
		for (int i = 1; i < (int)recorded_samples.size(); i++) {
			if (recorded_samples[i] == recorded_samples[i - 1]) {
				duplicate_count++;
				drawDotF(recorded_samples[i], 2, 0x80FF00);
			}
		}
		if (duplicate_count) printf("%d duplicate samples\n", duplicate_count);
	}

	// rendering
	for (int i = 0; i < SN; i++) {
		vec2 a = Segments[i][0], b = Segments[i][1], c = (a + b)*0.5;
		drawLineF(a, b, 0xFFFFFF);
		drawLineF(c, c + (b - a).rot(), 0xFFFF00);
	}

	// calculate area
	double Area = 0;
	for (int i = 0; i < SN; i++) {
		Area += det(Segments[i][0], Segments[i][1]);
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

