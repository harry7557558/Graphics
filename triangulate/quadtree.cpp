// Graphing Implicit Curve

// Still have a bug and repeated samples


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
void drawDot(vec2 p, double r, COLORREF col);
#define drawLineF(p,q,col) drawLine(fromFloat(p),fromFloat(q),col)
#define drawDotF(p,r,col) drawDot(fromFloat(p),r,col)


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

	// list of vertice on a unit square
	const static ivec2 VERTICE_LIST[4] = {
		ivec2(0,0), ivec2(1,0), ivec2(1,1), ivec2(0,1)  // bottom-left, bottom-right, top-right, top-left
	};
	// list of edges connecting two vertices on a unit square
	const static ivec2 EDGE_LIST[4] = {
		ivec2(0,1), ivec2(1,2), ivec2(2,3), ivec2(3,0)  // bottom, right, top, left
	};

	// indicate segments connecting edges from bitwise compressed vertice singns
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

	// used in quadtree subdivision, which edge to which edge
	const static int TREE_EDGES[4][4][2] = {
		{{-1,-1},{1,3},{3,0},{-1,-1}},
		{{-1,-1},{-1,-1},{2,0},{0,1}},
		{{1,2},{-1,-1},{-1,-1},{3,1}},
		{{0,2},{2,3},{-1,-1},{-1,-1}}
	};
	// "inner" and "outer" edges when merging subtrees
	const static int TREE_INNER_EDGES[4][2][2] = {
		{{0,1},{1,3}},
		{{1,2},{2,0}},
		{{2,3},{3,1}},
		{{3,0},{0,2}}
	};
	const static int TREE_OUTER_EDGES[4][2][2] = {
		{{0,0},{1,0}},
		{{1,1},{2,1}},
		{{2,2},{3,2}},
		{{3,3},{0,3}}
	};

	const vec2 NAV = vec2(NAN);
}


struct quadtree_node {
	vec2 pos[4];
	double val[4];
	quadtree_node* children[4] = { 0,0,0,0 };
	~quadtree_node() {
		for (int i = 0; i < 4; i++) if (children[i]) {
			delete children[i]; children[i] = 0;
		}
	}
};

#include <algorithm>
void quadtree_recurse(quadtree_node* R, std::vector<vec2> edgepoints[4], std::vector<vec2> findlist, int recurse_remain);

void marchSquare(vec2 p0, vec2 p1, const ivec2 SEARCH_DIF, const int PLOT_DPS) {
	using namespace MARCH_LOOKUP_TABLES;
	vec2 dp = (p1 - p0) / vec2(SEARCH_DIF);
	double epsilon = 1e-5*min(dp.x, dp.y)*exp2(-PLOT_DPS);

	// initial samples
	double *val = new double[(SEARCH_DIF.x + 1)*(SEARCH_DIF.y + 1)];
	for (int i = 0; i <= SEARCH_DIF.x; i++) {
		for (int j = 0; j <= SEARCH_DIF.y; j++) {
			vec2 p = p0 + vec2(i, j)*dp;
			val[i*(SEARCH_DIF.y + 1) + j] = fun(p.x, p.y);
		}
	}
	auto getVal = [&](ivec2 ij)->double { return val[ij.x*(SEARCH_DIF.y + 1) + ij.y]; };

	// sample squares
	const int GRID_SIZE = SEARCH_DIF.x*SEARCH_DIF.y;
	quadtree_node *sqrs = new quadtree_node[GRID_SIZE];
	auto getSquare = [&](int x, int y)->quadtree_node* { return &sqrs[x*(SEARCH_DIF.y) + y]; };
	for (int i = 0; i < SEARCH_DIF.x; i++) {
		for (int j = 0; j < SEARCH_DIF.y; j++) {
			quadtree_node* qt = getSquare(i, j);
			for (int u = 0; u < 4; u++) {
				ivec2 ips = ivec2(i, j) + VERTICE_LIST[u];
				qt->pos[u] = p0 + vec2(ips)*dp;
				qt->val[u] = getVal(ips);
			}
		}
	}

	// grid to store "edge points" for identifying missed samples
	std::vector<vec2> *edge_points_pnt = 0, *contained_points_pnt = 0;
	if (PLOT_DPS > 0) {
		edge_points_pnt = new std::vector<vec2>[4*GRID_SIZE];
		contained_points_pnt = new std::vector<vec2>[GRID_SIZE];
	}
	auto getEdgePoints = [&](int x, int y)->std::vector<vec2>* { return &edge_points_pnt[4*(x*(SEARCH_DIF.y) + y)]; };
	auto getContainedPoints = [&](int x, int y)->std::vector<vec2>* { return &contained_points_pnt[x*(SEARCH_DIF.y) + y]; };

	// march squares
	for (int xi = 0; xi < SEARCH_DIF.x; xi++) {
		for (int yi = 0; yi < SEARCH_DIF.y; yi++) {

			// get signs and calculate index
			quadtree_node* qt = getSquare(xi, yi);
			vec2 pos[4]; double val[4];
			for (int u = 0; u < 4; u++) {
				pos[u] = qt->pos[u];
				val[u] = qt->val[u];
			}
			uint16_t index = calcIndex(val);

			// quadtree (recursive)
			if (PLOT_DPS > 0 && SEGMENT_TABLE[index][0] != -1) {
				std::vector<vec2> *edgepoints = getEdgePoints(xi, yi);
				quadtree_recurse(qt, edgepoints, std::vector<vec2>(), PLOT_DPS);
			}

			// add segments (standard marching square)
			else for (int u = 0; u < 4; u += 2) {
				int d0 = SEGMENT_TABLE[index][u];
				int d1 = SEGMENT_TABLE[index][u + 1];
				if (d0 == -1) break;
				Segments.push_back(segment(getInterpolation(pos, val, d0), getInterpolation(pos, val, d1)));
			}

			// visualize for debug
			if (SEGMENT_TABLE[index][0] != -1) drawBox(pos[0].x, pos[2].x, pos[0].y, pos[2].y, 0xFF0000);
		}
	}

	// check missed samples
	if (edge_points_pnt) {
		bool *march_needed_pnt = new bool[GRID_SIZE];
		auto march_needed = [&](ivec2 ij)->bool { return march_needed_pnt[ij.x*(SEARCH_DIF.y) + ij.y]; };
		bool foundMissed = false;
		do {
			for (int i = 0; i < GRID_SIZE; i++) march_needed_pnt[i] = false;
			foundMissed = false;

			// update contained points
			for (int xi = 0; xi < SEARCH_DIF.x; xi++) {
				for (int yi = 0; yi < SEARCH_DIF.y; yi++) {
					std::vector<vec2> *edgepoints = getEdgePoints(xi, yi);
					if (yi != 0) getContainedPoints(xi, yi - 1)->insert(getContainedPoints(xi, yi - 1)->begin(), edgepoints[0].begin(), edgepoints[0].end());
					if (xi + 1 != SEARCH_DIF.x) getContainedPoints(xi + 1, yi)->insert(getContainedPoints(xi + 1, yi)->begin(), edgepoints[1].begin(), edgepoints[1].end());
					if (yi + 1 != SEARCH_DIF.y) getContainedPoints(xi, yi + 1)->insert(getContainedPoints(xi, yi + 1)->begin(), edgepoints[2].begin(), edgepoints[2].end());
					if (xi != 0) getContainedPoints(xi - 1, yi)->insert(getContainedPoints(xi - 1, yi)->begin(), edgepoints[3].begin(), edgepoints[3].end());
				}
			}

			for (int xi = 0; xi < SEARCH_DIF.x; xi++) {
				for (int yi = 0; yi < SEARCH_DIF.y; yi++) {
					// find points that are connected but not contained
					std::vector<vec2> ep, *epp = getEdgePoints(xi, yi);
					for (int i = 0; i < 4; i++) ep.insert(ep.end(), epp[i].begin(), epp[i].end()), epp[i].clear();
					std::vector<vec2> cp = *getContainedPoints(xi, yi); getContainedPoints(xi, yi)->clear();
					int cpn = cp.size(), epn = ep.size();
					std::vector<vec2> miss_list;
					for (int i = 0; i < cpn; i++) {
						vec2 p = cp[i];
						bool has = false;
						for (int j = 0; j < epn; j++)
							if ((p - ep[j]).sqr() < epsilon*epsilon) { has = true; break; }
						if (!has) miss_list.push_back(p);
					}
					// re-sample these points
					if (!miss_list.empty()) {
						//foundMissed = true;  // still has a bug: this brings it into an infinite loop
						quadtree_recurse(getSquare(xi, yi), epp, miss_list, PLOT_DPS);
					}
				}
			}
		} while (foundMissed);
		delete march_needed_pnt;
	}

	delete val;
	delete[] sqrs;
	if (PLOT_DPS > 0) {
		delete[] edge_points_pnt;
		delete[] contained_points_pnt;
	}
}

void quadtree_recurse(quadtree_node* R, std::vector<vec2> edgepoints[4], std::vector<vec2> findlist, int recurse_remain) {
	using namespace MARCH_LOOKUP_TABLES;
	vec2 p0 = R->pos[0], dp = R->pos[2] - R->pos[0];
	double epsilon = 1e-5*min(dp.x, dp.y)*exp2(-recurse_remain);

	if (recurse_remain > 0) {
		drawBox(R->pos[0].x, R->pos[2].x, R->pos[0].y, R->pos[2].y, 0x4040FF);

		// initialize child nodes
		bool alreadyMarched[4] = { false, false, false, false };
		for (int i = 0; i < 4; i++) {
			if (!R->children[i]) R->children[i] = new quadtree_node;
			else alreadyMarched[i] = true;
			quadtree_node *qt = R->children[i];
			for (int u = 0; u < 4; u++) {
				qt->pos[u] = p0 + (0.5*vec2(VERTICE_LIST[i] + VERTICE_LIST[u]))*dp;
			}
			for (int u = 0; u < 4; u++) {
				qt->val[u] = fun(qt->pos[u].x, qt->pos[u].y);
			}
		}

		// decide whether to "sub"-march or not
		int index[4];
		for (int i = 0; i < 4; i++) index[i] = calcIndex(R->children[i]->val);
		std::vector<vec2> findl[4];
		for (int i = 0; i < 4; i++) {
			vec2 q0 = p0 + 0.5*vec2(VERTICE_LIST[i])*dp;
			vec2 q1 = q0 + 0.5*dp + vec2(epsilon); q0 -= vec2(epsilon);
			for (int d = 0, n = findlist.size(); d < n; d++) {
				vec2 q = findlist[d];
				if (q.x > q0.x && q.x<q1.x && q.y>q0.y && q.y < q1.y) findl[i].push_back(q);
			}
		}
		bool toMarch[4] = { false, false, false, false };
		for (int i = 0; i < 4; i++) {
			toMarch[i] = (SEGMENT_TABLE[index[i]][0] != -1 && !alreadyMarched[i]) || !findl[i].empty();
		}
		// recursive march
		std::vector<vec2> edgepoints_s[4][4];
		for (int i = 0; i < 4; i++) if (toMarch[i]) {
			quadtree_recurse(R->children[i], edgepoints_s[i], findl[i], recurse_remain - 1);
		}

		// check the inner edges to see if there are missed parts
		std::vector<vec2> missed[4];  // index is assigned to squares
		for (int c = 0; c < 4; c++) {
			int sqr0 = TREE_INNER_EDGES[c][0][0], sqr1 = TREE_INNER_EDGES[c][1][0];
			std::vector<vec2> e0 = edgepoints_s[sqr0][TREE_INNER_EDGES[c][0][1]];
			std::vector<vec2> e1 = edgepoints_s[sqr1][TREE_INNER_EDGES[c][1][1]];
			struct vec2s { vec2 p; int sqr; };
			std::vector<vec2s> et;
			for (int i = 0, n0 = e0.size(); i < n0; i++)
				et.push_back(vec2s{ e0[i], sqr1 });
			for (int i = 0, n1 = e1.size(); i < n1; i++)
				et.push_back(vec2s{ e1[i],sqr0 });
			std::sort(et.begin(), et.end(), [](vec2s a, vec2s b) { return a.p.x + a.p.y < b.p.x + b.p.y; });
			for (int i = 0, etl = et.size(); i < etl;) {
				if (i + 1 == etl) {
					missed[et[i].sqr].push_back(et[i].p);  // remain 1
					break;
				}
				if ((et[i].p - et[i + 1].p).sqr() < epsilon*epsilon) {
					if (et[i].sqr == et[i + 1].sqr) throw(__LINE__);  // should never happen
					i += 2;  // duplicate, pass
				}
				else {
					missed[et[i].sqr].push_back(et[i].p);  // missed
					i++;
				}
			}
		}
		// fix missed samples
		for (int c = 0; c < 4; c++) {
			if (!missed[c].empty()) {
				for (int u = 0; u < 4; u++) edgepoints_s[c][u].clear();  // should not matter
				quadtree_recurse(R->children[c], edgepoints_s[c], missed[c], recurse_remain - 1);
			}
		}

		// merge outer edges to be accessed by the call function
		for (int c = 0; c < 4; c++) {
			std::vector<vec2> e0 = edgepoints_s[TREE_OUTER_EDGES[c][0][0]][TREE_OUTER_EDGES[c][0][1]];
			std::vector<vec2> e1 = edgepoints_s[TREE_OUTER_EDGES[c][1][0]][TREE_OUTER_EDGES[c][1][1]];
			edgepoints[c].insert(edgepoints[c].begin(), e0.begin(), e0.end());
			edgepoints[c].insert(edgepoints[c].begin(), e1.begin(), e1.end());
		}

	}

	// search limit exceeded, create segments
	else {
		int index = calcIndex(R->val);
		for (int u = 0; u < 4; u += 2) {
			int d0 = SEGMENT_TABLE[index][u];
			int d1 = SEGMENT_TABLE[index][u + 1];
			if (d0 == -1) break;
			vec2 p0 = getInterpolation(R->pos, R->val, d0);
			vec2 p1 = getInterpolation(R->pos, R->val, d1);

			// need to check if the point is already added
			if (0) {
				bool alreadyHas = false;
				for (int i = 0, n = edgepoints[d0].size(); i < n; i++)
					if ((edgepoints[d0][i] - p0).sqr() < epsilon*epsilon) { alreadyHas = true; break; }
				if (alreadyHas) continue;
				for (int i = 0, n = edgepoints[d1].size(); i < n; i++)
					if ((edgepoints[d1][i] - p0).sqr() < epsilon*epsilon) { alreadyHas = true; break; }
				if (alreadyHas) continue;
			}

			// good
			edgepoints[d0].push_back(p0), edgepoints[d1].push_back(p1);
			Segments.push_back(segment(edgepoints[d0].back(), edgepoints[d1].back()));
		}
		drawBox(R->pos[0].x, R->pos[2].x, R->pos[0].y, R->pos[2].y, 0x00FF00);
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
	marchSquare(p0, p1, ivec2(16, 16), 4);
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

