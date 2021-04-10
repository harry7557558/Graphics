// a very limited STL and PLY file reader for ply_viewer.cpp


struct ply_triangle {
	int v[3];
	int& operator[] (int d) {
		return v[d];
	}
};



COLORREF toCOLORREF(vec3f c);

bool readPLY(FILE* fp, vec3f* &Vs, ply_triangle* &Fs, int &VN, int &FN, COLORREF* &v_col) {
	if (fp == 0) return false;

	const int MAX_SIZE = 0x10000;
	int buffer_size = MAX_SIZE;
	int index = buffer_size;
	uint8_t buf[MAX_SIZE + 1]; buf[MAX_SIZE] = 0;
	auto next_byte = [&]()->int {
		if (index >= buffer_size) {
			if (buffer_size == MAX_SIZE)
				buffer_size = (int)fread(buf, 1, MAX_SIZE, fp), index = 0;
			else return EOF;
		}
		return (int)buf[index++];
	};
	auto ignore_whitespace = [&]()->bool {
		for (int i = 0;; i++) {
			if (index >= buffer_size) {
				if (buffer_size == MAX_SIZE)
					buffer_size = (int)fread(buf, 1, MAX_SIZE, fp), index = 0;
				else return true;
			}
			char c = buf[index++];
			if (c != ' ' && c != '\r' && c != '\n') {
				index--; return i != 0;
			}
		}
	};
	auto check_str = [&](const char* s)->bool {
		while (*s) {
			if (*(s++) != next_byte()) return false;
		}
		return true;
	};

	auto read_string = [&](char end_char)->std::string {
		std::string s;
		for (int i = 0; ; i++) {
			char c = next_byte();
			if (c == end_char) {
				index--;
				if (end_char == '\n' && s.back() == '\r') s.pop_back();
				return s;
			}
			if (c == 0 || c == EOF) throw(c);
			s.push_back(c);
		}
	};

	if (!check_str("ply")) return false;
	if (!ignore_whitespace()) return false;

	std::vector<std::string> header_lines;
	try {
		while (1) {
			std::string s = read_string('\n');
			if (s == "end_header") break;
			header_lines.push_back(s);
			ignore_whitespace();
		}
	} catch (...) {
		return false;
	}



	const int ASCII = 0, BINARY_BIG_ENDIAN = 1, BINARY_LITTLE_ENDIAN = 2;
	int format = -1;

	VN = FN = -1;

	int xi = -1, yi = -1, zi = -1, property_index = 0;
	int ri = -1, gi = -1, bi = -1;
	std::vector<int> vertex_size_list;  // in bytes
	std::string element_name = "";

	auto split_string = [](std::string &nm, std::string &s) {
		int first_space = (int)s.find(' ');
		if (first_space <= 0) return false;
		nm = s.substr(0, first_space);
		s = s.substr(first_space + 1, s.size() - first_space - 1);
		return true;
	};
	for (std::string s : header_lines) {
		std::string nm;
		if (!split_string(nm, s)) return false;

		if (nm == "format") {
			if (FN != -1 || VN != -1) return false;
			if (s == "ascii 1.0") format = ASCII;
			if (s == "binary_big_endian 1.0") format = BINARY_BIG_ENDIAN;
			if (s == "binary_little_endian 1.0") format = BINARY_LITTLE_ENDIAN;
		}

		else if (nm == "element") {
			if (!split_string(element_name, s)) return false;
			int d = std::stoi(s);
			if (element_name == "vertex") VN = d;
			else if (element_name == "face") {
				FN = d;
				if (VN == -1) return false;
			}
			else {
				if (FN == -1 || VN == -1) return false;
			}
		}

		else if (nm == "property") {
			if (element_name != "vertex" && element_name != "face") continue;
			std::string type;
			if (!split_string(type, s)) return false;
			if (type == "list") {
				if (!split_string(type, s)) return false;
				if (type != "uchar" && type != "char" && type != "int8" && type != "uint8") return false;
				if (!split_string(type, s)) return false;
				if (type != "uint" && type != "int" && type != "int32" && type != "uint32") return false;
				if (s != "vertex_indices" && s != "vertex_index") return false;
				if (FN == -1) return false;
			}
			else {
				if (element_name == "vertex") {
					if (type == "uchar" || type == "char" || type == "int8" || type == "uint8") vertex_size_list.push_back(1);
					else if (type == "short" || type == "ushort" || type == "int16" || type == "uint16") vertex_size_list.push_back(2);
					else if (type == "float" || type == "float32" || type == "int" || type == "uint" || type == "int32" || type == "uint32") vertex_size_list.push_back(4);
					else if (type == "double" || type == "float64" || type == "int64" || type == "uint64") vertex_size_list.push_back(8);
					else return false;

					if (type == "float" || type == "float32") {
						if (s == "x") xi = property_index;
						if (s == "y") yi = property_index;
						if (s == "z") zi = property_index;
					}
					else if (s == "x" || s == "y" || s == "z") return false;

					if (s == "red" || s == "green" || s == "blue") {
						if (s == "red") ri = property_index;
						if (s == "green") gi = property_index;
						if (s == "blue") bi = property_index;
						if (type != "uchar" && type != "uint8") return false;
					}

					property_index += 1;
				}
			}
		}

		else if (nm != "comment") return false;

	}

	if (format == -1 || VN == -1 || FN == -1 || xi == -1 || yi == -1 || zi == -1) return false;

	bool has_color = ri != -1 && gi != -1 && bi != -1;
	if (has_color) v_col = new COLORREF[VN];


	Vs = new vec3f[VN]; Fs = new ply_triangle[FN];

	if (format == ASCII) {

		char c[64];
		auto readFloat = [&]()->float {
			ignore_whitespace();
			for (int i = 0; i < 64; i++) {
				c[i] = next_byte();
				if (c[i] >= 0 && c[i] <= ' ') {
					c[i] = 0; break;
				}
			}
			//return std::stof(c);
			return strtof(c, NULL);
		};
		auto readInt = [&]()->int {
			ignore_whitespace();
			for (int i = 0; i < 64; i++) {
				c[i] = next_byte();
				if (c[i] >= 0 && c[i] <= ' ') {
					if (i == 0) return -1;
					c[i] = 0; break;
				}
			}
			//return std::stoi(c);
			return atoi(c);
		};

		float *fs = new float[property_index];
		for (int i = 0; i < VN; i++) {
			for (int u = 0; u < property_index; u++) fs[u] = readFloat();
			Vs[i].x = fs[xi], Vs[i].y = fs[yi], Vs[i].z = fs[zi];
			if (has_color) {
				vec3f col = (vec3f(fs[ri], fs[gi], fs[bi]) + vec3f(0.5)) / 255.;
				v_col[i] = toCOLORREF(col);
			}
		}
		delete fs;

		for (int i = 0; i < FN; i++) {
			int n = readInt();
			if (n != 3) return false;
			for (int u = 0; u < 3; u++) {
				int d = readInt();
				if (d >= 0 && d < VN) Fs[i][u] = d;
				else return false;
			}
		}
	}

	else if (format == BINARY_LITTLE_ENDIAN) {

		auto read32 = [&]()->uint32_t {
			uint32_t x = 0;
			for (int i = 0; i < 4; i++) {
				uint8_t c = next_byte();
				x = x | (uint32_t(c) << (8 * i));
			}
			return x;
		};

		if (next_byte() != '\n') return false;

		float *fs = new float[property_index];
		uint32_t cr = 0, cg = 0, cb = 0;
		for (int i = 0; i < VN; i++) {
			for (int u = 0; u < property_index; u++) {
				if (u == xi || u == yi || u == zi) *(uint32_t*)&fs[u] = read32();
				else if (u == ri) cr = next_byte();
				else if (u == gi) cg = next_byte();
				else if (u == bi) cb = next_byte();
				else for (int _ = 0; _ < vertex_size_list[u]; _++) next_byte();
			}
			Vs[i].x = fs[xi], Vs[i].y = fs[yi], Vs[i].z = fs[zi];
			if (has_color) v_col[i] = (cr << 16) | (cg << 8) | cb;
		}
		delete fs;

		for (int i = 0; i < FN; i++) {
			int n = (int)next_byte();
			if (n != 3) return false;
			for (int u = 0; u < 3; u++) {
				int d = (int)read32();
				if (d >= 0 && d < VN) Fs[i][u] = d;
				else return false;
			}
		}

	}

	else if (format == BINARY_BIG_ENDIAN) {

		auto read32 = [&]()->uint32_t {
			uint32_t x = 0;
			for (int i = 3; i >= 0; i--) {
				uint8_t c = next_byte();
				x = x | (uint32_t(c) << (8 * i));
			}
			return x;
		};

		if (next_byte() != '\n') return false;

		float *fs = new float[property_index];
		uint32_t cr = 0, cg = 0, cb = 0;
		for (int i = 0; i < VN; i++) {
			for (int u = 0; u < property_index; u++) {
				if (u == xi || u == yi || u == zi) *(uint32_t*)&fs[u] = read32();
				else if (u == ri) cr = next_byte();
				else if (u == gi) cg = next_byte();
				else if (u == bi) cb = next_byte();
				else for (int _ = 0; _ < vertex_size_list[u]; _++) next_byte();
			}
			Vs[i].x = fs[xi], Vs[i].y = fs[yi], Vs[i].z = fs[zi];
			if (has_color) v_col[i] = (cr << 16) | (cg << 8) | cb;
		}
		delete fs;

		for (int i = 0; i < FN; i++) {
			int n = (int)next_byte();
			if (n != 3) return false;
			for (int u = 0; u < 3; u++) {
				int d = (int)read32();
				if (d >= 0 && d < VN) Fs[i][u] = d;
				else return false;
			}
		}
	}

	else return false;


	return true;
}






void stl2ply(const std::vector<triangle_3d_f> &trigs, float epsilon, std::vector<vec3f> &vertice, std::vector<ply_triangle> &faces) {

#if 0
	{
		int FN = (int)trigs.size();
		vertice.resize(3 * FN);
		faces.resize(FN);
		for (int i = 0; i < FN; i++) {
			for (int u = 0; u < 3; u++) vertice[3 * i + u] = trigs[i][u], faces[i][u] = 3 * i + u;
		}
		return;
	}
#endif

	class disjoint_set {
		uint8_t *rank;
	public:
		int *parent;
		const int inf = 0x7fffffff;
		disjoint_set(int N) {
			parent = new int[N];
			rank = new uint8_t[N];
			for (int i = 0; i < N; i++) {
				parent[i] = -inf;
				rank[i] = 0;
			}
		}
		~disjoint_set() {
			delete parent; parent = 0;
			delete rank; rank = 0;
		}
		int findRepresentative(int i) {
			if (parent[i] < 0) return i;
			else {
				int ans = findRepresentative(parent[i]);
				parent[i] = ans;
				return ans;
			}
		}
		int representative_ID(int i) {
			while (parent[i] >= 0) i = parent[i];
			return -1 - parent[i];
		}
		bool unionSet(int i, int j) {
			int i_rep = findRepresentative(i);
			int j_rep = findRepresentative(j);
			if (i_rep == j_rep) return false;
			if (rank[i_rep] < rank[j_rep])
				parent[i_rep] = parent[i] = j_rep;
			else if (rank[i_rep] > rank[j_rep])
				parent[j_rep] = parent[j] = i_rep;
			else parent[j_rep] = parent[j] = i_rep, rank[i_rep]++;
			return true;
		}
	};

	int FN = (int)trigs.size();
	int VN = 0;

	// restore vertice
	struct vec3_id {
		vec3f p;
		int id;
	};
	vec3_id *vtx = new vec3_id[3 * FN];
	for (int i = 0; i < FN; i++) {
		for (int u = 0; u < 3; u++)
			vtx[3 * i + u] = vec3_id{ trigs[i][u], 3 * i + u };
	}
	vertice.clear();
	faces.resize(FN);

	if (!(epsilon > 0.)) {

		std::sort(vtx, vtx + 3 * FN, [](vec3_id a, vec3_id b) {
			return a.p.x < b.p.x ? true : a.p.x > b.p.x ? false : a.p.y < b.p.y ? true : a.p.y > b.p.y ? false : a.p.z < b.p.z;
		});

		vec3f previous_p = vec3f(NAN);
		for (int i = 0; i < 3 * FN; i++) {
			if (vtx[i].p != previous_p) {
				previous_p = vtx[i].p;
				vertice.push_back(vtx[i].p);
				VN++;
			}
			faces[vtx[i].id / 3].v[vtx[i].id % 3] = VN - 1;
		}
	}

	else {
		disjoint_set dsj(3 * FN);

		// apply a random rotation to avoid worst case runtime
		if (1) {
			const mat3f R(
				0.627040324915f, 0.170877213400f, 0.760014084653f,
				-0.607716612443f, -0.503066180808f, 0.614495676705f,
				0.487340691808f, -0.847186753714f, -0.211597860196f);
			for (int i = 0; i < 3 * FN; i++) vtx[i].p = R * vtx[i].p;
		}

		// three level sorting
		std::sort(vtx, vtx + 3 * FN, [](vec3_id a, vec3_id b) { return a.p.z < b.p.z; });
		for (int i = 0; i < 3 * FN;) {
			int j = i + 1;
			while (j < 3 * FN && vtx[j].p.z - vtx[j - 1].p.z < epsilon) j++;
			std::sort(vtx + i, vtx + j, [](vec3_id a, vec3_id b) { return a.p.y < b.p.y; });
			for (int u = i; u < j;) {
				int v = u + 1;
				while (v < j && vtx[v].p.y - vtx[v - 1].p.y < epsilon) v++;
				std::sort(vtx + u, vtx + v, [](vec3_id a, vec3_id b) { return a.p.x < b.p.x; });
				for (int m = u; m < v;) {
					int n = m + 1;
					while (n < v && vtx[n].p.x - vtx[n - 1].p.x < epsilon) n++;
					//printf("%d\n", n - m);  // mostly 6
					if (1) {  // O(N)
						for (int t = m; t + 1 < n; t++) dsj.unionSet(vtx[t].id, vtx[t + 1].id);
					}
					else {  // O(N²), more accurate, slower
						for (int t1 = m; t1 < n; t1++) for (int t2 = m; t2 < t1; t2++) {
							if ((vtx[t2].p - vtx[t1].p).sqr() < epsilon*epsilon) dsj.unionSet(vtx[t1].id, vtx[t2].id);
						}
					}
					m = n;
				}
				u = v;
			}
			i = j;
		}

		// pull points out from the disjoint set
		int unique_count = 0;
		int *vertice_map = new int[3 * FN];
		for (int i = 0; i < 3 * FN; i++)
			if (dsj.findRepresentative(i) == i) {
				vertice_map[i] = unique_count++;
				vertice.push_back(trigs[i / 3][i % 3]);
			}
		for (int i = 0; i < 3 * FN; i++)
			vertice_map[i] = vertice_map[dsj.findRepresentative(i)];
		for (int i = 0; i < FN; i++) for (int u = 0; u < 3; u++) {
			faces[i].v[u] = vertice_map[3 * i + u];
		}
		delete vertice_map;
	}

	delete vtx;

}



bool readBinarySTL(FILE* fp, vec3f* &Vs, ply_triangle* &Fs, int &VN, int &FN, COLORREF* &f_col) {

	char s[80]; if (fread(s, 1, 80, fp) != 80) return false;
	int N; if (fread(&N, sizeof(int), 1, fp) != 1) return false;
	if (N <= 0) return false;
	std::vector<triangle_3d_f> trigs;
	try {
		trigs.reserve(N);
		f_col = new COLORREF[N];
	} catch (std::bad_alloc) {
		return false;
	}

	// https://en.wikipedia.org/wiki/STL_(file_format)#Color_in_binary_STL
	auto stlColor = [](int16_t c)->COLORREF {
		if (c >= 0) return 0xffffffff;
		COLORREF r = 0; uint8_t *k = (uint8_t*)&r;
		k[2] = (((uint32_t)c & (uint32_t)0b111110000000000) >> 10) << 3;
		k[1] = (((uint32_t)c & (uint32_t)0b000001111100000) >> 5) << 3;
		k[0] = (((uint32_t)c & (uint32_t)0b000000000011111)) << 3;
		return r;
	};

	for (int i = 0; i < N; i++) {
		vec3f f[4];
		if (fread(f, sizeof(vec3f), 4, fp) != 4) return false;
		uint16_t col; if (fread(&col, 2, 1, fp) != 1) return false;
		trigs.push_back(triangle_3d_f(f[1], f[2], f[3]));
		f_col[i] = stlColor(col);
	}

	std::vector<vec3f> vertice;
	std::vector<ply_triangle> faces;
	stl2ply(trigs, 0.0f, vertice, faces);
	VN = (int)vertice.size();
	FN = (int)faces.size();
	Vs = new vec3f[VN];
	for (int i = 0; i < VN; i++) Vs[i] = vertice[i];
	vertice.clear(); vertice.shrink_to_fit();
	Fs = new ply_triangle[FN];
	for (int i = 0; i < FN; i++) Fs[i] = faces[i];
	faces.clear(); faces.shrink_to_fit();

	return true;
}


bool readAsciiSTL(FILE* fp, vec3f* &Vs, ply_triangle* &Fs, int &VN, int &FN) {

	const int BUFFER_SIZE = 0xffff;
	char buffer[BUFFER_SIZE + 1];
	const int STRING_SIZE = 0xff;
	char str[STRING_SIZE + 1];

	auto equal_string = [](const char* a, const char* b)->bool {
		while (*a && *b) {
			if (*a != *b) return false;
			a++, b++;
		}
		return *a == *b;
	};

	if (!fgets(buffer, BUFFER_SIZE, fp)) return false;
	sscanf(buffer, "%255s", str);
	if (!equal_string(str, "solid")) return false;


	std::vector<triangle_3d_f> trigs;

	while (1) {
		if (!fgets(buffer, BUFFER_SIZE, fp)) return false;
		sscanf(buffer, "%255s", str);
		if (equal_string(str, "endsolid")) break;
		else if (equal_string(str, "facet")) {
			if (!fgets(buffer, BUFFER_SIZE, fp)) return false;  // outer loop, already scanned
			sscanf(buffer, "%255s", str);
			if (!equal_string(str, "outer")) return false;
			triangle_3d_f trig;
			for (int i = 0; i < 3; i++) {
				if (!fgets(buffer, BUFFER_SIZE, fp)) return false;  // vertex
				sscanf(buffer, "%255s%f%f%f", str, &trig[i].x, &trig[i].y, &trig[i].z);
				if (!equal_string(str, "vertex")) return false;
			}
			trigs.push_back(trig);
			if (!fgets(buffer, BUFFER_SIZE, fp)) return false;  // endloop
			sscanf(buffer, "%255s", str);
			if (!equal_string(str, "endloop")) return false;
			if (!fgets(buffer, BUFFER_SIZE, fp)) return false;  // endfacet
			sscanf(buffer, "%255s", str);
			if (!equal_string(str, "endfacet")) return false;
		}
		else return false;
	}

	std::vector<vec3f> vertice;
	std::vector<ply_triangle> faces;
	stl2ply(trigs, 0.0f, vertice, faces);
	VN = (int)vertice.size();
	FN = (int)faces.size();
	Vs = new vec3f[VN];
	for (int i = 0; i < VN; i++) Vs[i] = vertice[i];
	vertice.clear(); vertice.shrink_to_fit();
	Fs = new ply_triangle[FN];
	for (int i = 0; i < FN; i++) Fs[i] = faces[i];
	faces.clear(); faces.shrink_to_fit();

	return true;
}


bool read3DFile(FILE* fp, vec3f* &Vs, ply_triangle* &Fs, int &VN, int &FN, COLORREF* &v_col, COLORREF* &f_col) {
	if (fp == NULL) return false;

	try {
		if (readPLY(fp, Vs, Fs, VN, FN, v_col)) return true;
		else throw(false);
	} catch (...) {
		rewind(fp);
	}

	try {
		if (readAsciiSTL(fp, Vs, Fs, VN, FN)) return true;
		else throw(false);
	} catch (...) {
		rewind(fp);
	}

	try {
		if (readBinarySTL(fp, Vs, Fs, VN, FN, f_col)) return true;
		else throw(false);
	} catch (...) {
		rewind(fp);
	}

	return false;
}
