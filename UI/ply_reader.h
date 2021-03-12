// a very limited PLY file reader

#include <vector>
#include <string>

struct ply_triangle {
	int v[3];
	int& operator[] (int d) {
		return v[d];
	}
};



bool readPLY(FILE* fp, vec3f* &Vs, ply_triangle* &Fs, int &VN, int &FN) {
	if (fp == 0) return false;

	const int MAX_SIZE = 4;
	int buffer_size = MAX_SIZE;
	int index = buffer_size;
	char buf[MAX_SIZE + 1]; buf[MAX_SIZE] = 0;
	auto next_byte = [&]()->int {
		if (index >= buffer_size) {
			if (buffer_size == MAX_SIZE)
				buffer_size = (int)fread(buf, 1, MAX_SIZE, fp), index = 0;
			else return EOF;
		}
		return buf[index++];
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
			if (c == end_char) { index--; return s; }
			if (c == 0) throw(c);
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

	int xi, yi, zi, property_index = 0;


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
			std::string nm;
			if (!split_string(nm, s)) return false;
			int d = std::stoi(s);
			if (nm == "vertex") VN = d;
			else if (nm == "face") {
				FN = d;
				if (VN == -1) return false;
			}
		}

		else if (nm == "property") {
			std::string type;
			if (!split_string(type, s)) return false;
			if (type == "list") {
				if (s != "uchar int vertex_indices") return false;
				if (FN == -1) return false;
			}
			else {
				if (format != ASCII && type != "float") return false;
				if (VN != -1 && FN == -1) {
					if (type == "float") {
						if (s == "x") xi = property_index;
						if (s == "y") yi = property_index;
						if (s == "z") zi = property_index;
					}
					else if (s == "x" || s == "y" || s == "z") return false;
					property_index += 1;
				}
			}
		}

		else if (nm != "comment") return false;

	}

	if (format == -1 || VN == -1 || FN == -1 || xi == -1 || yi == -1 || zi == -1) return false;


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
		for (int i = 0; i < VN; i++) {
			for (int u = 0; u < property_index; u++) *(uint32_t*)&fs[u] = read32();
			Vs[i].x = fs[xi], Vs[i].y = fs[yi], Vs[i].z = fs[zi];
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
		for (int i = 0; i < VN; i++) {
			for (int u = 0; u < property_index; u++) *(uint32_t*)&fs[u] = read32();
			Vs[i].x = fs[xi], Vs[i].y = fs[yi], Vs[i].z = fs[zi];
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
