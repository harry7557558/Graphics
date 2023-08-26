#include "elements.h"

#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>


std::string getNameFromFilename(std::string filename) {
    std::string name = "/" + std::string(filename);
    std::replace(name.begin(), name.end(), '\\', '/');
    name = name.substr(name.rfind('/')+1, name.length()-name.rfind('/')-1);
    name = name.substr(0, name.rfind('.'));
    return name;
}


void extendVector(std::vector<uint8_t> &v, size_t n, const void *p) {
    for (int i = 0; i < n; i++)
        v.push_back(((uint8_t*)p)[i]);
}
void extendVectorS(std::vector<uint8_t> &v, const char* s) {
    extendVector(v, strlen(s), s);
}


std::vector<uint8_t> writeSTL_(
    std::vector<vec3> verts, std::vector<ivec3> trigs
) {
    std::vector<uint8_t> res;
    for (int i = 0; i < 80; i++)
        res.push_back(0);
    int n = (int)trigs.size();
    extendVector(res, 4, &n);
    assert(sizeof(vec3) == 12);
    for (int i = 0; i < n; i++) {
        auto writevec3 = [&](vec3 v) {
            v = vec3(v.x, -v.z, v.y);
            extendVector(res, 12, &v);
        };
        vec3 v0 = verts[trigs[i][0]];
        vec3 v1 = verts[trigs[i][1]];
        vec3 v2 = verts[trigs[i][2]];
        vec3 n = normalize(cross(v1-v0, v2-v0));
        writevec3(n);
        writevec3(v0);
        writevec3(v1);
        writevec3(v2);
        res.push_back(0); res.push_back(0);
    }
    return res;
}

void writeSTL(
    const char* filename,
    std::vector<vec3> verts, std::vector<ivec3> trigs
) {
    FILE* fp = fopen(&filename[0], "wb");
    if (!fp) {
        printf("Error open file %s\n", filename);
    }
    std::vector<uint8_t> data = writeSTL_(verts, trigs);
    fwrite(&data[0], 1, data.size(), fp);
    fclose(fp);
}


std::vector<uint8_t> writePLY_(
    std::vector<vec3> verts, std::vector<ivec3> trigs,
    std::vector<vec3> normals = std::vector<vec3>()
) {
    std::vector<uint8_t> res;

    std::stringstream header;
    header << "ply\nformat binary_little_endian 1.0\n";
    header << "element vertex " << (int)verts.size() << "\n";
    header << "property float x\n";
    header << "property float y\n";
    header << "property float z\n";
    if (!normals.empty()) {
        header << "property float nx\n";
        header << "property float ny\n";
        header << "property float nz\n";
    }
    header << "element face " << (int)trigs.size() << "\n";
    header << "property list uchar int vertex_index\n";
    header << "end_header\n";
    extendVectorS(res, &header.str()[0]);

    assert(sizeof(vec3) == 12);
    for (int i = 0; i < (int)verts.size(); i++) {
        vec3 v = verts[i];
        extendVector(res, 12, &v);
        if (!normals.empty()) {
            vec3 n = normals[i];
            extendVector(res, 12, &n);
        }
    }
    assert(sizeof(ivec3) == 12);
    for (ivec3 t : trigs) {
        res.push_back(3);
        extendVector(res, 12, &t);
    }
    return res;
}

void writePLY(
    const char* filename,
    std::vector<vec3> verts, std::vector<ivec3> trigs,
    std::vector<vec3> normals = std::vector<vec3>()
) {
    FILE* fp = fopen(&filename[0], "wb");
    if (!fp) {
        printf("Error open file %s\n", filename);
    }
    std::vector<uint8_t> data = writePLY_(verts, trigs, normals);
    fwrite(&data[0], 1, data.size(), fp);
    fclose(fp);
}


std::vector<uint8_t> writeOBJ_(
    const char* name,
    std::vector<vec3> verts, std::vector<ivec3> trigs,
    std::vector<vec3> normals = std::vector<vec3>()
) {
    std::vector<uint8_t> res;
    char buf[1024];
    sprintf(buf, "o %s\n", name);
    extendVectorS(res, buf);
    for (vec3 v : verts) {
        sprintf(buf, "v %.6g %.6g %.6g\n", v.x, v.y, v.z);
        extendVectorS(res, buf);
    }
    for (vec3 n : normals) {
        sprintf(buf, "vn %.6g %.6g %.6g\n", n.x, n.y, n.z);
        extendVectorS(res, buf);
    }
    for (ivec3 t : trigs) {
        t.x += 1, t.y += 1, t.z += 1;
        if (!normals.empty()) {
            sprintf(buf, "f %d//%d %d//%d %d//%d\n", t.x, t.x, t.y, t.y, t.z, t.z);
            extendVectorS(res, buf);
        }
        else {
            sprintf(buf, "f %d %d %d\n", t.x, t.y, t.z);
            extendVectorS(res, buf);
        }
    }
    return res;
}

void writeOBJ(const char* filename,
    std::vector<vec3> verts, std::vector<ivec3> trigs,
    std::vector<vec3> normals = std::vector<vec3>(),
    std::vector<vec2> texcoords = std::vector<vec2>(),
    std::vector<uint8_t> texImage = std::vector<uint8_t>()
) {
    if (!normals.empty())
        assert(normals.size() == verts.size());
    if (!texcoords.empty())
        assert(texcoords.size() == verts.size());
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error open file %s\n", filename);
    }
    std::string name = getNameFromFilename(filename);
    std::string filename_mtl = std::string(filename) + ".mtl";
    std::string filename_tex = std::string(filename) + "_texture.png";

    if (!texImage.empty())
        fprintf(fp, "mtllib %s\n", &filename_mtl[0]);
    fprintf(fp, "o %s\n", &name[0]);

    for (vec3 v : verts)
        fprintf(fp, "v %.6g %.6g %.6g\n", v.x, v.y, v.z);
    for (vec2 t : texcoords)
        fprintf(fp, "vt %.6g %.6g\n", t.x, 1.0f-t.y);
    for (vec3 n : normals)
        fprintf(fp, "vn %.6g %.6g %.6g\n", n.x, n.y, n.z);

    if (!texImage.empty()) {
        fprintf(fp, "usemtl %s\n", &name[0]);
        fprintf(fp, "s 1\n");

        FILE* fpmtl = fopen(&filename_mtl[0], "wb");
        if (!fpmtl)
            printf("Error open file %s\n", &filename_mtl[0]);
        fprintf(fpmtl, "newmtl %s\n", &name[0]);
        fprintf(fpmtl, "map_Ka %s\n", &filename_tex[0]);
        fprintf(fpmtl, "map_Kd %s\n", &filename_tex[0]);
        fclose(fpmtl);
    }

    for (ivec3 t : trigs) {
        t.x += 1, t.y += 1, t.z += 1;
        if (!normals.empty() && !texcoords.empty())
            fprintf(fp, "f %d/%d/%d %d/%d/%d %d/%d/%d\n",
                t.x, t.x, t.x, t.y, t.y, t.y, t.z, t.z, t.z);
        else if (!normals.empty() && texcoords.empty())
            fprintf(fp, "f %d//%d %d//%d %d//%d\n",
                t.x, t.x, t.y, t.y, t.z, t.z);
        else if (normals.empty() && !texcoords.empty())
            fprintf(fp, "f %d/%d %d/%d %d/%d\n",
                t.x, t.x, t.y, t.y, t.z, t.z);
        else
            fprintf(fp, "f %d %d %d\n", t.x, t.y, t.z);
    }

    fclose(fp);

    if (!texImage.empty()) {
        FILE* fptex = fopen(&filename_tex[0], "wb");
        if (!fptex)
            printf("Error open file %s\n", &filename_tex[0]);
        fwrite(texImage.data(), 1, texImage.size(), fptex);
        fclose(fptex);
    }
}


std::vector<uint8_t> writeGLB_(
    const char* name,
    std::vector<vec3> verts, std::vector<ivec3> trigs,
    std::vector<vec3> normals = std::vector<vec3>(),
    std::vector<vec2> texcoords = std::vector<vec2>(),
    std::vector<uint8_t> texImage = std::vector<uint8_t>()
) {
    std::vector<uint8_t> res;
    if (!normals.empty())
        assert(normals.size() == verts.size());
    if (!texcoords.empty())
        assert(texcoords.size() == verts.size());
    assert(sizeof(vec3) == 12);
    assert(sizeof(ivec3) == 12);
    int vn = (int)verts.size();
    int tn = 3*(int)trigs.size();

    int hasNormal = (int)(!normals.empty());
    int hasTexcoord = (int)(!texcoords.empty());
    int hasImage = (int)(!texImage.empty());
    int positionIndex = 0;
    int normalIndex = positionIndex + hasNormal;
    int texcoordIndex = normalIndex + hasTexcoord;
    int indiceIndex = texcoordIndex + 1;
    int imageIndex = indiceIndex + hasImage;

    std::stringstream json;
    json << "{";
    // asset
    json << "\"asset\":{\"version\":\"2.0\"},";
    // scene
    json << "\"scene\":0,";
    json << "\"scenes\":[{\"name\":\"Scene\",\"nodes\":[0]}],";
    // node
    json << "\"nodes\":[{\"mesh\":0,\"name\":\"" << name << "\"}],";
    // material
    if (hasImage) json << "\"materials\":[{\"name\":\"" << name << "\",\"doubleSided\":true," << 
        "\"pbrMetallicRoughness\":{\"baseColorTexture\":{\"index\":0},\"metallicFactor\":0}}],";
    // mesh
    json << "\"meshes\":[{\"name\":\"" << name << "\",\"primitives\":[";
    json << "{\"attributes\":{\"POSITION\":" << positionIndex;
    if (hasNormal) json << ",\"NORMAL\":" << normalIndex;
    if (hasTexcoord) json << ",\"TEXCOORD_0\":" << texcoordIndex;
    json << "},\"indices\":" << indiceIndex;
    if (hasImage) json << ",\"material\":0";
    json << "}]}],";
    // texture
    if (hasImage)
        json << "\"textures\":[{\"sampler\":0,\"source\":0}],";
    // image
    if (hasImage)
        json << "\"images\":[{\"bufferView\":" << imageIndex <<
            ",\"mimeType\":\"image/png\",\"name\":\"" << name << "\"}],";
    // accessors
    json << "\"accessors\":[";
    json << "{\"bufferView\":" << positionIndex <<
        ",\"componentType\":5126,\"count\":" << vn << ",\"type\":\"VEC3\"},";
    if (hasNormal) json << "{\"bufferView\":" << normalIndex <<
        ",\"componentType\":5126,\"count\":" << vn << ",\"type\":\"VEC3\"},";
    if (hasTexcoord) json << "{\"bufferView\":" << texcoordIndex <<
        ",\"componentType\":5126,\"count\":" << vn << ",\"type\":\"VEC2\"},";
    json << "{\"bufferView\":" << indiceIndex <<
        ",\"componentType\":5125,\"count\":" << tn << ",\"type\":\"SCALAR\"}";
    json << "],";
    // buffer views
    json << "\"bufferViews\":[";
    int byteLength = 12*vn, byteOffset = 0;
    json << "{\"buffer\":0,\"byteLength\":" << byteLength <<
        ",\"byteOffset\":" << byteOffset << ",\"target\":34962},";
    if (hasNormal) {
        byteOffset += byteLength, byteLength = 12*vn;
        json << "{\"buffer\":0,\"byteLength\":" << byteLength <<
            ",\"byteOffset\":" << byteOffset << ",\"target\":34962},";
    }
    if (hasTexcoord) {
        byteOffset += byteLength, byteLength = 8*vn;
        json << "{\"buffer\":0,\"byteLength\":" << byteLength <<
            ",\"byteOffset\":" << byteOffset << ",\"target\":34962},";
    }
    byteOffset += byteLength, byteLength = 4*tn;
    json << "{\"buffer\":0,\"byteLength\":" << byteLength <<
        ",\"byteOffset\":" << byteOffset << ",\"target\":34963}";
    if (hasImage) {
        byteOffset += byteLength, byteLength = (int)texImage.size();
        json << ",{\"buffer\":0,\"byteLength\":" << byteLength <<
            ",\"byteOffset\":" << byteOffset << "}";
    }
    byteOffset += byteLength;
    json << "],";
    // sampler
    if (hasImage)
        json << "\"samplers\":[{\"magFilter\":9729,\"minFilter\":9987}],";
    // buffers
    json << "\"buffers\":[{\"byteLength\":" << byteOffset << "}]";
    json << "}";
    std::string jsons = json.str();
    while (jsons.size() % 4 != 0)
        jsons += " ";

    // header
    int temp;
    extendVectorS(res, "glTF");
    temp = 2;
    extendVector(res, 4, &temp);
    temp = 12 + (4+4+(int)jsons.size()) + (4+4+byteOffset);
    extendVector(res, 4, &temp);

    // JSON chunk
    temp = (int)jsons.size();
    extendVector(res, 4, &temp);
    temp = 0x4e4f534a;
    extendVector(res, 4, &temp);
    extendVectorS(res, &jsons[0]);

    // buffer
    temp = byteOffset;
    extendVector(res, 4, &temp);
    temp = 0x004e4942;
    extendVector(res, 4, &temp);
    extendVector(res, 12*vn, &verts[0]);
    if (hasNormal)
        extendVector(res, 12*vn, &normals[0]);
    if (hasTexcoord)
        extendVector(res, 8*vn, &texcoords[0]);
    extendVector(res, 12*trigs.size(), &trigs[0]);
    extendVector(res, texImage.size(), &texImage[0]);

    return res;
}

void writeGLB(const char* filename,
    std::vector<vec3> verts, std::vector<ivec3> trigs,
    std::vector<vec3> normals = std::vector<vec3>(),
    std::vector<vec2> texcoords = std::vector<vec2>(),
    std::vector<uint8_t> texImage = std::vector<uint8_t>()
) {
    FILE* fp = fopen(&filename[0], "wb");
    if (!fp) {
        printf("Error open file %s\n", filename);
    }
    std::vector<uint8_t> data = writeGLB_(
        getNameFromFilename(filename).data(),
        verts, trigs, normals, texcoords, texImage);
    fwrite(&data[0], 1, data.size(), fp);
    fclose(fp);
}
