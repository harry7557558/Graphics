"use strict";


// rendering related
var renderer = {
    canvas: null,
    gl: null,
    width: -1,
    height: -1,
    iFrame: 0,
    vsSource: "",
    fsConvTranspose2d421: "",
    programConvTranspose2d421: null,
    fsLeakyReLU: "",
    programLeakyReLU: null,
    fsOutput: "",
    programOutput: null,
};

// request shader sources
function loadShaderSource(path) {
    var request = new XMLHttpRequest();
    request.open("GET", path, false);
    request.send(null);
    if (request.status != 200) return "";
    var source = request.responseText;
    return source;
}

// compile shaders and create a shader program
function createShaderProgram(vsSource, fsSource) {
    let gl = renderer.gl;
    function loadShader(gl, type, source) {
        var shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
            throw new Error("Shader compile error: " + gl.getShaderInfoLog(shader));
        return shader;
    }
    var vShader = loadShader(gl, gl.VERTEX_SHADER, vsSource);
    var fShader = loadShader(gl, gl.FRAGMENT_SHADER, fsSource);
    var shaderProgram = gl.createProgram();
    gl.attachShader(shaderProgram, vShader);
    gl.attachShader(shaderProgram, fShader);
    gl.linkProgram(shaderProgram);
    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS))
        throw new Error(gl.getProgramInfoLog(shaderProgram));
    return shaderProgram;
}

// create texture/framebuffer
function createSampleTexture(width, height) {
    let gl = renderer.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    const level = 0;
    const internalFormat = gl.RGBA32F;
    const border = 0;
    const format = gl.RGBA;
    const type = gl.FLOAT;
    const data = null;
    gl.texImage2D(gl.TEXTURE_2D, level, internalFormat,
        width, height, border,
        format, type, data);
    // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.MIRRORED_REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.MIRRORED_REPEAT);
    return tex;
}
function createRenderTarget(width, height) {
    let gl = renderer.gl;
    const tex = createSampleTexture(width, height);
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    // const sampler = createSampleTexture(gl, width, height);
    return {
        texture: tex,
        framebuffer: framebuffer,
        // sampler: sampler
    };
}
function destroyRenderTarget(target) {
    let gl = renderer.gl;
    gl.deleteTexture(target.texture);
    gl.deleteFramebuffer(target.framebuffer);
}

function loadWeightTexture(name, texture_name) {
    let modelSelector = document.getElementById("model-select");
    let root = modelSelector.value;
    const gl = renderer.gl;

    function onload(weights) {
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);

        var l = weights.length / 4, rl = Math.sqrt(l);
        var w = 1, h = l;
        for (var i = 1; i <= rl; i++) {
            var j = Math.round(l / i);
            if (i * j == l) w = i, h = j;
        }
        console.log(texture_name, w, h);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F,
            w, h, 0,
            gl.RGBA, gl.FLOAT,
            weights);

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_BASE_LEVEL, 0);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAX_LEVEL, 0);

        renderer[texture_name] = tex;
        updateModel();
    }

    var req = new XMLHttpRequest();
    req.open("GET",  root+'/'+name+'.bin', true);
    req.responseType = "arraybuffer";
    req.onerror = function (e) {
        alert("Failed to load texture " + texture_name);
    };
    req.onload = function (e) {
        if (req.status == 200) {
            var weights = new Float32Array(req.response);
            renderer[name] = weights;
            onload(weights);
        }
        else {
            req.onerror();
        }
    };
    req.send();
}


// set position buffer for vertex shader
function setPositionBuffer(program) {
    let gl = renderer.gl;
    var vpLocation = gl.getAttribLocation(program, "vertexPosition");
    const numComponents = 2;
    const type = gl.FLOAT;
    const normalize = false;
    const stride = 0, offset = 0;
    gl.bindBuffer(gl.ARRAY_BUFFER, renderer.positionBuffer);
    gl.vertexAttribPointer(
        vpLocation,
        numComponents, type, normalize, stride, offset);
    gl.enableVertexAttribArray(vpLocation);
}


function CNNBuffer(n, w, h) {
    this.n = n;
    this.w = w, this.h = h;
    this.imgs = [];
    for (var i = 0; i < n; i += 4)
        this.imgs.push(createRenderTarget(w, h));
    this.setData = function(data) {
        var layers = [];
        for (var i = 0; i < this.n; i++)
            layers[i] = data.slice(i*w*h, (i+1)*w*h);
        let gl = renderer.gl;
        for (var i = 0; i < this.n; i += 4) {
            var data = new Float32Array(4*w*h);
            for (var j = 0; j < 4 && i+j < this.n; j++)
                for (var k = 0; k < w*h; k++)
                    data[4*k+j] = layers[i+j][k];
            gl.bindTexture(gl.TEXTURE_2D, this.imgs[Math.floor(i/4)].texture);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F,
                w, h, 0,
                gl.RGBA, gl.FLOAT,
                data);
        }
    };
}

function ConvTranspose2D421(
    grid_size, tile_size, n_in, n_out, weights, bn_biases
) {
    if (weights.length != n_in*n_out*16)
        throw new Error("Incorrect weight size");
    if (bn_biases.length != n_out)
        throw new Error("Incorrect bias size");
    if (typeof grid_size == 'number') {
        this.grid_size_x = grid_size;
        this.grid_size_y = grid_size;
    }
    else {
        this.grid_size_x = grid_size[0];
        this.grid_size_y = grid_size[1];
    }
    this.tile_size_in = tile_size;
    this.tile_size_out = tile_size * 2;
    this.n_in = n_in;
    this.n_out = n_out;
    this.weights = weights;
    this.bn_biases = Array.from(bn_biases);
    while (this.bn_biases.length % 4 != 0.0)
        this.bn_biases.push(0.0);

    this.mats = [];
    for (var i = 0; i < this.n_out; i += 4) {
        var mats = [];
        for (var j = 0; j < this.n_in; j += 4) {
            var matsj = [];
            for (var wi = 0; wi < 4; wi++) {
                for (var wj = 0; wj < 4; wj++) {
                    var mat = new Float32Array(16);
                    for (var a = 0; a < 4; a++) {
                        for (var b = 0; b < 4; b++) {
                            var mi = (j+a)*n_out+(i+b);
                            mat[4*a+b] = this.weights[16*mi+((3-wi)*4+(3-wj))];
                        }
                    }
                    matsj.push(mat);
                }
            }
            mats.push(matsj);
        }
        this.mats.push(mats);
    }

    this.forward = function(buffer_in, buffer_out) {
        if (buffer_in.n != this.n_in)
            throw new Error("Incorrect input buffer length ("+buffer_in.n+","+this.n_in+")");
        if (buffer_out.n != this.n_out)
            throw new Error("Incorrect output buffer length ("+buffer_out.n+","+this.n_out+")");
        let gl = renderer.gl;
        let program = renderer.programConvTranspose2d421;
        gl.useProgram(program);
        gl.viewport(0, 0, this.grid_size_x*this.tile_size_out, this.grid_size_y*this.tile_size_out);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE);
        for (var i = 0; i < this.n_out; i += 4) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, buffer_out.imgs[Math.floor(i/4)].framebuffer);
            gl.clearColor(this.bn_biases[i], this.bn_biases[i+1], this.bn_biases[i+2], this.bn_biases[i+3]);
            gl.clear(gl.COLOR_BUFFER_BIT);
            for (var j = 0; j < this.n_in; j += 4) {
                for (var li = 0; li < 16; li++) {
                    let uniformLocation = gl.getUniformLocation(program, 'w['+li+']');
                    var mat = this.mats[i/4][j/4][li];
                    gl.uniformMatrix4fv(uniformLocation, false, mat);
                }
                gl.uniform1i(gl.getUniformLocation(program, 'tileSize'), this.tile_size_in);
                setPositionBuffer(program);
                gl.activeTexture(gl.TEXTURE0);
                gl.bindTexture(gl.TEXTURE_2D, buffer_in.imgs[Math.floor(j/4)].texture);
                gl.uniform1i(gl.getUniformLocation(program, "uSrc"), 0);
                gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
            }
        }
        gl.disable(gl.BLEND);
    }
}

function LeakyReLU(grid_size, tile_size, n_in, negative_slope) {
    if (typeof grid_size == 'number') {
        this.grid_size_x = grid_size;
        this.grid_size_y = grid_size;
    }
    else {
        this.grid_size_x = grid_size[0];
        this.grid_size_y = grid_size[1];
    }
    this.tile_size = tile_size;
    this.n_in = n_in;
    this.n_out = n_in;
    this.negative_slope = negative_slope;

    this.forward = function(buffer_in, buffer_out) {
        let gl = renderer.gl;
        let program = renderer.programLeakyReLU;
        gl.useProgram(program);
        gl.viewport(0, 0, this.grid_size_x*this.tile_size, this.grid_size_y*this.tile_size);
        for (var i = 0; i < this.n_out; i += 4) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, buffer_out.imgs[Math.floor(i/4)].framebuffer);
            gl.uniform1f(gl.getUniformLocation(program, 'negative_slope'), this.negative_slope);
            setPositionBuffer(program);
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, buffer_in.imgs[Math.floor(i/4)].texture);
            gl.uniform1i(gl.getUniformLocation(program, "uSrc"), 0);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        }
    }
}


// call this function to re-render
async function drawScene() {
    let gl = renderer.gl;

    let r = renderer;

    var time0 = performance.now();

    let nx = Math.floor(canvas.width / 64);
    let ny = Math.floor(canvas.height / 64);
    var initial = new Float32Array(32*4*ny*4*nx);
    for (var gi = 0; gi < nx; gi++) {
        for (var gj = 0; gj < ny; gj++) {
            // random numbers
            function hash(i) {
                var s = 12345.67*Math.sin(12.34*gi+56.78*gj+9.012*i+3.45);
                return s - Math.floor(s);
            }
            var t = 1e-3*performance.now()*0.4;
            var latent = new Array(32).fill(0.0);
            for (var i = 0; i < 32; i++) {
                var t1 = t + hash(i);
                var i0 = Math.floor(t1), i1 = i0 + 1.0, tf = t1 - i0;
                var u1 = hash(i0*32+i), u2 = hash(i1*32+i);
                var u = u1 + (u2-u1) * tf;
                var v1 = hash(i0*32+i+0.5), v2 = hash(i1*32+i+0.5);
                var v = v1 + (v2-v1) * tf;
                u = 0.5 + 0.999999*(u-0.5);
                u = 0.5-u*u*(Math.log(u)-0.5)+(u-1.)*(u-1.)*(Math.log(1.-u)-0.5);
                v = 0.5 + 0.999999*(v-0.5);
                v = 0.5-v*v*(Math.log(v)-0.5)+(v-1.)*(v-1.)*(Math.log(1.-v)-0.5);
                latent[i] = Math.sqrt(-2.*Math.log(u)) * Math.sin(2.0*Math.PI*v);
            }
            
            // linear - (32) => (512)
            var layer1js = new Array(512).fill(0.0);
            let w01 = renderer.w01_512_32;
            for (var i = 0; i < 512; i++)
                for (var j = 0; j < 32; j++)
                    layer1js[i] += latent[j] * w01[i*32+j];
            // leaky relu
            for (var i = 0; i < 512; i++)
                layer1js[i] = Math.max(layer1js[i], 0.1*layer1js[i]);
            // add to data
            for (var i = 0; i < 32; i++)
                for (var y = 0; y < 4; y++)
                    for (var x = 0; x < 4; x++)
                        initial[i*4*ny*4*nx+(gj*4+y)*(4*nx)+(gi*4+x)] = layer1js[i*4*4+y*4+x];
        }
    }
    r.layer1.setData(initial);

    r.conv2.forward(r.layer1, r.layer2);
    r.relu2.forward(r.layer2, r.layer2a);
    r.conv3.forward(r.layer2a, r.layer3);
    r.relu3.forward(r.layer3, r.layer3a);
    r.conv4.forward(r.layer3a, r.layer4);
    r.relu4.forward(r.layer4, r.layer4a);
    r.conv5.forward(r.layer4a, r.layer5);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(renderer.programOutput);
    gl.viewport(0, 0, 64*nx, 64*ny);
    setPositionBuffer(renderer.programOutput);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, r.layer5.imgs[0].texture);
    gl.uniform1i(gl.getUniformLocation(renderer.programOutput, "uSrc"), 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    var time1 = performance.now();
    // console.log("JS time", (time1-time0).toFixed(2), "ms");
}


// load model
function loadModel() {
    loadWeightTexture("w01_512_32", "w01");
    loadWeightTexture("w02_32_64_4_4", "w02");
    loadWeightTexture("b02_64", "b02");
    loadWeightTexture("w03_64_32_4_4", "w03");
    loadWeightTexture("b03_32", "b03");
    loadWeightTexture("w04_32_16_4_4", "w04");
    loadWeightTexture("b04_16", "b04");
    loadWeightTexture("w05_16_3_4_4", "w05");
}

function updateModel() {
    let r = renderer;
    let nx = Math.floor(canvas.width / 64);
    let ny = Math.floor(canvas.height / 64);
    let n = [nx, ny];
    r.layer1 = new CNNBuffer(32, 4*nx, 4*ny);
    r.conv2 = new ConvTranspose2D421(n, 4, 32, 64, r.w02_32_64_4_4, r.b02_64);
    r.layer2 = new CNNBuffer(64, 8*nx, 8*ny);
    r.relu2 = new LeakyReLU(n, 8, 64, 0.1);
    r.layer2a = new CNNBuffer(64, 8*nx, 8*ny);
    r.conv3 = new ConvTranspose2D421(n, 8, 64, 32, r.w03_64_32_4_4, r.b03_32);
    r.layer3 = new CNNBuffer(32, 16*nx, 16*ny);
    r.relu3 = new LeakyReLU(n, 16, 32, 0.1);
    r.layer3a = new CNNBuffer(32, 16*nx, 16*ny);
    r.conv4 = new ConvTranspose2D421(n, 16, 32, 16, r.w04_32_16_4_4, r.b04_16);
    r.layer4 = new CNNBuffer(16, 32*nx, 32*ny);
    r.relu4 = new LeakyReLU(n, 32, 16, 0.1);
    r.layer4a = new CNNBuffer(16, 32*nx, 32*ny);
    r.conv5 = new ConvTranspose2D421(n, 32, 16, 3, r.w05_16_3_4_4, [0, 0, 0]);
    r.layer5 = new CNNBuffer(3, 64*nx, 64*ny);
}

// load renderer/interaction
window.onload = function () {
    // get context
    function onError(error) {
        console.error(error);
        let errorMessageContainer = document.getElementById("error-message");
        errorMessageContainer.style.display = "block";
        errorMessageContainer.innerHTML = error;
    }
    let canvas = document.getElementById("canvas");
    renderer.canvas = canvas;
    renderer.gl = canvas.getContext("webgl2") || canvas.getContext("experimental-webgl2");
    if (renderer.gl == null)
        return onError("Error: Your browser may not support WebGL.");
    if (renderer.gl.getExtension("EXT_color_buffer_float") == null)
        return onError("Error: Your device does not support the `EXT_color_buffer_float` extension.");
    if (renderer.gl.getExtension("EXT_float_blend") == null)
        return onError("Error: Your device does not support the `EXT_float_blend` extension.");
    canvas.addEventListener("webglcontextlost", function (event) {
        event.preventDefault();
        onError("Error: WebGL context lost.");
    }, false);
    renderer.width = canvas.width;
    renderer.height = canvas.height;

    // weights/textures
    loadModel();
    document.getElementById("model-select").addEventListener("input", loadModel);

    // GLSL source
    console.time("load glsl code");
    renderer.vsSource = "#version 300 es\nin vec4 vertexPosition;" +
        "void main(){gl_Position=vertexPosition;}";
    renderer.fsConvTranspose2d421 = loadShaderSource("convtranspose2d421.glsl");
    renderer.fsLeakyReLU = loadShaderSource("leakyrelu.glsl");
    renderer.fsOutput = loadShaderSource("output.glsl");
    console.timeEnd("load glsl code");

    // shaders
    console.time("compile shader");
    try {
        renderer.programConvTranspose2d421 = createShaderProgram(renderer.vsSource, renderer.fsConvTranspose2d421);
        renderer.programLeakyReLU = createShaderProgram(renderer.vsSource, renderer.fsLeakyReLU);
        renderer.programOutput = createShaderProgram(renderer.vsSource, renderer.fsOutput);
    }
    catch (e) {
        return onError(e);
    }
    console.timeEnd("compile shader");

    // position buffer
    renderer.positionBuffer = renderer.gl.createBuffer();
    renderer.gl.bindBuffer(renderer.gl.ARRAY_BUFFER, renderer.positionBuffer);
    var positions = [-1, 1, 1, 1, -1, -1, 1, -1];
    renderer.gl.bufferData(renderer.gl.ARRAY_BUFFER, new Float32Array(positions), renderer.gl.STATIC_DRAW);

    // rendering
    var time0 = performance.now();
    function render() {
        drawScene();
        renderer.iFrame += 1;
        setTimeout(() => requestAnimationFrame(render), 20);
        var time1 = performance.now();
        // console.log("Total time", (time1-time0).toFixed(2), "ms");
        time0 = time1;
    }
    requestAnimationFrame(render);

    // interactions
    window.addEventListener("resize", function (event) {
    });

}
