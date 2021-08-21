"use strict";


var viewport = {
    iRx: 0.2,
    iRz: 0.9,
    iRy: 0.1,
    iDist: 3.0,
    renderNeeded: true
};

// generate a grid for sampling terrain heightmap
function generateGrid(p0, p1, dif) {
    var row_size = dif[0] + 1;
    var vertices = new Array(3 * (dif[0] + 1) * (dif[1] + 1));
    for (var yi = 0; yi <= dif[1]; yi++) {
        for (var xi = 0; xi <= dif[0]; xi++) {
            var xf = xi / dif[0], yf = yi / dif[1];
            var px = p0[0] + (p1[0] - p0[0]) * xf;
            var py = p0[1] + (p1[1] - p0[1]) * yf;
            var i = 3 * (yi * row_size + xi);
            vertices[i] = px;
            vertices[i + 1] = py;
            vertices[i + 2] = 0;
        }
    }
    var indices = new Array(6 * dif[0] * dif[1]);
    for (var yi = 0; yi < dif[1]; yi++) {
        for (var xi = 0; xi < dif[0]; xi++) {
            var i = 6 * (yi * dif[0] + xi);
            indices[i] = yi * row_size + xi;
            indices[i + 1] = yi * row_size + (xi + 1);
            indices[i + 2] = (yi + 1) * row_size + xi;
            indices[i + 3] = (yi + 1) * row_size + (xi + 1);
            indices[i + 4] = (yi + 1) * row_size + xi;
            indices[i + 5] = yi * row_size + (xi + 1);
        }
    }
    return {
        vertices: vertices,
        indices: indices
    }
}

// initialize WebGL: load and compile shader, initialize buffers
function initWebGL(gl) {

    // load vertex and fragment shader sources
    function loadGLSLSource(path) {
        var request = new XMLHttpRequest();
        request.open("GET", path, false);
        request.send(null);
        if (request.status == 200) {
            return request.responseText;
        }
        return "";
    }
    console.time("request glsl code");
    var general = loadGLSLSource("noise.glsl") + loadGLSLSource("terrain.glsl");
    var vsSource = general + loadGLSLSource("vs-source.glsl");
    var fsSource = general + loadGLSLSource("fs-source.glsl");
    console.timeEnd("request glsl code");

    // compile shaders
    function initShaderProgram(gl, vsSource, fsSource) {
        function loadShader(gl, type, source) {
            var shader = gl.createShader(type);  // create a new shader
            gl.shaderSource(shader, source);  // send the source code to the shader
            gl.compileShader(shader);  // compile shader
            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))  // check if compiled succeed
                throw new Error(gl.getShaderInfoLog(shader));  // compile error message
            return shader;
        }
        var vShader = loadShader(gl, gl.VERTEX_SHADER, vsSource);
        var fShader = loadShader(gl, gl.FRAGMENT_SHADER, fsSource);
        // create the shader program
        var shaderProgram = gl.createProgram();
        gl.attachShader(shaderProgram, vShader);
        gl.attachShader(shaderProgram, fShader);
        gl.linkProgram(shaderProgram);
        // if creating shader program failed
        if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS))
            throw new Error(gl.getProgramInfoLog(shaderProgram));
        return shaderProgram;
    }
    console.time("compile shader");
    var shaderProgram = initShaderProgram(gl, vsSource, fsSource);
    console.timeEnd("compile shader");

    // initialize position buffer and indice buffer
    var grid = generateGrid([-6, -6], [6, 6], [600, 600]);

    // position buffer
    var positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    var positions = grid.vertices;
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

    // indice buffer
    // to use 32 bit indice: https://computergraphics.stackexchange.com/questions/3637/how-to-use-32-bit-integers-for-element-indices-in-webgl-1-0
    if (gl.getExtension("OES_element_index_uint") == null)
        throw "Error: Unable to use 32 bit integer";
    const indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    const indices = grid.indices;
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(indices), gl.STATIC_DRAW);

    // return a JSON object
    var programInfo = {
        program: shaderProgram,
        attribLocations: {  // attribute variables, receive values from buffers
            vertexPosition: gl.getAttribLocation(shaderProgram, 'aVertexPosition'),
        },
        uniformLocations: {  // uniform variables
            iRx: gl.getUniformLocation(shaderProgram, 'iRx'),
            iRz: gl.getUniformLocation(shaderProgram, 'iRz'),
            iRy: gl.getUniformLocation(shaderProgram, 'iRy'),
            iDist: gl.getUniformLocation(shaderProgram, 'iDist'),
            iResolution: gl.getUniformLocation(shaderProgram, "iResolution"),
        },
        // buffers
        buffers: {
            positionBuffer: positionBuffer,
            indiceBuffer: indexBuffer,
        },
        vertexCount: indices.length
    };
    return programInfo;
}


// call this function to re-render
function drawScene(gl, programInfo) {

    // clear the canvas
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.7, 0.89, 0.95, 1);
    gl.clearDepth(-1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.GEQUAL);

    // tell WebGL how to pull out the positions from the position buffer into the vertexPosition attribute
    {
        const numComponents = 3;  // pull out 2 values per iteration
        const type = gl.FLOAT;  // the data in the buffer is 32bit floats
        const normalize = false;  // don't normalize
        const stride = 0; // how many bytes to get from one set of values to the next
        const offset = 0; // how many bytes inside the buffer to start from
        gl.bindBuffer(gl.ARRAY_BUFFER, programInfo.buffers.positionBuffer);
        gl.vertexAttribPointer(
            programInfo.attribLocations.vertexPosition,
            numComponents, type, normalize, stride, offset);
        gl.enableVertexAttribArray(programInfo.attribLocations.vertexPosition);
    }

    // indice buffer
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, programInfo.buffers.indiceBuffer);

    // make sure it uses the program
    gl.useProgram(programInfo.program);

    // set shader uniforms
    // https://webglfundamentals.org/webgl/lessons/webgl-shaders-and-glsl.html
    gl.uniform1f(programInfo.uniformLocations.iRx, viewport.iRx);
    gl.uniform1f(programInfo.uniformLocations.iRz, viewport.iRz);
    gl.uniform1f(programInfo.uniformLocations.iRy, viewport.iRy);
    gl.uniform1f(programInfo.uniformLocations.iDist, viewport.iDist);
    gl.uniform2f(programInfo.uniformLocations.iResolution, canvas.clientWidth, canvas.clientHeight);

    // render
    {
        const vertexCount = programInfo.vertexCount;
        const type = gl.UNSIGNED_INT;
        const offset = 0;
        gl.drawElements(gl.TRIANGLES, vertexCount, type, offset);
    }
}


// ============================ MAIN ==============================

function main() {

    // load WebGL
    const canvas = document.getElementById("canvas");
    const gl = canvas.getContext("webgl");
    var programInfo = initWebGL(gl);

    // rendering
    let then = 0;
    function render_main(now) {
        if (viewport.renderNeeded) {
            // display fps
            now *= 0.001;
            var time_delta = now - then;
            then = now;
            if (time_delta != 0) {
                document.getElementById("fps").textContent = (1.0 / time_delta).toFixed(1) + " fps";
            }

            canvas.width = canvas.style.width = window.innerWidth;
            canvas.height = canvas.style.height = window.innerHeight;
            drawScene(gl, programInfo);

            viewport.renderNeeded = false;
        }
        requestAnimationFrame(render_main);
    }
    requestAnimationFrame(render_main);


    // interactions
    canvas.oncontextmenu = function (e) {
        //e.preventDefault();
    };
    canvas.addEventListener("wheel", function (e) {
        e.preventDefault();
        var sc = Math.exp(-0.0002 * e.wheelDeltaY);
        viewport.iDist *= sc;
        viewport.renderNeeded = true;
    }, { passive: false });

    var mouseDown = false;
    canvas.addEventListener("pointerdown", function (event) {
        event.preventDefault();
        mouseDown = true;
    });
    window.addEventListener("pointerup", function (event) {
        event.preventDefault();
        mouseDown = false;
    });
    window.addEventListener("resize", function (event) {
        canvas.width = canvas.style.width = window.innerWidth;
        canvas.height = canvas.style.height = window.innerHeight;
        viewport.renderNeeded = true;
    });
    canvas.addEventListener("pointermove", function (e) {
        if (mouseDown) {
            viewport.iRx += 0.01 * e.movementY;
            viewport.iRz -= 0.01 * e.movementX;
            viewport.renderNeeded = true;
        }
    });

}

window.onload = function (event) {
    setTimeout(function () {
        try {
            main();
        }
        catch (e) {
            console.error(e);
            document.body.innerHTML = "<h1 style='color:red;'>" + e + "</h1>";
        }
    }, 0);
};
