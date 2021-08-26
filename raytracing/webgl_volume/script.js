"use strict";

const VOLUMES = [
    {
        name: "Brain MRI",
        link: "http://paulbourke.net/geometry/polygonise/",
        path: "v_mri_200x160x160_uint8.raw",
        dims: [200, 160, 160],
        ratio: [1, 1, 1],
    },
    {
        name: "Stanford CThead",
        link: "https://graphics.stanford.edu/data/voldata/",
        path: "v_cthead_256x256x113_uint8.raw",
        dims: [256, 256, 113],
        ratio: [1, 1, 2],
    },
    {
        name: "Stanford MRbrain",
        link: "https://graphics.stanford.edu/data/voldata/",
        path: "v_mrbrain_256x109x200_uint8.raw",
        dims: [256, 109, 200],
        ratio: [1, 1.2, 1],
    },
    {
        name: "Visible Female",
        link: "https://mri.radiology.uiowa.edu/visible_human_datasets.html",
        path: "v_visiblefemale_256x256x433_uint8.raw",
        dims: [256, 256, 433],
        ratio: [1, 1, 2],
    },
    {
        name: "Maine Coon Head",
        link: "https://www.embodi3d.com/files/file/41904-maine-coon-ct-head-10-years-old/",
        path: "v_mainecoonhead_220x256x256_uint8.raw",
        dims: [220, 256, 256],
        ratio: [0.75, 0.52, 0.52],
    },
    {
        name: "Dog",
        link: "https://www.embodi3d.com/files/file/39813-dogs-skeleton/",
        path: "v_dog_460x192x128_uint8.raw",
        dims: [460, 192, 128],
        ratio: [1.6, 1.62, 1.62],
    },
    {
        name: "Fish 1",
        link: "https://osf.io/8y6kx/",
        path: "v_fish1_64x384x112_uint8.raw",
        dims: [64, 384, 112],
        ratio: [1, 1, 1],
    },
    {
        name: "Fish 2",
        link: "https://osf.io/j44bk/",
        path: "v_fish2_88x320x144_uint8.raw",
        dims: [88, 320, 144],
        ratio: [1, 1, 1],
    },
    {
        name: "Chameleon",
        link: "https://klacansky.com/open-scivis-datasets/",
        path: "v_chameleon_256x270x256_uint8.raw",
        dims: [256, 270, 256],
        ratio: [1, 1, 1],
    },
    {
        name: "Foot",
        link: "https://klacansky.com/open-scivis-datasets/",
        path: "v_foot_256x256x256_uint8.raw",
        dims: [256, 256, 256],
        ratio: [1, 1, 1],
    },
    {
        name: "Tooth",
        link: "https://klacansky.com/open-scivis-datasets/",
        path: "v_tooth_100x94x160_uint8.raw",
        dims: [100, 94, 160],
        ratio: [1, 1, 1],
    },
    {
        name: "Bonsai",
        link: "https://klacansky.com/open-scivis-datasets/",
        path: "v_bonsai_256x256x256_uint8.raw",
        dims: [256, 256, 256],
        ratio: [1, 1, 1],
    },
    {
        name: "Aneurism",
        link: "https://klacansky.com/open-scivis-datasets/",
        path: "v_aneurism_256x256x256_uint8.raw",
        dims: [256, 256, 256],
        ratio: [1, 1, 1],
    },
    {
        name: "Stag Beetle",
        link: "https://klacansky.com/open-scivis-datasets/",
        path: "v_stagbeetle_208x208x123_uint8.raw",
        dims: [208, 208, 123],
        ratio: [1, 1, 1],
    },
    {
        name: "Lobster",
        link: "https://klacansky.com/open-scivis-datasets/",
        path: "v_lobster_256x256x56_uint8.raw",
        dims: [256, 256, 56],
        ratio: [1, 1, 1.4],
    },
    {
        name: "Beechnut",
        link: "https://klacansky.com/open-scivis-datasets/",
        path: "v_beechnut_256x256x194_uint8.raw",
        dims: [256, 256, 194],
        ratio: [1, 1, 1.5],
    },
    {
        name: "CSAFE Heptane",
        link: "https://klacansky.com/open-scivis-datasets/",
        path: "v_heptane_148x148x148_uint8.raw",
        dims: [148, 148, 148],
        ratio: [1, 1, 1],
    },
    {
        name: "Engine",
        link: "https://klacansky.com/open-scivis-datasets/",
        path: "v_engine_256x256x128_uint8.raw",
        dims: [256, 256, 128],
        ratio: [1, 1, 1],
    },
    {
        name: "Backpack Scan",
        link: "https://klacansky.com/open-scivis-datasets/",
        path: "v_backpack_256x256x186_uint8.raw",
        dims: [256, 256, 186],
        ratio: [0.98, 0.98, 1.25],
    },
    {
        name: "Hydrogen Atom",
        link: "https://klacansky.com/open-scivis-datasets/",
        path: "v_hydrogenatom_128x128x128_uint8.raw",
        dims: [128, 128, 128],
        ratio: [1, 1, 1],
    },
];


var viewport = {
    iRz: 0.9,
    iRx: 0.2,
    iRy: 0.0,
    iSc: 1.3,
    uIso: 0.5,
    renderMode: -1,
    colormap: "",
    renderNeeded: true
};

var texture = {
    id: -1,
    object: {},
    texture: null,
    dims: [1, 1, 1],
    bbox: [0, 0, 0]
};


// ============================ WEBGL ==============================

var requestCache = {};

function loadShaderSource(path) {
    var source = "";
    if (requestCache[path] != undefined) {
        source = requestCache[path];
    }
    else {
        var request = new XMLHttpRequest();
        request.open("GET", path, false);
        request.send(null);
        if (request.status == 200) {
            source = request.responseText;
            requestCache[path] = source;
        }
    }
    source = source.replace("{%uVisual%}", String(viewport.renderMode));
    source = source.replace("{%uColormap%}", viewport.colormap);
    return source;
}

function loadTexture3D(gl, volume) {
    document.getElementById("volume-select").disabled = true;
    document.getElementById("volume-title").innerHTML = "&nbsp;";
    document.getElementById("volume-dims").innerHTML = "Loading...";

    const tex = gl.createTexture();

    var url = volume.path;
    var dims = volume.dims;
    var ratio = volume.ratio;

    function onload(volumeBuffer) {
        gl.bindTexture(gl.TEXTURE_3D, tex);
        gl.texStorage3D(gl.TEXTURE_3D, 1, gl.R8, dims[0], dims[1], dims[2]);

        // set wrapping to clamp to edge
        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);

        // image
        gl.texSubImage3D(gl.TEXTURE_3D, 0, 0, 0, 0,
            dims[0], dims[1], dims[2],
            gl.RED, gl.UNSIGNED_BYTE, volumeBuffer);

        texture.object = volume;
        texture.texture = tex;
        texture.dims = dims;
        var boxL = 2.0 * (ratio[0] * dims[0] + ratio[1] * dims[1] + ratio[2] * dims[2]) / 3.0;
        texture.bbox = [ratio[0] * dims[0] / boxL, ratio[1] * dims[1] / boxL, ratio[2] * dims[2] / boxL];

        document.getElementById("volume-title").innerHTML =
            "<a href='" + volume.link + "' " + (volume.link == "#" ? "" : "target='_blank'") + ">" + volume.name + "</a>";
        document.getElementById("volume-dims").innerHTML = dims.join('×');
        document.getElementById("volume-select").disabled = false;

        viewport.renderNeeded = true;
    }

    function onerror() {
        document.getElementById("volume-dims").innerHTML
            = "<span style='color:red;'>Failed to load volume.</span>";
        document.getElementById("volume-select").disabled = false;
    }

    if (requestCache[url] != undefined) {
        onload(requestCache[url]);
        return;
    }

    var req = new XMLHttpRequest();
    req.open("GET", url, true);
    req.responseType = "arraybuffer";

    req.onprogress = function (e) {
        var tot_bytes = dims[0] * dims[1] * dims[2];
        var loaded_bytes = e.loaded;
        var percent = 100.0 * loaded_bytes / tot_bytes;
        document.getElementById("volume-dims").innerHTML =
            "Loading... (" + percent.toFixed(1) + "%)";
    };
    req.onload = function (e) {
        if (req.status == 200) {
            var volumeBuffer = new Uint8Array(req.response);
            requestCache[url] = volumeBuffer;
            onload(volumeBuffer);
        }
        else {
            onerror();
        }
    };
    req.onerror = function (e) {
        onerror();
    };
    req.send();
}


// initialize WebGL: load and compile shader, initialize buffers
function initWebGL(gl) {

    //console.time("request glsl code");
    var vsSource = loadShaderSource("vs-source.glsl");
    var fsSource = loadShaderSource("fs-source.glsl");
    //console.timeEnd("request glsl code");

    // compile shaders
    function initShaderProgram(gl, vsSource, fsSource) {
        function loadShader(gl, type, source) {
            var shader = gl.createShader(type); // create a new shader
            gl.shaderSource(shader, source); // send the source code to the shader
            gl.compileShader(shader); // compile shader
            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) // check if compiled succeed
                throw new Error(gl.getShaderInfoLog(shader)); // compile error message
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
    //console.time("compile shader");
    var shaderProgram = initShaderProgram(gl, vsSource, fsSource);
    //console.timeEnd("compile shader");

    // position buffer
    var positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    var positions = [-1, 1, 1, 1, -1, -1, 1, -1];
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

    // return a JSON object
    var programInfo = {
        program: shaderProgram,
        attribLocations: { // attribute variables, receive values from buffers
            vertexPosition: gl.getAttribLocation(shaderProgram, 'vertexPosition'),
        },
        uniformLocations: { // uniform variables
            iRz: gl.getUniformLocation(shaderProgram, 'iRz'),
            iRx: gl.getUniformLocation(shaderProgram, 'iRx'),
            iRy: gl.getUniformLocation(shaderProgram, 'iRy'),
            iSc: gl.getUniformLocation(shaderProgram, 'iSc'),
            iResolution: gl.getUniformLocation(shaderProgram, "iResolution"),
            uIso: gl.getUniformLocation(shaderProgram, 'uIso'),
            uBoxRadius: gl.getUniformLocation(shaderProgram, 'uBoxRadius'),
            uSampler3D: gl.getUniformLocation(shaderProgram, "uSampler3D"),
        },
        buffers: {
            positionBuffer: positionBuffer,
        },
    };
    return programInfo;
}


// call this function to re-render
function drawScene(gl, programInfo) {

    // clear the canvas
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clearDepth(-1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.GEQUAL);

    // tell WebGL how to pull out the positions from the position buffer into the vertexPosition attribute
    {
        const numComponents = 2; // pull out 2 values per iteration
        const type = gl.FLOAT; // the data in the buffer is 32bit floats
        const normalize = false; // don't normalize
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
    gl.uniform1f(programInfo.uniformLocations.iRz, viewport.iRz + 1e-4);
    gl.uniform1f(programInfo.uniformLocations.iRx, viewport.iRx + 1e-4);
    gl.uniform1f(programInfo.uniformLocations.iRy, viewport.iRy + 1e-4);
    gl.uniform1f(programInfo.uniformLocations.iSc, viewport.iSc);
    gl.uniform2f(programInfo.uniformLocations.iResolution, canvas.clientWidth, canvas.clientHeight);

    gl.uniform1f(programInfo.uniformLocations.uIso, viewport.uIso);
    gl.uniform3f(programInfo.uniformLocations.uBoxRadius, texture.bbox[0], texture.bbox[1], texture.bbox[2]);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, texture.texture);
    gl.uniform1i(programInfo.uniformLocations.uSampler3D, 0);

    // render
    {
        const offset = 0;
        const vertexCount = 4;
        gl.drawArrays(gl.TRIANGLE_STRIP, offset, vertexCount);
    }
}


// ============================ MAIN ==============================

function getParametersFromHash() {
    try {
        // load hash
        var hash = document.location.hash.replace('#', '');
        if (hash == "") return;
        hash = hash.split('&').reduce(function (res, s) {
            s = s.split('=');
            res[s[0]] = decodeURIComponent(s[1]);
            return res;
        }, {});
        // volume
        if (hash['volume'] != undefined) {
            const selector = document.getElementById("volume-select");
            var volume_id = selector.selectedIndex;
            selector.value = hash['volume'];
            if (selector.selectedIndex == -1) selector.selectedIndex = volume_id;
        }
        // render mode / visual
        if (hash['visual'] != undefined) {
            const selector = document.getElementById("visual-select");
            var visual_id = selector.selectedIndex;
            selector.value = hash['visual'];
            if (selector.selectedIndex == -1) selector.selectedIndex = visual_id;
        }
        // colormap
        if (hash['colormap'] != undefined) {
            const selector = document.getElementById("colormap-select");
            var colormap_id = selector.selectedIndex;
            selector.value = hash['colormap'];
            if (selector.selectedIndex == -1) selector.selectedIndex = colormap_id;
        }
        // uIso
        if (hash['iso'] != undefined) {
            const slider = document.getElementById("iso-slider");
            slider.value = Number(hash['iso']);
        }
    } catch (e) {
        console.error(e);
    }
}

function copyLink() {
    var link = window.location.href + '#';
    link = link.substring(0, link.search('#', 0) + 1);
    var volume = document.getElementById("volume-select").value;
    var visual = document.getElementById("visual-select").value;
    var colormap = document.getElementById("colormap-select").value;
    var iso = Number(document.getElementById("iso-slider").value).toFixed(3);
    link += "volume=" + volume + "&visual=" + visual + "&colormap=" + colormap + "&iso=" + iso;
    console.log(link);

    // https://stackoverflow.com/a/30810322
    function fallbackCopyTextToClipboard(text) {
        var textArea = document.createElement("textarea");
        textArea.value = text;
        textArea.style.top = "0";
        textArea.style.left = "0";
        textArea.style.position = "fixed";
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        try {
            var successful = document.execCommand('copy');
            var msg = successful ? 'successful' : 'unsuccessful';
            alert('Link copy was ' + msg);
        } catch (err) {
            alert('Error copying link: ', err);
        }
        document.body.removeChild(textArea);
    }
    function copyTextToClipboard(text) {
        if (!navigator.clipboard) {
            fallbackCopyTextToClipboard(text);
            return;
        }
        navigator.clipboard.writeText(text).then(function () {
            alert('Link copied to clipboard');
        }, function (err) {
            alert('Error copying link: ', err);
        });
    }

    copyTextToClipboard(link);
}


function main() {
    const canvas = document.getElementById("canvas");
    const gl = canvas.getContext("webgl2");

    // load volume selectors
    {
        var select = document.getElementById("volume-select");
        const selected = 0;
        for (var i = 0; i < VOLUMES.length; i++) {
            var v = VOLUMES[i];
            var mb = v.dims[0] * v.dims[1] * v.dims[2] / 1048576;
            var title = v.name + " (" + mb.toFixed(2) + " MB)";
            var value = v.name.toLowerCase().replace(/\s/g, '-');
            select.innerHTML += "<option value='" + value + "' "
                + (i == selected ? "selected" : "") + ">" + title + "</option>";
        }
    }
    getParametersFromHash();

    // input
    var updateParameters = function (e) {
        // texture id
        var volume_id = document.getElementById("volume-select").selectedIndex;
        if (texture.id != volume_id) {
            loadTexture3D(gl, VOLUMES[volume_id]);
            texture.id = volume_id;
        }
        // visual/rendering
        viewport.renderMode = document.getElementById("visual-select").selectedIndex;
        viewport.colormap = document.getElementById("colormap-select").value;
        // uIso
        var iso = Number(document.getElementById("iso-slider").value);
        document.getElementById("iso-val").innerHTML = iso.toFixed(3);
        viewport.uIso = iso;
        viewport.renderNeeded = true;
    };
    updateParameters();
    document.getElementById("iso-slider").addEventListener("input", updateParameters);
    document.getElementById("volume-select").addEventListener("input", updateParameters);
    document.getElementById("visual-select").addEventListener("input", function (e) {
        updateParameters(e);
        programInfo = initWebGL(gl);
    });
    document.getElementById("colormap-select").addEventListener("input", function (e) {
        updateParameters(e);
        programInfo = initWebGL(gl);
    });
    window.addEventListener("hashchange", function (e) {
        getParametersFromHash();
        updateParameters(e);
        programInfo = initWebGL(gl);
    });

    // load WebGL
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
    canvas.addEventListener("wheel", function (e) {
        e.preventDefault();
        var sc = Math.exp(0.0002 * e.wheelDeltaY);
        viewport.iSc *= sc;
        viewport.renderNeeded = true;
    }, { passive: false });
    var mouseDown = false;
    canvas.addEventListener("pointerdown", function (event) {
        //event.preventDefault();
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
        } catch (e) {
            console.error(e);
            document.body.innerHTML = "<h1 style='color:red;'>" + e + "</h1>";
        }
    }, 0);
};