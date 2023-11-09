
window.ModelExporter = {
    name: "model",
    assertModelNonempty: function() {
        let isEmpty = Module.ccall('isModelEmpty', 'int', [], []);
        if (isEmpty) {
            alert("Model is empty.");
            return true;
        }
        return false;
    },
    downloadFile: function(ptr, filename, type) {
        let size = Module.ccall('getFileSize', 'int', [], []);
        let resultArray = new Uint8Array(Module.HEAPU8.buffer, ptr, size);
        let resultBuffer = resultArray.slice().buffer;
        console.log(size + " bytes");
        let link = document.createElement('a');
        link.href = URL.createObjectURL(new Blob([resultBuffer], { type: type }));
        link.download = filename;
        link.click();
        URL.revokeObjectURL(link.href);
    },
    downloadSTL: function() {
        if (ModelExporter.assertModelNonempty()) return;
        let ptr = Module.ccall('generateSTL', 'int', [], []);
        ModelExporter.downloadFile(ptr, ModelExporter.name+'.stl', 'model/stl');
    },
    downloadPLY: function() {
        if (ModelExporter.assertModelNonempty()) return;
        let ptr = Module.ccall('generatePLY', 'int', [], []);
        ModelExporter.downloadFile(ptr, ModelExporter.name+'.ply', 'model/stl');
    },
    downloadOBJ: function() {
        if (ModelExporter.assertModelNonempty()) return;
        let ptr = Module.ccall('generateOBJ', 'int', [], []);
        ModelExporter.downloadFile(ptr, ModelExporter.name+'.obj', 'model/obj');
    },
    downloadGLB: function() {
        if (ModelExporter.assertModelNonempty()) return;
        let ptr = Module.ccall('generateGLB', 'int', [], []);
        ModelExporter.downloadFile(ptr, ModelExporter.name+'.glb', 'model/gltf-binary');
    },
    init: function() {
        document.getElementById("export-stl").onclick = ModelExporter.downloadSTL;
        document.getElementById("export-ply").onclick = ModelExporter.downloadPLY;
        document.getElementById("export-obj").onclick = ModelExporter.downloadOBJ;
        document.getElementById("export-glb").onclick = ModelExporter.downloadGLB;
    }
};

function onError(message) {
    let errorMessage = document.getElementById("error-message");
    errorMessage.style.display = "inline-block";
    errorMessage.innerHTML = message;
}

function initDragDrop() {
    let dropArea = document.getElementById("control");
    let img = new Image();
    let onerror = function() {
        onError("error: unsupported image format");
    };
    function imgOnload(img_src) {
        let canvas = document.createElement("canvas");
        if (img.width == 0 || img.height == 0)
            return onerror();
        canvas.width = img.width == 0 ? 1024 : img.width;
        canvas.height = img.height == 0 ? 1024 : img.height;
        let ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        let pixelData = imageData.data;
        let heapSpace = Module._malloc(pixelData.length);
        Module.HEAP8.set(pixelData, heapSpace);
        var name = img_src.split('/');
        name = name[name.length - 1].split('.');
        name = name.slice(0, name.length-1).join('.');
        ModelExporter.name = name;
        Module.ccall('updateImage',
            null, ['string', 'number', 'number', 'number'],
            [name, canvas.width, canvas.height, heapSpace]);
    }
    img.onload = function(e) { imgOnload(img.src); }
    img.src = "hermit_crab.svg";

    window.addEventListener("dragover", function(e) {
        e.preventDefault();
        dropArea.style.backgroundColor = "yellow";
    });
    window.addEventListener("dragleave", function(e) {
        e.preventDefault();
        dropArea.style.backgroundColor = null;
    });
    window.addEventListener("drop", function(e) {
        e.preventDefault();
        let errorMessage = document.getElementById("error-message");
        errorMessage.style.display = "none";
        dropArea.style.backgroundColor = null;
        let file = e.dataTransfer.files[0];
        if (file && (
            file.type == "image/png" ||
            file.type == "image/jpeg" ||
            file.type == "image/gif" ||
            file.type == "image/webp" ||
            file.type == "image/svg+xml"
            )) {
            let reader = new FileReader();
            reader.onload = function(event) {
                img = new Image();
                img.onload = function(e) { imgOnload(file.name); };
                img.onerror = onerror;
                img.src = event.target.result;
            }
            reader.onerror = onerror;
            reader.readAsDataURL(file);
        }
        else onerror();
    });
}

function initInteraction() {
    let canvas = document.getElementById("emscripten-canvas");
    function onresize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        canvas.style.width = canvas.width + "px";
        canvas.style.height = canvas.height + "px";
        Module.ccall('resizeWindow',
            null,
            ['number', 'number'],
            [canvas.width, canvas.height]);
    };
    window.addEventListener("resize", onresize);
    onresize();
}

function initConfig() {
    let checkboxEdge = document.getElementById("checkbox-edge");
    let checkboxNormal = document.getElementById("checkbox-normal");
    let checkboxDoubleSided = document.getElementById("checkbox-double-sided");
    let checkboxTexture = document.getElementById("checkbox-texture");
    function updateCheckboxes() {
        Module.ccall('setMeshEdge', null, ['int'], [checkboxEdge.checked]);
        Module.ccall('setMeshNormal', null, ['int'], [checkboxNormal.checked]);
        Module.ccall('setMeshDoubleSided', null, ['int'], [checkboxDoubleSided.checked]);
        Module.ccall('setMeshTexture', null, ['int'], [checkboxTexture.checked]);
    }
    checkboxEdge.addEventListener("input", updateCheckboxes);
    checkboxNormal.addEventListener("input", updateCheckboxes);
    checkboxDoubleSided.addEventListener("input", updateCheckboxes);
    checkboxTexture.addEventListener("input", updateCheckboxes);
    updateCheckboxes();
}

function onReady() {
    initInteraction();
    initDragDrop();
    initConfig();
    ModelExporter.init();
}

