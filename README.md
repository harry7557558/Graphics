# Graphics

This repository contains source files for my experiments covering a wide range of topics including rendering, geometry processing, numerical analysis, image processing, deep learning, etc. A majority of them test specific algorithms rather than creating a functioning application or library. (For my projects, check the "Outside this repository" section below.)

An outline of the repository is below (approximately ordered by update recency):

`mapping/`: Related generating 3D data from sensors, including generating 3D models from multi-view images, LiDAR point cloud processing, etc.

`fitting/`: Related to regression tasks, including vectorization, neural representations, generative models, etc.

`image/`: Related to image processing, such as denoising, super-resolution, color replacement,  image stylization, etc.

`modeling/`: Related to geometric modeling, including scripts that generates various 3D models, either procedually or from an image/volume.

`simulation/`: Related to physical simulation, including dynamic ones like 2D and 3D rigid body, fluid, cloth, mass-spring, shallow water; and static ones like optimizing the pose of single rigid bodies and solving for static stress.

`raytracing/`: Related to ray tracing, reflectance models, lighting, and intersectors like BVH, sphere raymarching, and volumetric ray casting.

`triangulate/`: Related to generating triangular meshes, such as from 3D implicit and parametric surfaces and 2D implicit functions.

`numerical/`: Related to numerical analysis, including numerical optimization, solving sparse linear systems, numerical integration of functions and ODEs, etc.

`path/`: Processing 2D curves like polynomial and trigonometric splines, as well as studying the boundary of 2D projection of 3D surfaces.

`UI/`: Includes GUI templates with software rasterization, as well as projects for high school art courses.

`libraries/`: Contains header-only third-party libraries.


# Outside this repository

I have a number of larger projects that appear more useful than scripts of experimental nature in this repository. Many of them use scripts and/or techniques from this repository. Some of them are listed below:

[Spirulae](https://github.com/harry7557558/spirulae): A web-based GPU-accelerated math function grapher focusing on real-time performance and exceptional high quality. It is capable of rendering 3D implicit and parametric surfaces, generating 2D and 3D complex function plots, path tracing denoising, and exporting math functions to download-able 3D models. The tool supports equations with custom variables and functions, complex numbers, special functions, custom colors, automatic differentation, comments, etc.

[Img23d](https://github.com/harry7557558/img23d): A web-based tool that turns 2D images into 3D models by solving Poisson's equation. The produced 3D models have round appearance rather than being a simple extrusion. Supports removing image background within the application.

[SVG to Desmos](https://github.com/harry7557558/svg-to-desmos): Script that turns SVG images into Desmos graphs, with FFT data compression.

For a more comprehensive list of my projects, check [this README](https://github.com/harry7557558/harry7557558.github.io?tab=readme-ov-file#harry7557558s-website). To see more detailed explaination of these projects with visuals, check [List of My Projects](https://github.com/harry7557558/harry7557558/blob/master/list-of-projects.md).
