# VxDraw - Simple 2D rendering for Rust #


## Introduction ##

`vxdraw` is a rendering library for drawing sprites, text, and streaming textures to a window. It is not an engine, just a library, and has no runtime.

Note: This software is in its early stages and has only been built to satisfy the requirements of a work in progress. There will be some rough edges. Any contributions are very much welcome.

## Features and Motivation ##

Documentation of the API can be found on https://docs.rs/vxdraw.

`vxdraw` is made for streaming changing sprites to the GPU, and is meant mainly for video games with animated, moving, and/or deforming sprites. If no sprite changes are made, the data is not re-uploaded to the GPU for the sake of efficiency.

`vxdraw` packs data tightly to acquire near-optimal GPU-upload performance. Its main organization point is the `layer` - which defines an absolute draw ordering. A layer is a collection of the same type of drawable item, of which there are 4:

 * dyntex - Dynamic Textures, sprites based on a single texture
 * strtex - Streaming Texture, sprites based on a texture of which pixels can be edited
 * quads - Colored 4-point shape
 * text - Font rendering and letter control

And finally a pseudo-layer for debugging

 * debtri - Debug triangle, triangles that are always drawn on top of everything else. Mainly to check if something works or to implement a quick-and-dirty visual tool to check a condition in code.

Further features:

 * Custom blend modes (per-layer)
 * Filter mode (per-layer)
 * Fixed or dynamic perspective matrices (per-layer)

### Snapshot Testing ###
`vxdraw` allows for snapshot testing by retrieving the full frame data.

## Example Outputs ##

* Multiple sprites of the same texture with different opacity settings
![result-1](tests/vxdraw/bunch_of_different_opacity_sprites.png)

* Linear filtering mode on a 3-colored texture
![result-2](tests/vxdraw/linear_filtering_mode.png)

* Various quads
![result-3](tests/vxdraw/quad_mass_manip.png)
