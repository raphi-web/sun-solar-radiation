Overview: Port the grass module r.sun to rust. The project contains r.sun..cpp which is the original c++ code and r.sun.daily.py which is a python wrapper for the c++ code.

1. Focus on single threaded execution.
2. Use gdal to read the input rasters.
3. Use gdal to write the output rasters.
4. Create a dummy raster to test the code.
5. Compare the output with the original grass function output
6. Do use the rust librarys: PyO3 and GDAL
7. After the rust code is working and tested use webgpu in the rust code to accelerate the calculations.
8. Test the webgpu code on dummy raster and compare the results with the original grass code.
9. Add a extra argument to the radiation computation, a raster - mask geotiff, the mask has true and false values. True pixels are the target pixels for which the radiation should be computed, for false pixels the radiation computation is not needed.
10. Test the code with the mask raster.
11. Implement Quiet flag with no output to stdout.
12. Parralize the cpu code with rayon.
