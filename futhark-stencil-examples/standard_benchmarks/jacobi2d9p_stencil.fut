import "./common/jacobi2dCommon"
-- ==
-- entry: main
-- compiled random input { [4096][4096]f32 } auto output
-- compiled random input { [8192][8192]f32 } auto output
entry main = jacobi2dCommon.bench_9p_stencil

