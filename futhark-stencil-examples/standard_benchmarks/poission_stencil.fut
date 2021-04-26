import "./common/poissonCommon"
-- ==
-- entry: main
-- compiled random input { [256][256][256]f32 } auto output
-- compiled random input { [512][512][512]f32 } auto output
entry main = poissonCommon.bench_stencil
