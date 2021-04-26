import "./common/hotspot3dCommon"
-- ==
-- entry: main
-- compiled random input { [8][512][512]f32 [8][512][512]f32 } auto output
-- compiled random input { [128][512][512]f32 [128][512][512]f32 }  auto output
entry main = hotspot3dCommon.bench_maps
