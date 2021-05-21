import "./common/hotspot3dCommon"
-- ==
-- entry: main
-- compiled random input { [8][512][512]f32 } auto output
-- compiled random input { [128][512][512]f32 }  auto output
entry main arr = hotspot3dCommon.bench_maps arr arr
